#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VLA MaaS Studio: browser console for debugging a VLA MaaS deployment.

The studio runs the simulators server-side, opens one OpenPI websocket per
robot, and streams JPEG frames plus telemetry to a browser. The API key stays
on the server; the browser never talks to the gateway directly.

Usage:
    python3 apps/vla-maas-studio/studio.py \
        --gateway http://127.0.0.1:9000 --api-key dev-key-change-me
    # open http://127.0.0.1:9100

    # Standard Robot protocol (maas-robot/1) with a robot file from
    # packages/vla-maas-robot: a MuJoCo twin by default, the real arm with --hardware.
    python3 apps/vla-maas-studio/studio.py \
        --robot-config packages/vla-maas-robot/robots/piper.yaml [--hardware]

Requires the client extras: websockets, msgpack, msgpack-numpy, numpy, pillow
and openpi-client. Real LIBERO scenes additionally need LIBERO/robosuite.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import struct
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit

from runner import FRAME_KIND_DETAIL, MAX_ROBOTS, Run, parse_run_config, resolve_endpoint
from scenes import build_catalog
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

logger = logging.getLogger("vla_maas_studio")

STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_GATEWAY = "http://127.0.0.1:9000"
MIME_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}


class Client:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.watched: set[int] = set()

    def push(self, item) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            pass


class Studio:
    def __init__(self, args: argparse.Namespace, catalog: dict, robot_cfg=None) -> None:
        self.args = args
        self.catalog = catalog
        # Set with --robot-config: runs speak the standard Robot protocol (robot_run.py).
        self.robot_cfg = robot_cfg
        self.clients: set[Client] = set()
        self.run = None  # runner.Run or robot_run.ActRun
        self.run_lock = asyncio.Lock()

    # -- catalog / gateway helpers -------------------------------------------

    def gateway_url(self) -> str:
        if self.robot_cfg is not None:
            from robot_run import gateway_http_url

            return gateway_http_url(self.args.gateway, self.robot_cfg)
        url = self.args.gateway or DEFAULT_GATEWAY
        return url.rstrip("/")

    def api_key(self) -> str | None:
        if self.args.api_key:
            return self.args.api_key
        return self.robot_cfg.gateway.api_key if self.robot_cfg is not None else None

    def catalog_public(self) -> dict:
        common = {
            **self.catalog,
            "gateway": {"url": self.gateway_url(), "has_key": bool(self.api_key())},
            "limits": {"max_robots": int(self.catalog.get("max_robots", MAX_ROBOTS))},
        }
        if self.robot_cfg is not None:
            hardware = self.catalog["robot_config"]["hardware"]
            return {
                **common,
                "defaults": {
                    "robots": 1,
                    "duration_s": 120,
                    "control_hz": 0,
                    "prefetch": -1,
                    "frame_fps": 5,
                    "episodes": 1,
                    "max_steps": 600,
                    "instruction": "pick up the red cube",
                    "mode": self.robot_cfg.safety.start_mode if hardware else "live",
                },
            }
        return {
            **common,
            "defaults": {
                "robots": 8,
                "duration_s": 30,
                "control_hz": 50,
                "replan_steps": 20,
                "prefetch": 10,
                "resolution": 256,
                "frame_fps": 2,
                "base_camera_key": "observation.images.image",
                "wrist_camera_key": "observation.images.image2",
                "action_scale": 0.005,
                "seed": 0,
                "report_episodes": True,
            },
        }

    def _fetch(self, path: str) -> dict:
        if not self.gateway_url():
            raise RuntimeError("no --gateway configured")
        request = urllib.request.Request(self.gateway_url() + path)
        if self.api_key():
            request.add_header("Authorization", f"Api-Key {self.api_key()}")
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"gateway returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"gateway unreachable: {exc.reason}") from exc

    def watched_union(self) -> set[int]:
        watched: set[int] = set()
        for client in self.clients:
            watched |= client.watched
        return watched

    async def refresh_watched(self) -> None:
        if self.run is not None:
            self.run.set_watched(self.watched_union())

    # -- run control ----------------------------------------------------------

    def _act_models(self) -> dict:
        """Models served on /act. A robot key cannot list models, so fall back to the robot file's."""
        try:
            ids = [model["id"] for model in self._fetch("/v1/models").get("models", [])]
        except RuntimeError:
            ids = []
        ids = list(dict.fromkeys([self.robot_cfg.gateway.model, *ids]))
        models, errors = [], []
        for model_id in ids:
            try:
                info = self._fetch(f"/v1/models/{model_id}")
            except RuntimeError as exc:
                errors.append(f"{model_id}: {exc}")
                continue
            if info.get("act"):
                models.append(info)
        result = {"models": models}
        if not models and errors:
            result["error"] = "; ".join(errors)
        return result

    async def start_act_run(self, payload: dict) -> dict:
        from robot_run import ActRun, parse_act_config

        config = parse_act_config(payload, self.catalog)
        # Fail fast with the gateway's answer (this route accepts robot keys too).
        info = await asyncio.to_thread(self._fetch, f"/v1/models/{config.model}")
        if not info.get("act"):
            raise RuntimeError(f"model {config.model!r} has no adapter for the standard Robot protocol (/act)")
        async with self.run_lock:
            previous = self.run
            if previous is not None and previous.running:
                await asyncio.to_thread(previous.stop)
            run = ActRun(
                config,
                self.robot_cfg,
                gateway=self.gateway_url(),
                api_key=self.api_key(),
                hardware=self.catalog["robot_config"]["hardware"],
                menagerie=self.args.menagerie,
            )
            run.set_watched(self.watched_union())
            await asyncio.to_thread(run.start)
            self.run = run
            logger.info(
                "run %s started: %d x %s on %s, mode=%s, physics=%s",
                run.id,
                config.robots,
                self.robot_cfg.robot_id,
                config.model,
                config.mode,
                self.catalog["physics"],
            )
            return run.snapshot()

    async def start_run(self, payload: dict) -> dict:
        if self.robot_cfg is not None:
            return await self.start_act_run(payload)
        config = parse_run_config(payload, self.catalog)
        config.menagerie = self.args.menagerie
        endpoint = resolve_endpoint(config, self.gateway_url(), self.args.api_key)
        if config.target == "gateway":
            # Fail fast (and with the gateway's error) instead of letting N
            # sessions each sit in a 10 s websocket connect timeout.
            models = await asyncio.to_thread(self._fetch, "/v1/models")
            served = {model["id"] for model in models.get("models", [])}
            if config.model not in served:
                raise RuntimeError(f"gateway does not serve model {config.model!r}; available: {sorted(served)}")
        async with self.run_lock:
            previous = self.run
            if previous is not None and previous.running:
                await asyncio.to_thread(previous.stop)
            run = Run(config, endpoint)
            run.set_watched(self.watched_union())
            await asyncio.to_thread(run.start)
            self.run = run
            logger.info(
                "run %s started: %d robots, suite=%s, target=%s, model=%s",
                run.id,
                config.robots,
                config.suite,
                config.target,
                config.model or config.direct_url,
            )
            return run.snapshot()

    async def stop_run(self) -> dict | None:
        async with self.run_lock:
            run = self.run
            if run is None:
                return None
            await asyncio.to_thread(run.stop)
            logger.info("run %s stopped", run.id)
            return run.snapshot()

    async def handle_message(self, client: Client, message: dict) -> None:
        kind = message.get("type")
        if kind == "start":
            try:
                snapshot = await self.start_run(message.get("config") or {})
            except (ValueError, RuntimeError) as exc:
                client.push(json.dumps({"type": "error", "message": str(exc)}))
                return
            client.push(json.dumps({"type": "run_started", "id": snapshot["id"]}))
        elif kind == "stop":
            snapshot = await self.stop_run()
            if snapshot is not None:
                client.push(json.dumps({"type": "run_stopped", "id": snapshot["id"]}))
        elif kind == "watch":
            try:
                session_id = int(message.get("session", -1))
            except (TypeError, ValueError):
                return
            if message.get("on", True):
                client.watched.add(session_id)
            else:
                client.watched.discard(session_id)
            await self.refresh_watched()
        elif kind == "models":
            try:
                fetch = self._act_models if self.robot_cfg is not None else lambda: self._fetch("/v1/models")
                models = await asyncio.to_thread(fetch)
                client.push(json.dumps({"type": "models", **models}))
            except RuntimeError as exc:
                client.push(json.dumps({"type": "models", "models": [], "error": str(exc)}))
        elif kind == "gateway_metrics":
            try:
                metrics = await asyncio.to_thread(self._fetch, "/metrics")
                client.push(json.dumps({"type": "gateway_metrics", "metrics": metrics}))
            except RuntimeError as exc:
                client.push(json.dumps({"type": "gateway_metrics", "metrics": None, "error": str(exc)}))
        elif kind == "ping":
            client.push(json.dumps({"type": "pong"}))


async def _writer(connection: ServerConnection, client: Client) -> None:
    try:
        while True:
            item = await client.queue.get()
            await connection.send(item)
    except (ConnectionClosed, asyncio.CancelledError):
        pass


async def _broadcaster(studio: Studio) -> None:
    tick = 0
    while True:
        await asyncio.sleep(0.1)
        run = studio.run
        if run is None:
            continue
        tick += 1
        # 5 Hz while sessions run; 1 Hz afterwards so the final state stays
        # visible without pinning the socket at full rate.
        if tick % (2 if run.running else 10) == 0:
            payload = json.dumps({"type": "telemetry", **run.snapshot()})
            for client in list(studio.clients):
                client.push(payload)
        for frame_kind, session_id, jpeg in run.drain_frames():
            if not studio.clients:
                break
            frame = struct.pack("!II", frame_kind, session_id) + jpeg
            for client in list(studio.clients):
                if frame_kind == FRAME_KIND_DETAIL and session_id not in client.watched:
                    continue
                client.push(frame)


def _json_response(status: HTTPStatus, body: dict) -> Response:
    payload = json.dumps(body).encode()
    headers = Headers(
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(payload))),
            ("Cache-Control", "no-store"),
        ]
    )
    return Response(status.value, status.phrase, headers, payload)


def _file_response(name: str) -> Response:
    path = STATIC_DIR / name
    if not path.is_file():
        return _json_response(HTTPStatus.NOT_FOUND, {"error": f"missing static file {name}"})
    payload = path.read_bytes()
    headers = Headers(
        [
            ("Content-Type", MIME_TYPES.get(name, "application/octet-stream")),
            ("Content-Length", str(len(payload))),
            ("Cache-Control", "no-store"),
        ]
    )
    return Response(HTTPStatus.OK.value, HTTPStatus.OK.phrase, headers, payload)


def make_process_request(studio: Studio):
    def process_request(_: ServerConnection, request: Request) -> Response | None:
        path = urlsplit(request.path).path
        if path == "/ws":
            return None
        if path in ("/", "/index.html"):
            return _file_response("index.html")
        if path in ("/app.js", "/style.css"):
            return _file_response(path.lstrip("/"))
        if path == "/healthz":
            return _json_response(HTTPStatus.OK, {"status": "ok"})
        if path == "/api/catalog":
            return _json_response(HTTPStatus.OK, studio.catalog_public())
        return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown path {path}"})

    return process_request


async def _main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    robot_cfg = None
    if args.robot_config:
        from robot_run import build_robot_catalog
        from vla_maas_robot.config import load_robot_config

        robot_cfg = load_robot_config(args.robot_config)
        catalog = build_robot_catalog(robot_cfg, args.hardware, args.menagerie)
    elif args.hardware:
        raise SystemExit("--hardware needs --robot-config")
    else:
        catalog = build_catalog(args.physics, args.menagerie)
    studio = Studio(args, catalog, robot_cfg)

    async def handler(connection: ServerConnection) -> None:
        client = Client()
        studio.clients.add(client)
        writer = asyncio.create_task(_writer(connection, client))
        try:
            await connection.send(json.dumps({"type": "hello", "catalog": studio.catalog_public()}))
            async for raw in connection:
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    try:
                        await studio.handle_message(client, message)
                    except Exception as exc:  # keep one bad message from closing the console
                        logger.exception("message %r failed", message.get("type"))
                        client.push(json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}))
        except ConnectionClosed:
            pass
        finally:
            studio.clients.discard(client)
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                await writer
            await studio.refresh_watched()

    async with serve(
        handler,
        args.host,
        args.port,
        process_request=make_process_request(studio),
        compression=None,
        max_size=None,
    ) as server:
        broadcaster = asyncio.create_task(_broadcaster(studio))
        logger.info(
            "studio on http://%s:%d (physics=%s, gateway=%s)",
            args.host,
            args.port,
            catalog["physics"],
            studio.gateway_url() or "none",
        )
        try:
            await server.serve_forever()
        finally:
            broadcaster.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await broadcaster
            if studio.run is not None:
                await asyncio.to_thread(studio.run.stop)


def main() -> int:
    parser = argparse.ArgumentParser(description="Browser console for a VLA MaaS deployment")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (keep it on localhost).")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--gateway",
        default=None,
        help=f"Gateway URL (default {DEFAULT_GATEWAY}, or gateway.url of --robot-config).",
    )
    parser.add_argument("--api-key", default=os.environ.get("VLA_MAAS_API_KEY"))
    parser.add_argument("--physics", default="mock", choices=["mock", "libero", "mujoco"])
    parser.add_argument(
        "--menagerie",
        default=os.environ.get("MUJOCO_MENAGERIE", "~/.cache/mujoco_menagerie"),
        help="MuJoCo Menagerie checkout (used with --physics mujoco).",
    )
    parser.add_argument(
        "--robot-config",
        default=None,
        help="Robot YAML from packages/vla-maas-robot: run it on the standard Robot protocol (/act).",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="With --robot-config: use the file's real driver and cameras instead of the MuJoCo twin.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
