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
    def __init__(self, args: argparse.Namespace, catalog: dict) -> None:
        self.args = args
        self.catalog = catalog
        self.clients: set[Client] = set()
        self.run: Run | None = None
        self.run_lock = asyncio.Lock()

    # -- catalog / gateway helpers -------------------------------------------

    def gateway_url(self) -> str:
        url = self.args.gateway
        return url.rstrip("/") if url else ""

    def catalog_public(self) -> dict:
        return {
            **self.catalog,
            "gateway": {"url": self.gateway_url(), "has_key": bool(self.args.api_key)},
            "limits": {"max_robots": int(self.catalog.get("max_robots", MAX_ROBOTS))},
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
        if not self.args.gateway:
            raise RuntimeError("no --gateway configured")
        request = urllib.request.Request(self.gateway_url() + path)
        if self.args.api_key:
            request.add_header("Authorization", f"Api-Key {self.args.api_key}")
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

    async def start_run(self, payload: dict) -> dict:
        config = parse_run_config(payload, self.catalog)
        config.menagerie = self.args.menagerie
        endpoint = resolve_endpoint(config, self.args.gateway, self.args.api_key)
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
                models = await asyncio.to_thread(self._fetch, "/v1/models")
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
    catalog = build_catalog(args.physics, args.menagerie)
    studio = Studio(args, catalog)

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
            args.physics,
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
    parser.add_argument("--gateway", default="http://127.0.0.1:9000")
    parser.add_argument("--api-key", default=os.environ.get("VLA_MAAS_API_KEY"))
    parser.add_argument("--physics", default="mock", choices=["mock", "libero", "mujoco"])
    parser.add_argument(
        "--menagerie",
        default=os.environ.get("MUJOCO_MENAGERIE", "~/.cache/mujoco_menagerie"),
        help="MuJoCo Menagerie checkout (used with --physics mujoco).",
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
