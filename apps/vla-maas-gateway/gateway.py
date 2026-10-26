#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MaaS gateway in front of VLA policy servers (vllm-omni or any OpenPI server).

Two robot-facing routes, both with ``Authorization: Api-Key <key>``:

* ``ws://<gateway>/v1/robot/<model>/act?robot_id=<id>[&embodiment=<e>]``: the
  standard Robot protocol (``vla_maas_protocol``). Robots send model-agnostic
  frames; the model's ``adapter`` (``model_adapters.py``) translates them to the
  backend's native OpenPI frames and back. Robots never see model keys.
* ``ws://<gateway>/v1/robot/<model>/openpi``: the native OpenPI relay. Frames
  pass through one-to-one; the handshake metadata is the backend's, plus
  ``model`` and ``backend`` so clients can log what served them.

Both pick the least-loaded backend for the model per session.

Clients may send ``{"endpoint": "episode_result", "success": bool}`` on the
same socket after an episode; the gateway records it for the dashboard, answers
``{"status": "ok"}`` and does not forward it.

Keys: ``api_keys`` are operator keys (every route). ``robots.<id>.api_key`` (or
``api_key_env``) is valid only on ``/act`` for that ``robot_id`` and its
``models``.

HTTP endpoints on the same port:
    GET /healthz                   liveness
    GET /dashboard                 live demo panel (polls /metrics with the key typed into it)
    GET /v1/models                 configured models and backend counts (operator key)
    GET /v1/models/<model>         what the model requires and provides on /act (any valid key)
    GET /v1/robots                 robots connected on /act (operator key)
    GET /v1/robots/<id>/stop       answer that robot's next observation with ``stop`` (operator key).
                                   GET because the websockets HTTP server accepts no other method.
    GET /metrics                   per-model request counts, errors and relay latency (operator key)

Usage:
    python gateway.py --config gateway.example.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import vla_maas_protocol as proto
import yaml
from model_adapters import ModelAdapter, make_adapter
from openpi_client import msgpack_numpy
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

logger = logging.getLogger("vla_maas_gateway")

ROBOT_PATH = re.compile(r"^/v1/robot/(?P<model>[A-Za-z0-9._-]+)/openpi$")
ACT_PATH = re.compile(r"^/v1/robot/(?P<model>[A-Za-z0-9._-]+)/act$")
MODEL_PATH = re.compile(r"^/v1/models/(?P<model>[A-Za-z0-9._-]+)$")
STOP_PATH = re.compile(r"^/v1/robots/(?P<robot>[A-Za-z0-9._-]+)/stop$")
ROBOT_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
LATENCY_WINDOW = 1000
HELLO_TIMEOUT_S = 10.0
# Control frames carry no images; anything larger is an observation, relayed undecoded.
MAX_CONTROL_FRAME_BYTES = 4096
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


@dataclass
class Backend:
    url: str
    active: int = 0


@dataclass
class ModelStats:
    requests: int = 0
    errors: int = 0
    sessions: int = 0
    episodes: int = 0
    successes: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))

    def snapshot(self) -> dict:
        lat = np.asarray(self.latencies_ms) if self.latencies_ms else None
        return {
            "requests": self.requests,
            "errors": self.errors,
            "active_sessions": self.sessions,
            "episodes": self.episodes,
            "success_rate": self.successes / self.episodes if self.episodes else None,
            "relay_p50_ms": float(np.percentile(lat, 50)) if lat is not None else None,
            "relay_p99_ms": float(np.percentile(lat, 99)) if lat is not None else None,
        }


@dataclass
class RobotAccess:
    api_key: str
    models: list[str] | None = None  # None = every model


@dataclass
class LiveRobot:
    robot_id: str
    model: str
    embodiment: str
    backend: str
    session_id: str
    connected_at: float = field(default_factory=time.time)
    requests: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    stop_reason: str | None = None

    def snapshot(self) -> dict:
        lat = np.asarray(self.latencies_ms) if self.latencies_ms else None
        return {
            "robot_id": self.robot_id,
            "model": self.model,
            "embodiment": self.embodiment,
            "backend": self.backend,
            "session_id": self.session_id,
            "connected_s": round(time.time() - self.connected_at, 1),
            "requests": self.requests,
            "server_p50_ms": float(np.percentile(lat, 50)) if lat is not None else None,
            "stop_pending": self.stop_reason is not None,
        }


class Gateway:
    def __init__(self, config: dict) -> None:
        keys = config.get("api_keys") or []
        env_key = os.environ.get("VLA_MAAS_API_KEY")
        self.api_keys = [k for k in [*keys, env_key] if k]
        if not self.api_keys:
            raise ValueError("Configure at least one api_key (or set VLA_MAAS_API_KEY).")
        model_specs = config.get("models") or {}
        self.models = {name: [Backend(url) for url in spec["backends"]] for name, spec in model_specs.items()}
        if not self.models:
            raise ValueError("Configure at least one model with backends.")
        self.adapters = {name: spec["adapter"] for name, spec in model_specs.items() if spec.get("adapter")}
        self.robots = {rid: _robot_access(rid, spec) for rid, spec in (config.get("robots") or {}).items()}
        for rid, access in self.robots.items():
            unknown = set(access.models or []) - set(self.models)
            if unknown:
                raise ValueError(f"robot {rid!r} lists unknown models {sorted(unknown)}")
        self.stats = {name: ModelStats() for name in self.models}
        self.request_timeout_s = float(config.get("request_timeout_s", 5.0))
        self.live: dict[str, LiveRobot] = {}

    @staticmethod
    def _key(headers: Headers) -> str | None:
        scheme, _, key = headers.get("Authorization", "").partition(" ")
        return key if scheme in ("Api-Key", "Bearer") and key else None

    def authorized(self, headers: Headers) -> bool:
        """Operator key: valid on every route."""
        key = self._key(headers)
        return key is not None and any(hmac.compare_digest(key, expected) for expected in self.api_keys)

    def robot_authorized(self, headers: Headers, robot_id: str | None, model: str | None) -> bool:
        """Operator key, or this robot's own key on a model it may use."""
        if self.authorized(headers):
            return True
        key, access = self._key(headers), self.robots.get(robot_id or "")
        if key is None or access is None or not hmac.compare_digest(key, access.api_key):
            return False
        return model is None or access.models is None or model in access.models

    def any_robot_key(self, headers: Headers) -> bool:
        key = self._key(headers)
        return key is not None and any(hmac.compare_digest(key, a.api_key) for a in self.robots.values())

    def pick_backend(self, model: str) -> Backend:
        return min(self.models[model], key=lambda backend: backend.active)

    # -- HTTP -----------------------------------------------------------------

    def process_request(self, _: ServerConnection, request: Request) -> Response | None:
        url = urlsplit(request.path)
        act = ACT_PATH.match(url.path)
        if act is not None:
            return self._check_act(request, act["model"], parse_qs(url.query))
        model_info = MODEL_PATH.match(url.path)
        if model_info is not None and (self.authorized(request.headers) or self.any_robot_key(request.headers)):
            return self._model_info(model_info["model"])
        if request.path == "/healthz":
            return _json_response(HTTPStatus.OK, {"status": "ok"})
        if request.path == "/dashboard":
            # The page holds no data; it calls /metrics with the key the viewer enters.
            body = DASHBOARD_HTML.read_bytes()
            headers = Headers([("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
            return Response(HTTPStatus.OK.value, HTTPStatus.OK.phrase, headers, body)
        if not self.authorized(request.headers):
            return _json_response(HTTPStatus.UNAUTHORIZED, {"error": "invalid or missing API key"})
        if request.path == "/v1/models":
            return _json_response(
                HTTPStatus.OK, {"models": [{"id": m, "backends": len(b)} for m, b in self.models.items()]}
            )
        if request.path == "/metrics":
            return _json_response(HTTPStatus.OK, {m: s.snapshot() for m, s in self.stats.items()})
        if url.path == "/v1/robots":
            return _json_response(HTTPStatus.OK, {"robots": [r.snapshot() for r in self.live.values()]})
        stop = STOP_PATH.match(url.path)
        if stop is not None:
            live = self.live.get(stop["robot"])
            if live is None:
                return _json_response(HTTPStatus.NOT_FOUND, {"error": f"robot {stop['robot']!r} is not connected"})
            live.stop_reason = parse_qs(url.query).get("reason", ["operator stop"])[0]
            logger.warning("stop requested for robot %s: %s", live.robot_id, live.stop_reason)
            return _json_response(HTTPStatus.OK, {"status": "stop pending", "robot_id": live.robot_id})
        match = ROBOT_PATH.match(request.path)
        if match is None:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown path {request.path}"})
        if match["model"] not in self.models:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown model {match['model']!r}"})
        return None  # proceed with the websocket handshake

    def _check_act(self, request: Request, model: str, query: dict) -> Response | None:
        robot_id = (query.get("robot_id") or [None])[0]
        if robot_id is None or not ROBOT_ID.match(robot_id):
            return _json_response(HTTPStatus.BAD_REQUEST, {"error": "robot_id query parameter is required"})
        if not self.robot_authorized(request.headers, robot_id, model):
            return _json_response(HTTPStatus.UNAUTHORIZED, {"error": "key not valid for this robot or model"})
        if model not in self.models:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown model {model!r}"})
        if model not in self.adapters:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"model {model!r} has no adapter for /act"})
        return None

    def _model_info(self, model: str) -> Response:
        if model not in self.models:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown model {model!r}"})
        info: dict = {"id": model, "backends": len(self.models[model]), "act": model in self.adapters}
        if model in self.adapters:
            # Without the backend's metadata the horizon may be unknown; robots get the exact
            # values in the handshake.
            adapter = make_adapter(self.adapters[model], {}, None)
            info.update(requires=adapter.requirements().__dict__, provides=adapter.provides().__dict__)
        return _json_response(HTTPStatus.OK, info)

    async def handle(self, client: ServerConnection) -> None:
        if ACT_PATH.match(urlsplit(client.request.path).path):
            await self.handle_act(client)
        else:
            await self.handle_openpi(client)

    # -- Standard Robot protocol -----------------------------------------------

    async def handle_act(self, client: ServerConnection) -> None:
        url = urlsplit(client.request.path)
        model = ACT_PATH.match(url.path)["model"]
        query = parse_qs(url.query)
        robot_id = query["robot_id"][0]
        embodiment = (query.get("embodiment") or [None])[0]
        if robot_id in self.live:
            await client.close(proto.CLOSE_DUPLICATE_ROBOT, "robot_id already connected")
            return
        backend = self.pick_backend(model)
        stats = self.stats[model]
        backend.active += 1
        stats.sessions += 1
        live: LiveRobot | None = None
        try:
            async with connect(backend.url, compression=None, max_size=None) as upstream:
                metadata = proto.unpackb(await upstream.recv())
                adapter = make_adapter(self.adapters[model], metadata, embodiment)
                server = proto.ServerHello(model, adapter.requirements(), adapter.provides())
                await client.send(proto.packb(server.to_wire()))
                robot, problems = await self._robot_hello(client, server, adapter, robot_id, embodiment)
                if robot is None:
                    return
                if problems:
                    await client.send(proto.packb(proto.error("incompatible", "robot cannot run this model", problems)))
                    await client.close(proto.CLOSE_INCOMPATIBLE, "incompatible robot and model")
                    return
                if robot_id in self.live:  # another session won the race
                    await client.close(proto.CLOSE_DUPLICATE_ROBOT, "robot_id already connected")
                    return
                live = LiveRobot(robot_id, model, robot.embodiment, backend.url, uuid.uuid4().hex)
                self.live[robot_id] = live
                logger.info("robot %s (%s) on %s via %s", robot_id, robot.embodiment, model, backend.url)
                await client.send(proto.packb({"type": "ready", "session_id": live.session_id}))
                async for frame in client:
                    reply = await self._act_frame(frame, upstream, adapter, server, robot, live, stats)
                    await client.send(proto.packb(reply))
                    if reply.get("code") == "backend_failed":
                        return
        except OSError as exc:
            stats.errors += 1
            logger.warning("backend %s unreachable for %s: %s", backend.url, model, exc)
            await client.close(1011, "backend unavailable")
        finally:
            if live is not None and self.live.get(robot_id) is live:
                del self.live[robot_id]
            backend.active -= 1
            stats.sessions -= 1

    async def _robot_hello(
        self,
        client: ServerConnection,
        server: proto.ServerHello,
        adapter: ModelAdapter,
        robot_id: str,
        embodiment: str | None,
    ) -> tuple[proto.RobotHello | None, list[str]]:
        try:
            robot = proto.RobotHello.from_wire(proto.unpackb(await asyncio.wait_for(client.recv(), HELLO_TIMEOUT_S)))
        except (asyncio.TimeoutError, ConnectionClosed):
            return None, []
        except (ValueError, TypeError, KeyError) as exc:
            await client.send(proto.packb(proto.error("bad_hello", f"invalid robot_hello: {exc}")))
            await client.close(proto.CLOSE_INCOMPATIBLE, "invalid robot_hello")
            return None, []
        problems = proto.compatibility_problems(server, robot) + adapter.problems(robot)
        if robot.robot_id != robot_id:
            problems.append(f"robot_hello robot_id {robot.robot_id!r} != connection robot_id {robot_id!r}")
        if embodiment is not None and robot.embodiment != embodiment:
            problems.append(f"robot_hello embodiment {robot.embodiment!r} != connection embodiment {embodiment!r}")
        return robot, problems

    async def _act_frame(
        self,
        frame: bytes | str,
        upstream,
        adapter: ModelAdapter,
        server: proto.ServerHello,
        robot: proto.RobotHello,
        live: LiveRobot,
        stats: ModelStats,
    ) -> dict:
        try:
            msg = proto.unpackb(frame)
        except Exception as exc:
            return proto.error("bad_frame", f"cannot decode frame: {exc}")
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "episode_result":
            stats.episodes += 1
            stats.successes += int(bool(msg.get("success")))
            return {"type": "ok"}
        if kind == "reset":
            try:
                await upstream.send(proto.packb({"endpoint": "reset"}))
                await asyncio.wait_for(upstream.recv(), self.request_timeout_s)
            except (asyncio.TimeoutError, ConnectionClosed) as exc:
                stats.errors += 1
                return proto.error("backend_failed", f"backend failed: {exc!r}")
            return {"type": "ok"}
        if kind != "obs":
            return proto.error("bad_frame", f"unknown message type {kind!r}")
        if live.stop_reason is not None:
            reason, live.stop_reason = live.stop_reason, None
            return {"type": "stop", "reason": reason}
        problems = proto.obs_problems(msg, server.requires, robot.dof)
        if problems:
            return proto.error("bad_obs", "observation does not match the handshake", problems)
        start = time.perf_counter()
        stats.requests += 1
        live.requests += 1
        try:
            native = adapter.to_native(msg, robot, live.session_id)
        except Exception as exc:
            stats.errors += 1
            return proto.error("bad_obs", f"cannot translate observation: {exc}")
        try:
            await upstream.send(proto.packb(native))
            reply = proto.unpackb(await asyncio.wait_for(upstream.recv(), self.request_timeout_s))
        except (asyncio.TimeoutError, ConnectionClosed) as exc:
            stats.errors += 1
            return proto.error("backend_failed", f"backend failed: {exc!r}")
        if isinstance(reply, dict) and reply.get("type") == "error":
            stats.errors += 1
            return proto.error("model_error", str(reply.get("message", reply)))
        try:
            actions = adapter.from_native(reply, msg, robot)
        except Exception as exc:
            stats.errors += 1
            return proto.error("model_error", f"cannot translate model output: {exc}")
        server_ms = (time.perf_counter() - start) * 1e3
        stats.latencies_ms.append(server_ms)
        live.latencies_ms.append(server_ms)
        provides = server.provides
        return {
            "type": "actions",
            "seq": msg.get("seq"),
            "t_obs": msg.get("t"),
            "space": provides.space,
            "mode": "absolute",
            "control_hz": provides.control_hz,
            "server_ms": round(server_ms, 2),
            "actions": actions,
        }

    # -- Native OpenPI relay ---------------------------------------------------

    async def handle_openpi(self, client: ServerConnection) -> None:
        model = ROBOT_PATH.match(client.request.path)["model"]
        backend = self.pick_backend(model)
        stats = self.stats[model]
        packer = msgpack_numpy.Packer()
        backend.active += 1
        stats.sessions += 1
        try:
            async with connect(backend.url, compression=None, max_size=None) as upstream:
                metadata = msgpack_numpy.unpackb(await upstream.recv())
                metadata.update(model=model, backend=backend.url)
                await client.send(packer.pack(metadata))
                async for frame in client:
                    if isinstance(frame, bytes) and len(frame) <= MAX_CONTROL_FRAME_BYTES:
                        control = msgpack_numpy.unpackb(frame)
                        if isinstance(control, dict) and control.get("endpoint") == "episode_result":
                            stats.episodes += 1
                            stats.successes += int(bool(control.get("success")))
                            await client.send(packer.pack({"status": "ok"}))
                            continue
                    start = time.perf_counter()
                    stats.requests += 1
                    try:
                        await upstream.send(frame)
                        reply = await asyncio.wait_for(upstream.recv(), self.request_timeout_s)
                    except (asyncio.TimeoutError, ConnectionClosed) as exc:
                        stats.errors += 1
                        await client.send(packer.pack({"type": "error", "message": f"backend failed: {exc!r}"}))
                        return
                    stats.latencies_ms.append((time.perf_counter() - start) * 1e3)
                    await client.send(reply)
        except OSError as exc:
            stats.errors += 1
            logger.warning("backend %s unreachable for %s: %s", backend.url, model, exc)
            await client.close(1011, "backend unavailable")
        finally:
            backend.active -= 1
            stats.sessions -= 1


def _robot_access(robot_id: str, spec: dict) -> RobotAccess:
    key = spec.get("api_key") or (os.environ.get(spec["api_key_env"]) if spec.get("api_key_env") else None)
    if not key:
        raise ValueError(f"robot {robot_id!r} needs api_key or a set api_key_env")
    models = spec.get("models")
    return RobotAccess(api_key=key, models=list(models) if models is not None else None)


def _json_response(status: HTTPStatus, body: dict) -> Response:
    payload = json.dumps(body).encode()
    headers = Headers([("Content-Type", "application/json"), ("Content-Length", str(len(payload)))])
    return Response(status.value, status.phrase, headers, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="VLA MaaS gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config) as f:
        gateway = Gateway(yaml.safe_load(f))

    async def run() -> None:
        async with serve(
            gateway.handle,
            args.host,
            args.port,
            process_request=gateway.process_request,
            compression=None,
            max_size=None,
        ) as server:
            logger.info("gateway on %s:%d serving %s", args.host, args.port, sorted(gateway.models))
            await server.serve_forever()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
