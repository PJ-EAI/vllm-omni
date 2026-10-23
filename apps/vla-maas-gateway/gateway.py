#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-tenant MaaS gateway in front of vllm-omni VLA policy servers.

Clients connect to ``ws://<gateway>/v1/robot/<model>/openpi`` with
``Authorization: Api-Key <key>`` (the header ``openpi-client`` sends when given
``api_key``). The gateway authenticates, picks the least-loaded backend for that
model, and relays OpenPI frames one-to-one. The handshake metadata is the
backend's, plus ``model`` and ``backend`` so clients can log what served them.

Clients may send ``{"endpoint": "episode_result", "success": bool}`` on the
same socket after an episode; the gateway records it for the dashboard, answers
``{"status": "ok"}`` and does not forward it.

HTTP endpoints on the same port:
    GET /healthz      liveness
    GET /dashboard    live demo panel (polls /metrics with the key typed into it)
    GET /v1/models    configured models and backend counts (auth required)
    GET /metrics      per-model request counts, errors and relay latency (auth required)

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
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path

import numpy as np
import yaml
from openpi_client import msgpack_numpy
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

logger = logging.getLogger("vla_maas_gateway")

ROBOT_PATH = re.compile(r"^/v1/robot/(?P<model>[A-Za-z0-9._-]+)/openpi$")
LATENCY_WINDOW = 1000
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


class Gateway:
    def __init__(self, config: dict) -> None:
        keys = config.get("api_keys") or []
        env_key = os.environ.get("VLA_MAAS_API_KEY")
        self.api_keys = [k for k in [*keys, env_key] if k]
        if not self.api_keys:
            raise ValueError("Configure at least one api_key (or set VLA_MAAS_API_KEY).")
        self.models = {
            name: [Backend(url) for url in spec["backends"]] for name, spec in (config.get("models") or {}).items()
        }
        if not self.models:
            raise ValueError("Configure at least one model with backends.")
        self.stats = {name: ModelStats() for name in self.models}
        self.request_timeout_s = float(config.get("request_timeout_s", 5.0))

    def authorized(self, headers: Headers) -> bool:
        value = headers.get("Authorization", "")
        scheme, _, key = value.partition(" ")
        if scheme not in ("Api-Key", "Bearer"):
            return False
        return any(hmac.compare_digest(key, expected) for expected in self.api_keys)

    def pick_backend(self, model: str) -> Backend:
        return min(self.models[model], key=lambda backend: backend.active)

    # -- HTTP -----------------------------------------------------------------

    def process_request(self, _: ServerConnection, request: Request) -> Response | None:
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
        match = ROBOT_PATH.match(request.path)
        if match is None:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown path {request.path}"})
        if match["model"] not in self.models:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown model {match['model']!r}"})
        return None  # proceed with the websocket handshake

    # -- WebSocket relay -------------------------------------------------------

    async def handle(self, client: ServerConnection) -> None:
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
                    except (TimeoutError, ConnectionClosed) as exc:
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
