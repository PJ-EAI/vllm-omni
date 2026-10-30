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
    GET /v1/admin/reload           re-read --config and swap changed models' checkpoint, model type,
                                   backends or adapter live (operator key; see routes.py). Also SIGHUP.
                                   ``?cutover=stop`` stops /act sessions still on a replaced revision.
    GET /v1/admin/revisions        each model's current revision and the recent swaps (operator key)
    WS  /v1/monitor                live scheduling, performance and per-session camera frames and state
                                   for the dashboard (operator key in the first message; see monitor.py)

Logs: one structured event per line (``maas_log.py``), every line of a session
tagged with its session id, robot, model, revision, checkpoint and backend.

Usage:
    python gateway.py --config gateway.example.yaml [--log-file logs/gateway.jsonl]
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import re
import signal
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import maas_log
import vla_maas_protocol as proto
import yaml
from maas_log import event
from model_adapters import ModelAdapter, make_adapter
from monitor import Backend, ModelStats, Monitor, SessionView, _percentiles
from openpi_client import msgpack_numpy
from routes import PROBE_TIMEOUT_S, Revision, RouteSpec, normalize_model_type, probe_route
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
MONITOR_PATH = "/v1/monitor"
RELOAD_PATH = "/v1/admin/reload"
REVISIONS_PATH = "/v1/admin/revisions"
HISTORY_KEPT = 50
ROBOT_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
HELLO_TIMEOUT_S = 10.0
# Control frames carry no images; anything larger is an observation, relayed undecoded.
MAX_CONTROL_FRAME_BYTES = 4096
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


@dataclass
class RobotAccess:
    api_key: str
    models: list[str] | None = None  # None = every model


class Gateway:
    def __init__(self, config: dict, config_path: str | Path | None = None, request_log_level: int = logging.INFO):
        """``config_path``: the YAML ``GET /v1/admin/reload`` and SIGHUP re-read; None disables reloads."""
        self.config_path = Path(config_path) if config_path is not None else None
        self.request_log_level = request_log_level
        specs = _route_specs(config)
        self._use(_settings(config, specs))
        self._revision_n = {name: 1 for name in specs}
        self.routes: dict[str, Revision] = {name: Revision.build(name, 1, spec) for name, spec in specs.items()}
        # Every backend still holding sessions, current and draining; the monitor shows these.
        self.models: dict[str, list[Backend]] = {name: list(rev.backends) for name, rev in self.routes.items()}
        self.stats = {name: ModelStats() for name in self.models}
        self.history: deque[dict] = deque(maxlen=HISTORY_KEPT)
        self._reload_lock = asyncio.Lock()
        self.live: dict[str, SessionView] = {}  # /act sessions by robot_id
        self.monitor = Monitor(self.models, self.stats, self.operator_key)

    def _use(self, settings: dict) -> None:
        self.api_keys: list[str] = settings["api_keys"]
        self.robots: dict[str, RobotAccess] = settings["robots"]
        self.request_timeout_s: float = settings["request_timeout_s"]
        self.probe_timeout_s: float = settings["probe_timeout_s"]

    @staticmethod
    def _key(headers: Headers) -> str | None:
        scheme, _, key = headers.get("Authorization", "").partition(" ")
        return key if scheme in ("Api-Key", "Bearer") and key else None

    def operator_key(self, key: str | None) -> bool:
        return bool(key) and any(hmac.compare_digest(key, expected) for expected in self.api_keys)

    def authorized(self, headers: Headers) -> bool:
        """Operator key: valid on every route."""
        return self.operator_key(self._key(headers))

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

    def pick_backend(self, model: str) -> tuple[Revision | None, Backend | None]:
        """The model's current revision and its least-loaded backend (None while paused or just removed)."""
        rev = self.routes.get(model)
        if rev is None:
            return None, None
        return rev, min(rev.backends, key=lambda backend: backend.active, default=None)

    def _session_closed(self, model: str, backend: Backend) -> None:
        backend.active -= 1
        if backend.draining and backend.active <= 0:
            self._retire(model, backend)

    def _retire(self, model: str, backend: Backend) -> None:
        """Forget a drained backend of a replaced revision."""
        backends = self.models.get(model, [])
        if backend in backends:
            backends.remove(backend)
            event(
                logger,
                "backend.drained",
                f"{backend.url} drained",
                model=model,
                revision=backend.revision,
                backend=backend.url,
            )
        if not backends and model not in self.routes:
            self.models.pop(model, None)
            self.stats.pop(model, None)

    def _note_backend(self, rev: Revision, backend: Backend, metadata) -> None:
        """Record what a backend reports in its handshake; warn when it is not what the route claims."""
        server = metadata.get("server") if isinstance(metadata, dict) else None
        if not isinstance(server, dict):
            return
        reported_ckpt, reported_type = server.get("checkpoint"), server.get("model_type")
        if backend.checkpoint is None:
            backend.checkpoint = reported_ckpt
        if backend.model_type is None:
            backend.model_type = reported_type
        maas_log.bind(checkpoint=backend.checkpoint, model_type=backend.model_type)
        mismatch = (rev.spec.checkpoint and reported_ckpt and reported_ckpt != rev.spec.checkpoint) or (
            rev.spec.model_type
            and reported_type
            and normalize_model_type(reported_type) != normalize_model_type(rev.spec.model_type)
        )
        if mismatch:
            event(
                logger,
                "backend.mismatch",
                "backend serves something other than the route's labels",
                logging.WARNING,
                reported_checkpoint=reported_ckpt,
                reported_model_type=reported_type,
                expected_checkpoint=rev.spec.checkpoint,
                expected_model_type=rev.spec.model_type,
            )

    # -- Live swaps -------------------------------------------------------------

    async def reload(self, source: str = "http", cutover: str | None = None) -> tuple[HTTPStatus, dict]:
        """Re-read the config; probe changed routes; swap them in, or change nothing on any problem."""
        if self.config_path is None:
            return HTTPStatus.CONFLICT, {"error": "gateway was started without a config file; nothing to reload"}
        if cutover not in (None, "drain", "stop"):
            return HTTPStatus.BAD_REQUEST, {"error": "cutover must be drain or stop"}
        async with self._reload_lock:
            try:
                config = yaml.safe_load(self.config_path.read_text()) or {}
                specs = _route_specs(config)
                settings = _settings(config, specs)  # validated here, applied only if the probes pass
            except Exception as exc:
                event(
                    logger,
                    "route.reload_rejected",
                    f"cannot load {self.config_path}: {exc}",
                    logging.ERROR,
                    source=source,
                )
                return HTTPStatus.BAD_REQUEST, {"error": f"invalid config: {exc}"}
            changed = {n: s for n, s in specs.items() if n not in self.routes or self.routes[n].spec != s}
            removed = [n for n in self.routes if n not in specs]
            probes = await asyncio.gather(*(probe_route(s, self.probe_timeout_s) for s in changed.values()))
            problems = {name: ps for name, (_, ps) in zip(changed, probes) if ps}
            if problems:
                event(
                    logger,
                    "route.reload_rejected",
                    "probe failed; nothing changed",
                    logging.ERROR,
                    source=source,
                    problems=problems,
                )
                return HTTPStatus.CONFLICT, {"error": "reload rejected; nothing changed", "problems": problems}
            self._use(settings)
            changes = [
                self._swap(name, spec, reported, source) for (name, spec), (reported, _) in zip(changed.items(), probes)
            ]
            for name in removed:
                changes.append(self._remove(name, source))
            stopped = self._cutover() if cutover == "stop" else 0
            event(
                logger,
                "route.reload",
                f"{len(changes)} model(s) changed",
                source=source,
                changed=[c["model"] for c in changes],
                stopped_sessions=stopped,
            )
            return HTTPStatus.OK, {
                "status": "reloaded" if changes else "unchanged",
                "changes": changes,
                "stopped_sessions": stopped,
            }

    def _swap(self, name: str, spec: RouteSpec, reported: list[dict], source: str) -> dict:
        old = self.routes.get(name)
        n = self._revision_n[name] = self._revision_n.get(name, 0) + 1
        rev = self.routes[name] = Revision.build(name, n, spec, reported)
        self.stats.setdefault(name, ModelStats())
        backends = self.models.setdefault(name, [])
        draining = self._drain(name, old)
        backends[:0] = rev.backends
        record = {
            "t": time.time(),
            "model": name,
            "source": source,
            "from": old.summary() if old else None,
            "to": rev.summary(),
            "draining_sessions": draining,
        }
        self.history.appendleft(record)
        event(
            logger,
            "route.revision",
            f"{name}: {old.id if old else '-'} -> {rev.id}",
            model=name,
            revision=rev.id,
            previous=old.id if old else None,
            checkpoint=rev.checkpoint,
            previous_checkpoint=old.checkpoint if old else None,
            model_type=rev.model_type,
            previous_model_type=old.model_type if old else None,
            backends=list(spec.backends),
            draining_sessions=draining,
            source=source,
        )
        return record

    def _remove(self, name: str, source: str) -> dict:
        old = self.routes.pop(name)
        draining = self._drain(name, old)
        record = {
            "t": time.time(),
            "model": name,
            "source": source,
            "from": old.summary(),
            "to": None,
            "draining_sessions": draining,
        }
        self.history.appendleft(record)
        event(
            logger,
            "route.removed",
            f"{name} removed",
            model=name,
            previous=old.id,
            draining_sessions=draining,
            source=source,
        )
        return record

    def _drain(self, name: str, old: Revision | None) -> int:
        """Mark a replaced revision's backends draining; drop the idle ones. Returns the sessions left on it."""
        if old is None:
            return 0
        sessions = 0
        for backend in old.backends:
            backend.draining = True
            sessions += backend.active
            if backend.active <= 0:
                self._retire(name, backend)
            else:
                event(
                    logger,
                    "backend.draining",
                    f"{backend.url} draining {backend.active} session(s)",
                    model=name,
                    revision=backend.revision,
                    backend=backend.url,
                    sessions=backend.active,
                )
        return sessions

    def _cutover(self) -> int:
        """Stop every /act session on a draining backend at its next observation."""
        views = [v for v in self.monitor.sessions.values() if v.kind == "act" and v.backend.draining]
        for view in views:
            current = self.routes.get(view.model)
            self.monitor.request_stop(view, f"model {view.model} swapped to {current.id if current else 'none'}")
        return len(views)

    def revisions(self) -> dict:
        return {
            "models": {name: rev.summary() for name, rev in self.routes.items()},
            "draining": {
                name: [b.snapshot() for b in backends if b.draining] for name, backends in self.models.items()
            },
            "history": list(self.history),
        }

    # -- HTTP -----------------------------------------------------------------

    async def process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        url = urlsplit(request.path)
        if url.path == RELOAD_PATH:
            if not self.authorized(request.headers):
                return _json_response(HTTPStatus.UNAUTHORIZED, {"error": "invalid or missing API key"})
            cutover = (parse_qs(url.query).get("cutover") or [None])[0]
            return _json_response(*await self.reload("http", cutover))
        return self._process_request(connection, request)

    def _process_request(self, _: ServerConnection, request: Request) -> Response | None:
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
        if request.path == MONITOR_PATH:
            return None  # browsers cannot set headers on a websocket; the key is the first message
        if not self.authorized(request.headers):
            return _json_response(HTTPStatus.UNAUTHORIZED, {"error": "invalid or missing API key"})
        if request.path == "/v1/models":
            return _json_response(HTTPStatus.OK, {"models": [self._model_row(m) for m in self.routes]})
        if request.path == REVISIONS_PATH:
            return _json_response(HTTPStatus.OK, self.revisions())
        if request.path == "/metrics":
            return _json_response(HTTPStatus.OK, {m: s.snapshot() for m, s in self.stats.items()})
        if url.path == "/v1/robots":
            return _json_response(HTTPStatus.OK, {"robots": [r.robot_snapshot() for r in self.live.values()]})
        stop = STOP_PATH.match(url.path)
        if stop is not None:
            live = self.live.get(stop["robot"])
            if live is None:
                return _json_response(HTTPStatus.NOT_FOUND, {"error": f"robot {stop['robot']!r} is not connected"})
            self.monitor.request_stop(live, parse_qs(url.query).get("reason", ["operator stop"])[0])
            return _json_response(HTTPStatus.OK, {"status": "stop pending", "robot_id": live.robot_id})
        match = ROBOT_PATH.match(request.path)
        if match is None:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown path {request.path}"})
        if match["model"] not in self.routes:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown model {match['model']!r}"})
        return self._paused(match["model"])  # None: proceed with the websocket handshake

    def _paused(self, model: str) -> Response | None:
        if self.routes[model].backends:
            return None
        return _json_response(HTTPStatus.SERVICE_UNAVAILABLE, {"error": f"model {model!r} is being swapped; retry"})

    def _check_act(self, request: Request, model: str, query: dict) -> Response | None:
        robot_id = (query.get("robot_id") or [None])[0]
        if robot_id is None or not ROBOT_ID.match(robot_id):
            return _json_response(HTTPStatus.BAD_REQUEST, {"error": "robot_id query parameter is required"})
        if not self.robot_authorized(request.headers, robot_id, model):
            return _json_response(HTTPStatus.UNAUTHORIZED, {"error": "key not valid for this robot or model"})
        if model not in self.routes:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown model {model!r}"})
        if self.routes[model].spec.adapter is None:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"model {model!r} has no adapter for /act"})
        return self._paused(model)

    def _model_row(self, model: str) -> dict:
        rev = self.routes[model]
        return {
            "id": model,
            "backends": len(rev.backends),
            "revision": rev.id,
            "checkpoint": rev.checkpoint,
            "model_type": rev.model_type,
            "draining": sum(b.draining for b in self.models.get(model, [])),
        }

    def _model_info(self, model: str) -> Response:
        if model not in self.routes:
            return _json_response(HTTPStatus.NOT_FOUND, {"error": f"unknown model {model!r}"})
        adapter_spec = self.routes[model].spec.adapter
        info: dict = {**self._model_row(model), "act": adapter_spec is not None}
        if adapter_spec is not None:
            # Without the backend's metadata the horizon may be unknown; robots get the exact
            # values in the handshake.
            adapter = make_adapter(adapter_spec, {}, None)
            info.update(requires=adapter.requirements().__dict__, provides=adapter.provides().__dict__)
        return _json_response(HTTPStatus.OK, info)

    async def handle(self, client: ServerConnection) -> None:
        path = urlsplit(client.request.path).path
        if path == MONITOR_PATH:
            await self.monitor.serve(client)
        elif ACT_PATH.match(path):
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
        rev, backend = self.pick_backend(model)
        if rev is None or backend is None:  # paused or removed between the HTTP check and here
            await client.close(1013, "model is being swapped; retry")
            return
        stats = self.stats[model]
        view = self.monitor.open(
            SessionView("act", model, stats, backend, _peer(client), robot_id=robot_id, embodiment=embodiment)
        )
        backend.active += 1
        stats.sessions += 1
        reason = "robot disconnected"
        with maas_log.context(
            session=view.session_id,
            robot=robot_id,
            model=model,
            revision=rev.id,
            checkpoint=backend.checkpoint,
            model_type=backend.model_type,
            backend=backend.url,
            kind="act",
        ):
            event(logger, "session.open", f"robot {robot_id} connecting", peer=view.peer, embodiment=embodiment)
            try:
                async with connect(backend.url, compression=None, max_size=None) as upstream:
                    metadata = proto.unpackb(await upstream.recv())
                    self._note_backend(rev, backend, metadata)
                    # The revision picked above, not the current one: a swap during the handshake must not
                    # pair the old backend with the new adapter.
                    adapter = make_adapter(rev.spec.adapter, metadata, embodiment)
                    server = proto.ServerHello(model, adapter.requirements(), adapter.provides())
                    await client.send(proto.packb(server.to_wire()))
                    robot, problems = await self._robot_hello(client, server, adapter, robot_id, embodiment)
                    if robot is None:
                        reason = "no valid robot_hello"
                        return
                    if problems:
                        reason = "incompatible: " + "; ".join(problems)
                        await client.send(
                            proto.packb(proto.error("incompatible", "robot cannot run this model", problems))
                        )
                        await client.close(proto.CLOSE_INCOMPATIBLE, "incompatible robot and model")
                        return
                    if robot_id in self.live:  # another session won the race
                        reason = "robot_id already connected"
                        await client.close(proto.CLOSE_DUPLICATE_ROBOT, "robot_id already connected")
                        return
                    view.embodiment, view.ready = robot.embodiment, True
                    self.live[robot_id] = view
                    view.event("ready", f"{robot.embodiment} via {backend.url} ({rev.id})")
                    event(
                        logger,
                        "session.ready",
                        f"robot {robot_id} ({robot.embodiment}) ready",
                        embodiment=robot.embodiment,
                        dof=robot.dof,
                        horizon=server.provides.horizon,
                        control_hz=server.provides.control_hz,
                    )
                    await client.send(
                        proto.packb(
                            {
                                "type": "ready",
                                "session_id": view.session_id,
                                "revision": rev.id,
                                "checkpoint": backend.checkpoint,
                                "model_type": backend.model_type,
                            }
                        )
                    )
                    async for frame in client:
                        reply = await self._act_frame(frame, upstream, adapter, server, robot, view)
                        await client.send(proto.packb(reply))
                        if reply.get("code") == "backend_failed":
                            reason = "backend failed"
                            return
            except OSError as exc:
                reason = "backend unavailable"
                view.fail("backend_unavailable", str(exc))
                event(logger, "backend.unavailable", f"backend unreachable: {exc}", logging.WARNING)
                await client.close(1011, "backend unavailable")
            finally:
                if self.live.get(robot_id) is view:
                    del self.live[robot_id]
                self._close_view(view, reason)
                self._session_closed(model, backend)
                stats.sessions -= 1

    def _close_view(self, view: SessionView, reason: str) -> None:
        self.monitor.close(view, reason)
        p50, p99 = _percentiles(view.latencies_ms)
        event(
            logger,
            "session.close",
            f"session closed: {reason}",
            reason=reason,
            duration_s=round(time.time() - view.connected_at, 1),
            requests=view.requests,
            errors=view.errors,
            episodes=view.episodes,
            successes=view.successes,
            p50_ms=round(p50, 2) if p50 is not None else None,
            p99_ms=round(p99, 2) if p99 is not None else None,
            last_error=view.last_error,
        )

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
        view: SessionView,
    ) -> dict:
        seq = None

        def failed(code: str, message: str, problems: list[str] | None = None) -> dict:
            view.fail(code, message)
            event(logger, "act.error", message, logging.WARNING, code=code, seq=seq, episode_id=view.episode_id)
            return proto.error(code, message, problems)

        try:
            msg = proto.unpackb(frame)
        except Exception as exc:
            event(logger, "act.bad_frame", f"cannot decode frame: {exc}", logging.WARNING)
            return proto.error("bad_frame", f"cannot decode frame: {exc}")
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "episode_result":
            view.episode(bool(msg.get("success")))
            event(
                logger,
                "act.episode_result",
                "episode result",
                success=bool(msg.get("success")),
                episode_id=view.episode_id,
            )
            return {"type": "ok"}
        if kind == "reset":
            view.event("reset")
            event(logger, "act.reset", "reset", episode_id=view.episode_id)
            try:
                await upstream.send(proto.packb({"endpoint": "reset"}))
                await asyncio.wait_for(upstream.recv(), self.request_timeout_s)
            except (asyncio.TimeoutError, ConnectionClosed) as exc:
                return failed("backend_failed", f"backend failed: {exc!r}")
            return {"type": "ok"}
        if kind != "obs":
            return proto.error("bad_frame", f"unknown message type {kind!r}")
        seq = msg.get("seq")
        if view.stop_reason is not None:
            reason, view.stop_reason = view.stop_reason, None
            view.event("stopped", reason)
            event(logger, "act.stop", f"stop sent: {reason}", logging.WARNING, seq=seq)
            return {"type": "stop", "reason": reason}
        problems = proto.obs_problems(msg, server.requires, robot.dof)
        if problems:
            view.event("bad_obs", "; ".join(problems))
            event(
                logger,
                "act.bad_obs",
                "observation does not match the handshake",
                logging.WARNING,
                seq=seq,
                problems=problems,
            )
            return proto.error("bad_obs", "observation does not match the handshake", problems)
        view.observe(msg)
        start = time.perf_counter()
        view.stats.requests += 1
        try:
            native = adapter.to_native(msg, robot, view.session_id)
        except Exception as exc:
            return failed("bad_obs", f"cannot translate observation: {exc}")
        try:
            await upstream.send(proto.packb(native))
            reply = proto.unpackb(await asyncio.wait_for(upstream.recv(), self.request_timeout_s))
        except (asyncio.TimeoutError, ConnectionClosed) as exc:
            return failed("backend_failed", f"backend failed: {exc!r}")
        if isinstance(reply, dict) and reply.get("type") == "error":
            return failed("model_error", str(reply.get("message", reply)))
        try:
            actions = adapter.from_native(reply, msg, robot)
        except Exception as exc:
            return failed("model_error", f"cannot translate model output: {exc}")
        server_ms = (time.perf_counter() - start) * 1e3
        view.record(server_ms, actions)
        provides = server.provides
        event(
            logger,
            "act.request",
            "actions",
            self.request_log_level,
            seq=seq,
            episode_id=view.episode_id,
            server_ms=round(server_ms, 2),
        )
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
        rev, backend = self.pick_backend(model)
        if rev is None or backend is None:
            await client.close(1013, "model is being swapped; retry")
            return
        stats = self.stats[model]
        view = self.monitor.open(SessionView("openpi", model, stats, backend, _peer(client)))
        packer = msgpack_numpy.Packer()
        backend.active += 1
        stats.sessions += 1
        reason = "client disconnected"
        with maas_log.context(
            session=view.session_id,
            model=model,
            revision=rev.id,
            checkpoint=backend.checkpoint,
            model_type=backend.model_type,
            backend=backend.url,
            kind="openpi",
        ):
            event(logger, "session.open", "native OpenPI client connecting", peer=view.peer)
            try:
                async with connect(backend.url, compression=None, max_size=None) as upstream:
                    metadata = msgpack_numpy.unpackb(await upstream.recv())
                    self._note_backend(rev, backend, metadata)
                    metadata.update(model=model, backend=backend.url, revision=rev.id)
                    await client.send(packer.pack(metadata))
                    view.ready = True
                    event(logger, "session.ready", "native OpenPI client ready")
                    async for frame in client:
                        if isinstance(frame, bytes) and len(frame) <= MAX_CONTROL_FRAME_BYTES:
                            control = msgpack_numpy.unpackb(frame)
                            if isinstance(control, dict) and control.get("endpoint") == "episode_result":
                                view.episode(bool(control.get("success")))
                                event(
                                    logger,
                                    "openpi.episode_result",
                                    "episode result",
                                    success=bool(control.get("success")),
                                )
                                await client.send(packer.pack({"status": "ok"}))
                                continue
                        else:
                            view.observe(frame)  # kept undecoded; the dashboard decodes it if it watches
                        start = time.perf_counter()
                        stats.requests += 1
                        try:
                            await upstream.send(frame)
                            reply = await asyncio.wait_for(upstream.recv(), self.request_timeout_s)
                        except (asyncio.TimeoutError, ConnectionClosed) as exc:
                            reason = "backend failed"
                            view.fail("backend_failed", f"backend failed: {exc!r}")
                            event(
                                logger,
                                "openpi.error",
                                f"backend failed: {exc!r}",
                                logging.WARNING,
                                code="backend_failed",
                                n=view.obs_seq,
                            )
                            await client.send(packer.pack({"type": "error", "message": f"backend failed: {exc!r}"}))
                            return
                        server_ms = (time.perf_counter() - start) * 1e3
                        view.record(server_ms, reply)
                        event(
                            logger,
                            "openpi.request",
                            "reply",
                            self.request_log_level,
                            n=view.obs_seq,
                            server_ms=round(server_ms, 2),
                        )
                        await client.send(reply)
            except OSError as exc:
                reason = "backend unavailable"
                view.fail("backend_unavailable", str(exc))
                event(logger, "backend.unavailable", f"backend unreachable: {exc}", logging.WARNING)
                await client.close(1011, "backend unavailable")
            finally:
                self._close_view(view, reason)
                self._session_closed(model, backend)
                stats.sessions -= 1


def _peer(client: ServerConnection) -> str:
    """The client's address, or the proxy's X-Forwarded-For when behind one."""
    forwarded = client.request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    address = client.remote_address
    return f"{address[0]}:{address[1]}" if address else "unknown"


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


def _settings(config: dict, specs: dict[str, RouteSpec]) -> dict:
    """Keys, robots and timeouts from a config, validated as a whole (startup and reload)."""
    keys = config.get("api_keys") or []
    env_key = os.environ.get("VLA_MAAS_API_KEY")
    api_keys = [k for k in [*keys, env_key] if k]
    if not api_keys:
        raise ValueError("Configure at least one api_key (or set VLA_MAAS_API_KEY).")
    robots = {rid: _robot_access(rid, spec) for rid, spec in (config.get("robots") or {}).items()}
    for rid, access in robots.items():
        unknown = set(access.models or []) - set(specs)
        if unknown:
            raise ValueError(f"robot {rid!r} lists unknown models {sorted(unknown)}")
    return {
        "api_keys": api_keys,
        "robots": robots,
        "request_timeout_s": float(config.get("request_timeout_s", 5.0)),
        "probe_timeout_s": float(config.get("probe_timeout_s", PROBE_TIMEOUT_S)),
    }


def _route_specs(config: dict) -> dict[str, RouteSpec]:
    specs = {name: RouteSpec.parse(name, spec) for name, spec in (config.get("models") or {}).items()}
    if not specs:
        raise ValueError("Configure at least one model with backends.")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="VLA MaaS gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--log-format", choices=["json", "text"], help="stderr format; overrides logging.format")
    parser.add_argument("--log-file", help="JSON-lines log file; overrides logging.file in the config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    log_cfg = dict(config.get("logging") or {})
    log_cfg.update({k: v for k, v in (("format", args.log_format), ("file", args.log_file)) if v})
    request_level = maas_log.configure(log_cfg)
    gateway = Gateway(config, config_path=args.config, request_log_level=request_level)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        reloads: set[asyncio.Task] = set()

        def on_sighup() -> None:
            task = loop.create_task(gateway.reload("sighup"))
            reloads.add(task)
            task.add_done_callback(reloads.discard)

        if hasattr(signal, "SIGHUP"):
            loop.add_signal_handler(signal.SIGHUP, on_sighup)
        async with serve(
            gateway.handle,
            args.host,
            args.port,
            process_request=gateway.process_request,
            compression=None,
            max_size=None,
        ) as server:
            event(
                logger,
                "gateway.start",
                f"gateway on {args.host}:{args.port}",
                models={name: rev.summary() for name, rev in gateway.routes.items()},
            )
            await server.serve_forever()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
