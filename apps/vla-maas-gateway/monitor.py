# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live view of the gateway's scheduling, performance and sessions for the dashboard.

Every session (``/act`` and the native OpenPI relay) gets a ``SessionView``: its
counters, recent events and a reference to the latest observation and action
chunk. The relay path only stores references; images are JPEG-encoded off the
event loop, and only for the session a dashboard is watching, so an unwatched
gateway pays nothing for the view.

``ws://<gateway>/v1/monitor`` speaks JSON text frames:

* client: ``{"type": "auth", "key": "<operator key>"}`` first (browsers cannot set
  headers on a websocket), then any of
  ``{"type": "watch", "session_id": ..., "fps": 5}``, ``{"type": "unwatch"}``,
  ``{"type": "stop", "session_id": ..., "reason": ...}`` (``/act`` sessions only).
* server: ``hello`` (per-model one-second history), a ``snapshot`` every second
  (models, backends, live and recently closed sessions, new history points) and,
  while watching, ``detail`` frames (camera JPEGs, state, actions, events) at
  most ``fps`` times a second.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("vla_maas_gateway")

LATENCY_WINDOW = 1000
SERIES_S = 600  # one-second points kept per model
EVENTS_KEPT = 50
CLOSED_KEPT = 20
IDLE_AFTER_S = 5.0  # no observation for this long shows as idle (chunked robots ask every 1-3 s)
AUTH_TIMEOUT_S = 10.0
CLOSE_UNAUTHORIZED = 4401
DEFAULT_FPS, MAX_FPS = 5.0, 15.0
JPEG_QUALITY = 75
MAX_ACTION_ROWS, MAX_DIMS = 100, 32


def _percentiles(values) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lat = np.asarray(values)
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 99))


def _fps(value: Any) -> float:
    """A watch request's fps, clamped into range; malformed values fall back, not raise."""
    try:
        return min(MAX_FPS, max(0.5, float(value)))
    except (TypeError, ValueError):
        return DEFAULT_FPS


@dataclass(eq=False)
class Backend:
    url: str
    active: int = 0
    requests: int = 0
    errors: int = 0
    revision: str | None = None  # the route revision this backend serves (routes.py)
    checkpoint: str | None = None
    model_type: str | None = None
    draining: bool = False  # replaced by a newer revision; keeps its open sessions, gets no new ones

    def snapshot(self) -> dict:
        return {
            "url": self.url,
            "active": self.active,
            "requests": self.requests,
            "errors": self.errors,
            "revision": self.revision,
            "checkpoint": self.checkpoint,
            "model_type": self.model_type,
            "state": "draining" if self.draining else "active",
        }


@dataclass
class ModelStats:
    requests: int = 0
    errors: int = 0
    sessions: int = 0
    episodes: int = 0
    successes: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    series: deque[dict] = field(default_factory=lambda: deque(maxlen=SERIES_S))
    _bucket: dict | None = field(default=None, repr=False)

    def snapshot(self) -> dict:
        p50, p99 = _percentiles(self.latencies_ms)
        return {
            "requests": self.requests,
            "errors": self.errors,
            "active_sessions": self.sessions,
            "episodes": self.episodes,
            "success_rate": self.successes / self.episodes if self.episodes else None,
            "relay_p50_ms": p50,
            "relay_p99_ms": p99,
        }

    def record(self, ms: float) -> None:
        self.latencies_ms.append(ms)
        bucket = self._current()
        bucket["requests"] += 1
        bucket["lat"].append(ms)

    def fail(self) -> None:
        self.errors += 1
        self._current()["errors"] += 1

    def history(self, since: int = 0) -> list[dict]:
        """Closed one-second points after ``since``; seconds without traffic read as zero."""
        now = int(time.time())
        if self._bucket is not None and self._bucket["t"] < now:
            self._close()
        points = {p["t"]: p for p in self.series}
        empty = {"requests": 0, "errors": 0, "p50_ms": None, "p99_ms": None}
        return [points.get(t, {"t": t, **empty}) for t in range(max(since + 1, now - SERIES_S), now)]

    def _current(self) -> dict:
        now = int(time.time())
        if self._bucket is not None and self._bucket["t"] != now:
            self._close()
        if self._bucket is None:
            self._bucket = {"t": now, "requests": 0, "errors": 0, "lat": []}
        return self._bucket

    def _close(self) -> None:
        bucket, self._bucket = self._bucket, None
        p50, p99 = _percentiles(bucket.pop("lat"))
        self.series.append({**bucket, "p50_ms": p50, "p99_ms": p99})


@dataclass(eq=False)
class SessionView:
    """One robot session as the dashboard sees it. Counting here also counts on its model and backend."""

    kind: str  # "act" (standard Robot protocol) or "openpi" (native relay)
    model: str
    stats: ModelStats
    backend: Backend
    peer: str
    robot_id: str | None = None
    embodiment: str | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    connected_at: float = field(default_factory=time.time)
    ready: bool = False
    closed_at: float | None = None
    close_reason: str | None = None
    stop_reason: str | None = None  # set by an operator, delivered on the next observation
    requests: int = 0
    errors: int = 0
    episodes: int = 0
    successes: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    obs_seq: int = 0
    last_obs_at: float | None = None
    last_obs: Any = None  # /act: the decoded obs dict; openpi: the raw msgpack frame
    last_actions: Any = None  # /act: the actions dict; openpi: the raw reply frame
    instruction: str | None = None
    episode_id: str | None = None
    last_server_ms: float | None = None
    last_error: dict | None = None
    events: deque[dict] = field(default_factory=lambda: deque(maxlen=EVENTS_KEPT))

    @property
    def label(self) -> str:
        return self.robot_id or f"openpi-{self.session_id[:6]}"

    def event(self, kind: str, detail: str = "") -> None:
        self.events.append({"t": time.time(), "kind": kind, "detail": detail})

    def observe(self, obs: Any) -> None:
        self.obs_seq += 1
        self.last_obs = obs
        self.last_obs_at = time.time()
        if isinstance(obs, dict):
            episode_id = obs.get("episode_id")
            if episode_id is not None and episode_id != self.episode_id:
                self.event("episode", str(episode_id))
            self.instruction, self.episode_id = obs.get("instruction"), episode_id

    def record(self, ms: float, actions: Any) -> None:
        self.requests += 1
        self.latencies_ms.append(ms)
        self.last_server_ms = ms
        self.last_actions = actions
        self.stats.record(ms)
        self.backend.requests += 1

    def fail(self, code: str, message: str) -> None:
        self.errors += 1
        self.stats.fail()
        self.backend.errors += 1
        self.last_error = {"t": time.time(), "code": code, "message": message}
        self.event("error", f"{code}: {message}")

    def episode(self, success: bool) -> None:
        self.episodes += 1
        self.successes += int(success)
        self.stats.episodes += 1
        self.stats.successes += int(success)
        self.event("episode_result", "success" if success else "failure")

    def status(self) -> str:
        if self.closed_at is not None:
            return "closed"
        if not self.ready:
            return "connecting"
        if self.stop_reason is not None:
            return "stopping"
        if self.last_obs_at is None or time.time() - self.last_obs_at > IDLE_AFTER_S:
            return "idle"
        return "running"

    def robot_snapshot(self) -> dict:
        """The ``GET /v1/robots`` row."""
        return {
            "robot_id": self.robot_id,
            "model": self.model,
            "embodiment": self.embodiment,
            "backend": self.backend.url,
            "revision": self.backend.revision,
            "checkpoint": self.backend.checkpoint,
            "session_id": self.session_id,
            "connected_s": round(time.time() - self.connected_at, 1),
            "requests": self.requests,
            "server_p50_ms": _percentiles(self.latencies_ms)[0],
            "stop_pending": self.stop_reason is not None,
        }

    def summary(self) -> dict:
        now = time.time()
        p50, p99 = _percentiles(self.latencies_ms)
        return {
            "session_id": self.session_id,
            "label": self.label,
            "kind": self.kind,
            "robot_id": self.robot_id,
            "model": self.model,
            "embodiment": self.embodiment,
            "backend": self.backend.url,
            "revision": self.backend.revision,
            "checkpoint": self.backend.checkpoint,
            "peer": self.peer,
            "status": self.status(),
            "connected_s": round((self.closed_at or now) - self.connected_at, 1),
            "requests": self.requests,
            "errors": self.errors,
            "episodes": self.episodes,
            "success_rate": self.successes / self.episodes if self.episodes else None,
            "p50_ms": p50,
            "p99_ms": p99,
            "last_server_ms": self.last_server_ms,
            "obs_age_s": round(now - self.last_obs_at, 1) if self.last_obs_at else None,
            "instruction": self.instruction,
            "close_reason": self.close_reason,
            "can_stop": self.kind == "act" and self.closed_at is None,
        }

    def capture(self) -> dict:
        """Copy what a detail frame needs on the event loop; ``render_detail`` does the slow part."""
        return {
            "summary": self.summary(),
            "kind": self.kind,
            "obs": self.last_obs,
            "actions": self.last_actions,
            "episode_id": self.episode_id,
            "latencies_ms": list(self.latencies_ms)[-120:],
            "events": list(self.events),
            "last_error": self.last_error,
        }


def render_detail(captured: dict, frames: bool) -> dict:
    """Turn a captured view into a ``detail`` message: JPEG frames, state and the latest action chunk."""
    obs, actions = captured["obs"], captured["actions"]
    if captured["kind"] == "openpi":
        obs = _unpack(obs)
        actions = _unpack(actions)
        images = {k: v for k, v in obs.items() if "image" in k} if isinstance(obs, dict) else {}
        state = {}
        if isinstance(obs, dict):
            state = {k: v for k, v in obs.items() if "state" in k and isinstance(v, np.ndarray)}
        if isinstance(actions, dict):
            actions = {"actions": actions.get("actions")}
        elif isinstance(actions, np.ndarray):
            actions = {"actions": actions}
        instruction = obs.get("prompt") if isinstance(obs, dict) else None
    else:
        images = (obs or {}).get("images") or {}
        state = (obs or {}).get("state") or {}
        instruction = (obs or {}).get("instruction")
    summary = captured["summary"]
    if instruction is not None:
        summary = {**summary, "instruction": str(instruction)}
    message = {
        "type": "detail",
        "session_id": summary["session_id"],
        "summary": summary,
        "episode_id": captured["episode_id"],
        "state": {k: _floats(v) for k, v in state.items()},
        "actions": {k: _rows(v) for k, v in (actions or {}).items() if v is not None},
        "latencies_ms": [round(v, 2) for v in captured["latencies_ms"]],
        "events": captured["events"],
        "last_error": captured["last_error"],
    }
    if frames:
        message["frames"], problems = {}, []
        for role, value in images.items():
            try:
                message["frames"][role] = _jpeg_b64(value)
            except Exception as exc:  # a bad frame must not end the dashboard's stream
                problems.append(f"{role}: {exc}")
        if problems:
            message["frame_errors"] = problems
    return message


def _unpack(frame: Any) -> Any:
    if isinstance(frame, bytes):
        try:
            return msgpack_numpy.unpackb(frame)
        except Exception:
            return None
    return frame


def _floats(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)[:MAX_DIMS]
    return [round(float(v), 4) for v in arr]


def _rows(value: Any) -> list[list[float]]:
    arr = np.asarray(value, dtype=np.float64)
    arr = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr.reshape(-1, 1)
    return np.round(arr[:MAX_ACTION_ROWS, :MAX_DIMS], 4).tolist()


def _jpeg_b64(value: Any) -> str:
    if isinstance(value, dict) and value.get("enc") == "jpeg":
        data = bytes(value["data"])
    else:
        from PIL import Image

        image = np.asarray(value)
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = image.transpose(1, 2, 0)  # CHW -> HWC
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
    return base64.b64encode(data).decode()


@dataclass
class _Watch:
    session_id: str | None = None
    fps: float = DEFAULT_FPS
    next_detail: float = 0.0
    wake: asyncio.Event = field(default_factory=asyncio.Event)


class Monitor:
    def __init__(
        self,
        models: dict[str, list[Backend]],
        stats: dict[str, ModelStats],
        check_key: Callable[[str], bool],
    ) -> None:
        self.models, self.stats, self.check_key = models, stats, check_key
        self.sessions: dict[str, SessionView] = {}
        self.closed: deque[SessionView] = deque(maxlen=CLOSED_KEPT)

    def open(self, view: SessionView) -> SessionView:
        self.sessions[view.session_id] = view
        view.event("connected", view.peer)
        return view

    def close(self, view: SessionView, reason: str) -> None:
        if self.sessions.pop(view.session_id, None) is None:
            return
        view.closed_at, view.close_reason = time.time(), reason
        view.stop_reason = None
        view.event("closed", reason)
        self.closed.appendleft(view)

    def find(self, session_id: str | None) -> SessionView | None:
        if session_id in self.sessions:
            return self.sessions[session_id]
        return next((v for v in self.closed if v.session_id == session_id), None)

    def request_stop(self, view: SessionView, reason: str) -> None:
        view.stop_reason = reason
        view.event("stop_requested", reason)
        logger.warning("stop requested for robot %s: %s", view.robot_id, reason)

    def snapshot(self, since: int) -> dict:
        return {
            "type": "snapshot",
            "t": time.time(),
            "models": {
                model: {
                    **stats.snapshot(),
                    "backends": [b.snapshot() for b in self.models.get(model, [])],
                    "series": stats.history(since),
                }
                for model, stats in self.stats.items()
            },
            "sessions": [v.summary() for v in self.sessions.values()],
            "closed": [v.summary() for v in self.closed],
        }

    # -- websocket ---------------------------------------------------------------

    async def serve(self, client: ServerConnection) -> None:
        try:
            auth = json.loads(await asyncio.wait_for(client.recv(), AUTH_TIMEOUT_S))
        except (asyncio.TimeoutError, ConnectionClosed, ValueError, TypeError):
            await client.close(CLOSE_UNAUTHORIZED, "send {type: auth, key} first")
            return
        if not (isinstance(auth, dict) and auth.get("type") == "auth" and self.check_key(str(auth.get("key", "")))):
            await client.close(CLOSE_UNAUTHORIZED, "invalid API key")
            return
        hello = {
            "type": "hello",
            "series_s": SERIES_S,
            "models": {m: s.history() for m, s in self.stats.items()},
        }
        await client.send(json.dumps(hello))
        watch = _Watch()
        reader = asyncio.create_task(self._read(client, watch))
        try:
            await self._push(client, watch, reader)
        except ConnectionClosed:
            pass
        finally:
            reader.cancel()

    async def _read(self, client: ServerConnection, watch: _Watch) -> None:
        try:
            await self._read_loop(client, watch)
        except ConnectionClosed:
            pass

    async def _read_loop(self, client: ServerConnection, watch: _Watch) -> None:
        async for raw in client:
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            kind = msg.get("type") if isinstance(msg, dict) else None
            if kind == "watch":
                watch.session_id = str(msg.get("session_id"))
                watch.fps = _fps(msg.get("fps"))
                watch.next_detail = 0.0
                watch.wake.set()
            elif kind == "unwatch":
                watch.session_id = None
            elif kind == "stop":
                view = self.sessions.get(str(msg.get("session_id")))
                if view is None or view.kind != "act":
                    reply = {"type": "stop_ack", "ok": False, "message": "only live /act sessions can be stopped"}
                else:
                    self.request_stop(view, str(msg.get("reason") or "operator stop (dashboard)"))
                    reply = {"type": "stop_ack", "ok": True, "session_id": view.session_id}
                await client.send(json.dumps(reply))

    async def _push(self, client: ServerConnection, watch: _Watch, reader: asyncio.Task) -> None:
        last_t = int(time.time()) - 1
        next_snapshot = 0.0
        sent: tuple[str | None, int] = (None, -1)  # (session, obs_seq) of the last frames sent
        while not reader.done():
            now = time.monotonic()
            if now >= next_snapshot:
                snapshot = self.snapshot(last_t)
                for model in snapshot["models"].values():
                    if model["series"]:
                        last_t = max(last_t, model["series"][-1]["t"])
                await client.send(json.dumps(snapshot))
                next_snapshot = now + 1.0
            if watch.session_id is not None and now >= watch.next_detail:
                view = self.find(watch.session_id)
                if view is None:
                    await client.send(json.dumps({"type": "gone", "session_id": watch.session_id}))
                    watch.session_id = None
                else:
                    fresh = (view.session_id, view.obs_seq) != sent
                    detail = await asyncio.to_thread(render_detail, view.capture(), fresh)
                    if view.kind == "openpi":  # the prompt is only known once a frame is decoded
                        view.instruction = detail["summary"]["instruction"]
                    await client.send(json.dumps(detail))
                    if fresh:
                        sent = (view.session_id, view.obs_seq)
                    watch.next_detail = now + 1.0 / watch.fps
            due = min(next_snapshot, watch.next_detail) if watch.session_id is not None else next_snapshot
            try:
                await asyncio.wait_for(watch.wake.wait(), max(0.01, due - time.monotonic()))
            except asyncio.TimeoutError:
                pass
            watch.wake.clear()
