# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run orchestration for the VLA MaaS studio.

A run is N independent OpenPI sessions, each with its own simulator and control
loop. Physics runs in session threads; the web server only consumes snapshots
and JPEG frames, so the MaaS service -- not the studio -- is the thing under
test.

This is a debugging console, not a timing benchmark: sessions share one Python
process, so use ``examples/online_serving/vla_maas/fleet_sim.py`` when latency
numbers must be free of client-side GIL contention.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from scenes import ACTION_DIM, create_scene
from vla_client import AsyncChunkExecutor, connect_policy

MAX_ROBOTS = 64
THUMBNAIL_SIZE = 128
FRAME_KIND_THUMBNAIL = 1
FRAME_KIND_DETAIL = 2


def _encode_jpeg(image: np.ndarray, size: int | None, quality: int = 70) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - PIL ships with the server extras
        return None
    picture = Image.fromarray(image)
    if size is not None and picture.width != size:
        picture = picture.resize((size, size), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


@dataclass(frozen=True)
class Endpoint:
    url: str
    api_key: str | None


@dataclass
class RunConfig:
    target: str = "gateway"
    model: str = ""
    direct_url: str = ""
    physics: str = "mock"
    robot: str = "panda"
    menagerie: str = ""
    suite: str = "libero_spatial"
    task_ids: list[int] = field(default_factory=list)
    robots: int = 8
    duration_s: float = 30.0
    control_hz: float = 50.0
    replan_steps: int = 20
    prefetch: int = 10
    resolution: int = 256
    frame_fps: float = 2.0
    base_camera_key: str = "observation.images.image"
    wrist_camera_key: str = "observation.images.image2"
    action_scale: float = 0.005
    seed: int = 0
    report_episodes: bool = True
    stagger_s: float = 0.01


def _number(payload: dict, name: str, default, cast, low, high):
    try:
        value = cast(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def parse_run_config(payload: dict, catalog: dict) -> RunConfig:
    """Validate the browser payload against the server's scene catalog."""
    physics = catalog["physics"]
    suites = catalog["suites"]

    suite = str(payload.get("suite", "libero_spatial"))
    if suite not in suites:
        raise ValueError(f"unknown task suite {suite!r}")

    known_tasks = {task["id"] for task in suites[suite]["tasks"]}
    raw_tasks = payload.get("task_ids")
    if raw_tasks is None:
        task_ids = sorted(known_tasks)
    else:
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("task_ids must be a non-empty list")
        task_ids = []
        for value in raw_tasks:
            try:
                task_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("task_ids must contain integers") from exc
            if task_id not in known_tasks:
                raise ValueError(f"task {task_id} does not exist in {suite!r}")
            if task_id not in task_ids:
                task_ids.append(task_id)

    target = str(payload.get("target", "gateway"))
    if target not in ("gateway", "direct"):
        raise ValueError("target must be 'gateway' or 'direct'")

    model = str(payload.get("model", "")).strip()
    direct_url = str(payload.get("direct_url", "")).strip()
    if target == "gateway" and not model:
        raise ValueError("model is required when target is 'gateway'")
    if target == "direct" and not direct_url.startswith(("ws://", "wss://")):
        raise ValueError("direct_url must start with ws:// or wss://")

    base_camera_key = str(payload.get("base_camera_key", "observation.images.image")).strip()
    wrist_camera_key = str(payload.get("wrist_camera_key", "observation.images.image2")).strip()
    if not base_camera_key or not wrist_camera_key:
        raise ValueError("camera keys must not be empty")

    robots = {entry["id"]: entry for entry in catalog["robots"]}
    robot = str(payload.get("robot", next(iter(robots), "panda")))
    if robot not in robots:
        raise ValueError(f"unknown robot {robot!r}")
    if physics not in robots[robot].get("physics", []):
        raise ValueError(f"robot {robot!r} is not available for physics={physics!r}")

    max_robots = int(catalog.get("max_robots", MAX_ROBOTS))
    config = RunConfig(
        target=target,
        model=model,
        direct_url=direct_url,
        physics=physics,
        robot=robot,
        suite=suite,
        task_ids=task_ids,
        robots=_number(payload, "robots", 8, int, 1, max_robots),
        duration_s=_number(payload, "duration_s", 30.0, float, 1.0, 3600.0),
        control_hz=_number(payload, "control_hz", 50.0, float, 0.0, 120.0),
        replan_steps=_number(payload, "replan_steps", 20, int, 1, 100),
        prefetch=_number(payload, "prefetch", 10, int, 0, 100),
        resolution=_number(payload, "resolution", 256, int, 64, 512),
        frame_fps=_number(payload, "frame_fps", 2.0, float, 0.0, 30.0),
        base_camera_key=base_camera_key,
        wrist_camera_key=wrist_camera_key,
        action_scale=_number(payload, "action_scale", 0.005, float, 0.0001, 0.1),
        seed=_number(payload, "seed", 0, int, 0, 2**31 - 1),
        report_episodes=bool(payload.get("report_episodes", True)),
    )
    return config


def resolve_endpoint(config: RunConfig, gateway_url: str, api_key: str | None) -> Endpoint:
    if config.target == "direct":
        return Endpoint(config.direct_url, None)

    url = gateway_url if "://" in gateway_url else f"http://{gateway_url}"
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"invalid gateway URL {gateway_url!r}")
    secure = parts.scheme == "https"
    port = parts.port or (443 if secure else 80)
    return Endpoint(
        url=f"{'wss' if secure else 'ws'}://{parts.hostname}:{port}/v1/robot/{config.model}/openpi",
        api_key=api_key,
    )


class Telemetry:
    """Thread-safe run state shared by session threads and the web server."""

    def __init__(self, config: RunConfig) -> None:
        self.lock = threading.Lock()
        self.config = config
        self.started = time.perf_counter()
        self.sessions = {robot_id: self._new_session(robot_id) for robot_id in range(config.robots)}
        self.latencies: deque[float] = deque(maxlen=8192)
        self.inferences = 0
        self.episodes = 0
        self.successes = 0
        self.ticks = 0
        self.late_ticks = 0

    @staticmethod
    def _new_session(robot_id: int) -> dict:
        return {
            "id": robot_id,
            "state": "starting",
            "task_id": None,
            "language": "",
            "episode": 0,
            "episodes": 0,
            "successes": 0,
            "ticks": 0,
            "late_ticks": 0,
            "underruns": 0,
            "last_latency_ms": None,
            "lat_recent": deque(maxlen=64),
            "last_action": [],
            "backend": "",
            "model": "",
            "error": "",
            "_lat_seen": 0,
        }

    def update(self, robot_id: int, **fields) -> None:
        with self.lock:
            self.sessions[robot_id].update(fields)

    def tick(self, robot_id: int, *, late: bool, action: np.ndarray, executor: AsyncChunkExecutor) -> None:
        with self.lock:
            session = self.sessions[robot_id]
            session["ticks"] += 1
            self.ticks += 1
            if late:
                session["late_ticks"] += 1
                self.late_ticks += 1
            session["underruns"] = executor.underruns

            seen = session["_lat_seen"]
            new_values = executor.latencies_ms[seen:]
            for value in new_values:
                self.latencies.append(value)
                session["lat_recent"].append(value)
                self.inferences += 1
            if new_values:
                session["last_latency_ms"] = new_values[-1]
            session["_lat_seen"] = len(executor.latencies_ms)
            session["last_action"] = [round(float(value), 3) for value in np.asarray(action).reshape(-1)[:7]]

    def episode_done(self, robot_id: int, success: bool) -> None:
        with self.lock:
            session = self.sessions[robot_id]
            session["episodes"] += 1
            session["successes"] += int(success)
            session["episode"] = session["episodes"]
            self.episodes += 1
            self.successes += int(success)

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = max(time.perf_counter() - self.started, 1e-6)
            inferences = self.inferences
            episodes = self.episodes
            successes = self.successes
            ticks = self.ticks
            late_ticks = self.late_ticks
            sessions = []
            for session in self.sessions.values():
                recent = np.fromiter(session["lat_recent"], dtype=np.float64) if session["lat_recent"] else None
                item = {key: value for key, value in session.items() if not key.startswith("_") and key != "lat_recent"}
                item["latency_p50_ms"] = float(np.percentile(recent, 50)) if recent is not None else None
                sessions.append(item)
            latencies = np.fromiter(self.latencies, dtype=np.float64) if self.latencies else None
            underruns = sum(session["underruns"] for session in sessions)
            errors = sum(session["state"] == "error" for session in sessions)
            active = sum(session["state"] in ("starting", "running") for session in sessions)

        return {
            "elapsed_s": round(elapsed, 1),
            "active": active,
            "errors": errors,
            "global": {
                "inferences": inferences,
                "throughput_rps": round(inferences / elapsed, 1),
                "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies is not None else None,
                "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies is not None else None,
                "underruns": underruns,
                "late_tick_pct": (
                    round(100.0 * late_ticks / ticks, 2) if ticks and self.config.control_hz > 0 else None
                ),
                "episodes": episodes,
                "success_rate": round(successes / episodes, 3) if episodes else None,
            },
            "sessions": sessions,
        }


class Run:
    """N concurrent sessions against one OpenPI endpoint."""

    def __init__(self, config: RunConfig, endpoint: Endpoint) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.config = config
        self.endpoint = endpoint
        self.telemetry = Telemetry(config)
        self.stop_event = threading.Event()
        self.frames: deque[tuple[int, int, bytes]] = deque(maxlen=4096)
        self.frames_lock = threading.Lock()
        self.watched: set[int] = set()
        self.watched_lock = threading.Lock()
        self.threads: list[threading.Thread] = []
        self._last_frame: dict[int, float] = {}
        # JPEG encoding runs off the control thread so a thumbnail never shows
        # up as a late tick.
        self._frame_requests: queue.Queue = queue.Queue(maxsize=256)
        self._encoder: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self.threads)

    def start(self) -> None:
        self._encoder = threading.Thread(target=self._encode_loop, name="vla-studio-frames", daemon=True)
        self._encoder.start()
        for robot_id in range(self.config.robots):
            thread = threading.Thread(
                target=self._session_loop,
                args=(robot_id,),
                name=f"vla-studio-session-{robot_id}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)
            time.sleep(self.config.stagger_s)

    def stop(self) -> None:
        self.stop_event.set()
        deadline = time.perf_counter() + 6.0
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.perf_counter()))
        if self._encoder is not None:
            while True:
                try:
                    self._frame_requests.get_nowait()
                except queue.Empty:
                    break
            try:
                self._frame_requests.put_nowait(None)
            except queue.Full:  # pragma: no cover - queue was just drained
                pass
            self._encoder.join(timeout=2.0)

    def set_watched(self, session_ids: set[int]) -> None:
        with self.watched_lock:
            self.watched = set(session_ids)

    def drain_frames(self) -> list[tuple[int, int, bytes]]:
        with self.frames_lock:
            frames = list(self.frames)
            self.frames.clear()
        return frames

    def _push_frame(self, kind: int, robot_id: int, jpeg: bytes) -> None:
        with self.frames_lock:
            self.frames.append((kind, robot_id, jpeg))

    def snapshot(self) -> dict:
        payload = self.telemetry.snapshot()
        payload.update(
            {
                "id": self.id,
                "running": self.running,
                "config": asdict(self.config),
            }
        )
        return payload

    def _encode_loop(self) -> None:
        while True:
            item = self._frame_requests.get()
            if item is None:
                return
            robot_id, image, wants_detail = item
            thumbnail = _encode_jpeg(image, THUMBNAIL_SIZE)
            if thumbnail is not None:
                self._push_frame(FRAME_KIND_THUMBNAIL, robot_id, thumbnail)
            if wants_detail:
                detail = _encode_jpeg(image, None, quality=80)
                if detail is not None:
                    self._push_frame(FRAME_KIND_DETAIL, robot_id, detail)

    def _maybe_push_frame(self, robot_id: int, scene: Any, raw: dict) -> None:
        if self.config.frame_fps <= 0:
            return
        now = time.perf_counter()
        if now - self._last_frame.get(robot_id, 0.0) < 1.0 / self.config.frame_fps:
            return
        self._last_frame[robot_id] = now
        with self.watched_lock:
            wants_detail = robot_id in self.watched
        image = scene.render(raw)
        try:
            self._frame_requests.put_nowait((robot_id, image, wants_detail))
        except queue.Full:
            pass

    def _session_loop(self, robot_id: int) -> None:
        scene: Any = None
        policy = None
        try:
            self.telemetry.update(
                robot_id,
                state="connecting",
                task_id=self.config.task_ids[robot_id % len(self.config.task_ids)],
            )
            policy, metadata = connect_policy(
                self.endpoint.url,
                0,
                "",
                api_key=self.endpoint.api_key,
            )
            scene = create_scene(
                physics=self.config.physics,
                suite_name=self.config.suite,
                task_id=self.config.task_ids[robot_id % len(self.config.task_ids)],
                robot=self.config.robot,
                resolution=self.config.resolution,
                action_scale=self.config.action_scale,
                seed=self.config.seed + robot_id,
                menagerie=self.config.menagerie,
            )
            self.telemetry.update(
                robot_id,
                state="running",
                task_id=scene.task_id,
                language=scene.language,
                backend=str(metadata.get("backend", "")),
                model=str(metadata.get("model", self.config.model)),
            )
            executor = AsyncChunkExecutor(
                policy,
                action_dim=ACTION_DIM,
                prefetch_threshold=self.config.prefetch,
                execute_steps=self.config.replan_steps,
            )
            period = 1.0 / self.config.control_hz if self.config.control_hz > 0 else 0.0
            stop_at = time.perf_counter() + self.config.duration_s
            episode = 0
            while not self.stop_event.is_set() and time.perf_counter() < stop_at:
                init_state = scene.init_state(episode)
                raw = scene.reset(init_state)
                executor.reset()
                session_id = str(uuid.uuid4())
                success = False
                completed = True
                for _ in range(scene.max_steps):
                    if self.stop_event.is_set() or time.perf_counter() >= stop_at:
                        completed = False
                        break
                    tick_start = time.perf_counter()
                    action = executor.next_action(
                        lambda raw=raw: scene.policy_obs(
                            raw,
                            scene.language,
                            session_id,
                            self.config.base_camera_key,
                            self.config.wrist_camera_key,
                        )
                    )
                    raw, done = scene.step(action)
                    elapsed = time.perf_counter() - tick_start
                    self.telemetry.tick(
                        robot_id,
                        late=bool(period and elapsed > period),
                        action=action,
                        executor=executor,
                    )
                    self._maybe_push_frame(robot_id, scene, raw)
                    if done:
                        success = True
                        break
                    if period:
                        time.sleep(max(0.0, period - elapsed))
                if completed:
                    self.telemetry.episode_done(robot_id, success)
                    if self.config.report_episodes and self.config.target == "gateway":
                        executor.wait()
                        policy.infer({"endpoint": "episode_result", "success": success})
                episode += 1
            self.telemetry.update(robot_id, state="done")
        except BaseException as exc:  # one robot failing must not take the run down
            self.telemetry.update(robot_id, state="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if scene is not None:
                try:
                    scene.close()
                except BaseException:
                    pass
            if policy is not None:
                # openpi-client exposes no close(); release the socket so the
                # gateway session count drops when the run ends.
                connection = getattr(policy, "_ws", None)
                if connection is not None:
                    try:
                        connection.close()
                    except BaseException:
                        pass
