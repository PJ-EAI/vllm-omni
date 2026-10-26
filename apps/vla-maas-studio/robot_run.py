# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standard Robot protocol (``maas-robot/1``) runs for the VLA MaaS studio.

``studio.py --robot-config <robot.yaml>`` drives one robot file from
``packages/vla-maas-robot`` through the robot client's own pieces:
``RobotAdapter``, ``SafetyFilter``, ``PolicyClient`` and ``ControlLoop``. The
studio only adds observation hooks (state, commands, round trips, camera
frames) around them, so what the browser shows is what ``robotctl run`` does.

Two sources:

* default: a MuJoCo twin (``piper_sim.py``) behind the same ``driver:`` block,
  with rendered cameras under the YAML's camera roles; several sessions allowed.
* ``--hardware``: the YAML's real driver and cameras, exactly one session.
  ``slow`` and ``live`` need an explicit confirmation from the browser.

The safety filter, the hold-on-stop behaviour and the episode summary are the
robot client's; the studio adds no motion of its own.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for package in ("vla-maas-protocol", "vla-maas-robot"):
    path = str(REPO_ROOT / "packages" / package)
    if path not in sys.path:
        sys.path.insert(0, path)

from runner import FRAME_KIND_DETAIL, FRAME_KIND_THUMBNAIL  # noqa: E402
from vla_maas_protocol import PROTOCOL  # noqa: E402
from vla_maas_robot.adapter import RobotAdapter  # noqa: E402
from vla_maas_robot.cameras import Camera, make_camera  # noqa: E402
from vla_maas_robot.client import PolicyClient  # noqa: E402
from vla_maas_robot.config import RobotConfig  # noqa: E402
from vla_maas_robot.driver import Command, RobotDriver, RobotState, make_driver  # noqa: E402
from vla_maas_robot.loop import ControlLoop  # noqa: E402
from vla_maas_robot.safety import MODES, SafetyFilter  # noqa: E402

logger = logging.getLogger("vla_maas_studio")

THUMBNAIL_WIDTH = 160
DETAIL_HEIGHT = 360
MAX_SIM_ROBOTS = 8
# Driver kinds with a MuJoCo twin in the studio.
SIM_TWINS = {"piper": "Piper (MuJoCo twin)"}


def _round(values, digits: int = 3) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(values).reshape(-1)]


def _pct(values, q: float) -> float | None:
    return float(np.percentile(np.fromiter(values, dtype=np.float64), q)) if values else None


def build_robot_catalog(cfg: RobotConfig, hardware: bool, menagerie: str = "") -> dict:
    kind = str(cfg.driver.get("kind"))
    if not hardware:
        if kind not in SIM_TWINS:
            raise RuntimeError(
                f"driver kind {kind!r} has no MuJoCo twin in the studio (twins: {sorted(SIM_TWINS)}); "
                "add --hardware to use the robot file's own driver"
            )
        from piper_sim import CLONE_HINT, piper_model_path

        if not piper_model_path(menagerie).is_file():
            path = piper_model_path(menagerie)
            raise RuntimeError(f"no MuJoCo Menagerie Piper at {path}\nClone it with:\n  {CLONE_HINT}")
    physics = "hardware" if hardware else "mujoco-twin"
    label = f"{kind} (hardware)" if hardware else SIM_TWINS[kind]
    return {
        "physics": physics,
        "protocol": PROTOCOL,
        "max_robots": 1 if hardware else MAX_SIM_ROBOTS,
        "robots": [{"id": kind, "label": label, "physics": [physics]}],
        "suites": {},
        "safety_modes": list(MODES),
        "robot_config": {
            "robot_id": cfg.robot_id,
            "driver": kind,
            "cameras": {role: str(cam.get("kind")) for role, cam in cfg.cameras.items()},
            "model": cfg.gateway.model,
            "gateway": cfg.gateway.url,
            "start_mode": cfg.safety.start_mode,
            "hardware": hardware,
        },
    }


@dataclass
class ActRunConfig:
    model: str
    instruction: str
    mode: str = "dry_run"
    robots: int = 1
    episodes: int = 1
    max_steps: int = 600
    duration_s: float = 120.0
    control_hz: float = 0.0  # 0 = the model's control_hz from the handshake
    execute_steps: int = 0  # 0 = the robot file's control.execute_steps
    prefetch: int = -1  # -1 = the robot file's control.prefetch_threshold
    frame_fps: float = 5.0
    confirm_motion: bool = False


def _number(payload: dict, name: str, default, cast, low, high):
    try:
        value = cast(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def parse_act_config(payload: dict, catalog: dict) -> ActRunConfig:
    hardware = bool(catalog["robot_config"]["hardware"])
    model = str(payload.get("model") or catalog["robot_config"]["model"]).strip()
    if not model:
        raise ValueError("model is required")
    instruction = str(payload.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("instruction is required")
    mode = str(payload.get("mode", catalog["robot_config"]["start_mode"]))
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    confirm = bool(payload.get("confirm_motion", False))
    if hardware and mode != "dry_run" and not confirm:
        raise ValueError(f"mode {mode!r} moves the real arm: confirm the workspace is clear and the e-stop is at hand")
    return ActRunConfig(
        model=model,
        instruction=instruction,
        mode=mode,
        robots=_number(payload, "robots", 1, int, 1, int(catalog["max_robots"])),
        episodes=_number(payload, "episodes", 1, int, 1, 1000),
        max_steps=_number(payload, "max_steps", 600, int, 1, 100_000),
        duration_s=_number(payload, "duration_s", 120.0, float, 1.0, 3600.0),
        control_hz=_number(payload, "control_hz", 0.0, float, 0.0, 500.0),
        execute_steps=_number(payload, "execute_steps", 0, int, 0, 1000),
        prefetch=_number(payload, "prefetch", -1, int, -1, 1000),
        frame_fps=_number(payload, "frame_fps", 5.0, float, 0.0, 30.0),
        confirm_motion=confirm,
    )


class ActTelemetry:
    """Per-session counters fed by the observation hooks; read by the web server."""

    def __init__(self, robots: int) -> None:
        self.lock = threading.Lock()
        self.started = time.perf_counter()
        self.rtts: deque[float] = deque(maxlen=8192)
        self.sessions = {i: self._new(i) for i in range(robots)}

    @staticmethod
    def _new(i: int) -> dict:
        return {
            "id": i,
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
            "last_action": [],
            "backend": "",
            "model": "",
            "error": "",
            "robot": {"commands_sent": 0, "inferences": 0, "holds": 0, "dropped_rows": 0},
            "_rtt": deque(maxlen=64),
            "_server": deque(maxlen=64),
            "_safety": None,
        }

    def update(self, i: int, **fields) -> None:
        with self.lock:
            self.sessions[i].update(fields)

    def robot(self, i: int, **fields) -> None:
        with self.lock:
            self.sessions[i]["robot"].update(fields)

    def state(self, i: int, state: RobotState) -> None:
        with self.lock:
            session = self.sessions[i]
            session["ticks"] += 1
            robot = session["robot"]
            robot["joint_pos"] = _round(state.joint_pos)
            robot["gripper_raw"] = None if state.gripper is None else round(float(state.gripper), 4)
            robot["eef_xyz"] = None if state.eef_pose is None else _round(state.eef_pose[:3])

    def command(self, i: int, cmd: Command) -> None:
        with self.lock:
            session = self.sessions[i]
            target = _round(cmd.target)
            session["last_action"] = target + ([] if cmd.gripper is None else [round(float(cmd.gripper), 4)])
            robot = session["robot"]
            robot["commands_sent"] += 1
            robot["last_command"] = {"mode": cmd.mode, "target": target, "gripper_raw": cmd.gripper}

    def reply(self, i: int, rtt_ms: float, server_ms: float | None) -> None:
        with self.lock:
            session = self.sessions[i]
            session["last_latency_ms"] = rtt_ms
            session["_rtt"].append(rtt_ms)
            session["robot"]["inferences"] += 1
            if server_ms is not None:
                session["_server"].append(float(server_ms))
            self.rtts.append(rtt_ms)

    def episode_done(self, i: int, summary: dict) -> None:
        with self.lock:
            session = self.sessions[i]
            session["episodes"] += 1
            session["episode"] = session["episodes"]
            session["late_ticks"] += round(summary["late_tick_pct"] * summary["ticks"] / 100)
            session["underruns"] += summary["underruns"]
            robot = session["robot"]
            robot["holds"] += summary["holds"]
            robot["dropped_rows"] += summary["dropped_rows"]
            robot["last_episode"] = summary
            robot["stopped"] = summary["stopped"]

    def attach_safety(self, i: int, safety: SafetyFilter) -> None:
        with self.lock:
            self.sessions[i]["_safety"] = safety

    @staticmethod
    def _safety(safety: SafetyFilter) -> dict:
        # Counters written by the control thread; a snapshot may be one tick behind.
        stats = safety.stats
        return {
            "mode": safety.mode,
            "commands": stats.commands,
            "clamped": stats.clamped,
            "worst_streak": stats.worst_streak,
            "reasons": dict(stats.reasons),
        }

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = max(time.perf_counter() - self.started, 1e-6)
            sessions = []
            for session in self.sessions.values():
                item = {k: v for k, v in session.items() if not k.startswith("_")}
                item["robot"] = dict(session["robot"])
                if session["_safety"] is not None:
                    item["robot"]["safety"] = self._safety(session["_safety"])
                item["latency_p50_ms"] = _pct(session["_rtt"], 50)
                server = _pct(session["_server"], 50)
                item["robot"]["server_p50_ms"] = server
                rtt = item["latency_p50_ms"]
                item["robot"]["network_p50_ms"] = None if rtt is None or server is None else rtt - server
                sessions.append(item)
            inferences = len(self.rtts)
            ticks = sum(s["ticks"] for s in sessions)
            late = sum(s["late_ticks"] for s in sessions)
            episodes = sum(s["episodes"] for s in sessions)
            rtt_p50, rtt_p99 = _pct(self.rtts, 50), _pct(self.rtts, 99)
        return {
            "elapsed_s": round(elapsed, 1),
            "active": sum(s["state"] in ("starting", "connecting", "running") for s in sessions),
            "errors": sum(s["state"] == "error" for s in sessions),
            "global": {
                "inferences": inferences,
                "throughput_rps": round(inferences / elapsed, 1),
                "latency_p50_ms": rtt_p50,
                "latency_p99_ms": rtt_p99,
                "underruns": sum(s["underruns"] for s in sessions),
                "late_tick_pct": round(100.0 * late / ticks, 2) if ticks else None,
                "episodes": episodes,
                "success_rate": None,  # nothing judges success here; robotctl --ask-success does
            },
            "sessions": sessions,
        }


class ObservedDriver(RobotDriver):
    """Passes everything to ``inner`` and reports state and sent commands."""

    def __init__(self, inner: RobotDriver, telemetry: ActTelemetry, session: int) -> None:
        self.inner, self.telemetry, self.session = inner, telemetry, session
        self.spec = inner.spec

    def connect(self) -> None:
        self.inner.connect()

    def read_state(self) -> RobotState:
        state = self.inner.read_state()
        self.telemetry.state(self.session, state)
        return state

    def command(self, cmd: Command) -> None:
        self.inner.command(cmd)
        self.telemetry.command(self.session, cmd)

    def hold(self) -> None:
        self.inner.hold()

    def home(self) -> None:
        self.inner.home()

    def close(self) -> None:
        self.inner.close()


class ObservedClient(PolicyClient):
    """``PolicyClient`` that reports each round trip; the loop calls it off the control thread."""

    def __init__(self, *args, telemetry: ActTelemetry, session: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.telemetry, self.session = telemetry, session

    def request(self, msg: dict) -> dict:
        reply = super().request(msg)
        if msg.get("type") == "obs" and self.last_rtt_ms is not None:
            self.telemetry.reply(self.session, self.last_rtt_ms, self.last_server_ms)
        return reply


def _http_base(url: str) -> str:
    if "://" not in url:
        url = f"http://{url}"
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return f"{scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def gateway_http_url(cli_gateway: str | None, cfg: RobotConfig) -> str:
    """The studio's ``--gateway`` if given, else the robot file's ``gateway.url``, as http(s)."""
    return _http_base(cli_gateway or cfg.gateway.url)


def _encode(image: np.ndarray, width: int | None = None, height: int | None = None, quality: int = 75) -> bytes:
    from PIL import Image

    picture = Image.fromarray(image)
    if width is not None:
        picture = picture.resize((width, max(1, round(picture.height * width / picture.width))))
    elif height is not None and picture.height != height:
        picture = picture.resize((max(1, round(picture.width * height / picture.height)), height))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _side_by_side(frames: list[np.ndarray], height: int) -> np.ndarray:
    from PIL import Image

    tiles = []
    for frame in frames:
        picture = Image.fromarray(frame)
        width = max(1, round(picture.width * height / picture.height))
        tiles.append(np.asarray(picture.resize((width, height))))
    return np.concatenate(tiles, axis=1)


class ActRun:
    """N Robot-protocol sessions; same surface as ``runner.Run`` for ``studio.py``."""

    def __init__(
        self,
        config: ActRunConfig,
        robot_cfg: RobotConfig,
        *,
        gateway: str,
        api_key: str | None,
        hardware: bool,
        menagerie: str = "",
    ) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.config = config
        self.robot_cfg = robot_cfg
        self.gateway = gateway
        self.api_key = api_key
        self.hardware = hardware
        self.menagerie = menagerie
        self.telemetry = ActTelemetry(config.robots)
        self.stop_event = threading.Event()
        self.frames: deque[tuple[int, int, bytes]] = deque(maxlen=1024)
        self.frames_lock = threading.Lock()
        self.watched: set[int] = set()
        self.watched_lock = threading.Lock()
        self.cameras: dict[int, dict[str, Camera]] = {}
        self.threads: list[threading.Thread] = []
        self._frame_thread: threading.Thread | None = None
        self._timer: threading.Timer | None = None

    def robot_id(self, session: int) -> str:
        base = self.robot_cfg.robot_id
        return base if self.config.robots == 1 else f"{base}-{session}"

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self.threads)

    def start(self) -> None:
        self._timer = threading.Timer(self.config.duration_s, self.stop_event.set)
        self._timer.daemon = True
        self._timer.start()
        for i in range(self.config.robots):
            thread = threading.Thread(target=self._session, args=(i,), name=f"vla-studio-robot-{i}", daemon=True)
            thread.start()
            self.threads.append(thread)
        self._frame_thread = threading.Thread(target=self._frame_loop, name="vla-studio-robot-frames", daemon=True)
        self._frame_thread.start()

    def stop(self) -> None:
        # The control loop sees the event at its next tick, holds the arm and returns.
        self.stop_event.set()
        if self._timer is not None:
            self._timer.cancel()
        deadline = time.perf_counter() + self.robot_cfg.gateway.request_timeout_s + 5.0
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.perf_counter()))
        if self._frame_thread is not None:
            self._frame_thread.join(timeout=2.0)

    def set_watched(self, session_ids: set[int]) -> None:
        with self.watched_lock:
            self.watched = set(session_ids)

    def drain_frames(self) -> list[tuple[int, int, bytes]]:
        with self.frames_lock:
            frames = list(self.frames)
            self.frames.clear()
        return frames

    def snapshot(self) -> dict:
        payload = self.telemetry.snapshot()
        payload.update(
            {
                "id": self.id,
                "running": self.running,
                "protocol": PROTOCOL,
                "hardware": self.hardware,
                "config": {**asdict(self.config), "robot_id": self.robot_cfg.robot_id},
            }
        )
        return payload

    # -- sessions ---------------------------------------------------------------

    def _driver_and_cameras(self, stack: ExitStack) -> tuple[RobotDriver, dict[str, Camera]]:
        cfg = self.robot_cfg
        if self.hardware:
            driver = make_driver(cfg.driver)
            cameras = {role: make_camera(cam) for role, cam in cfg.cameras.items()}
        else:
            from piper_sim import PiperSimDriver, SimCamera

            sizes = [(int(cam.get("width", 640)), int(cam.get("height", 480))) for cam in cfg.cameras.values()]
            max_render = (max([w for w, _ in sizes] or [640]), max([h for _, h in sizes] or [480]))
            driver_cfg = {k: v for k, v in cfg.driver.items() if k != "kind"}
            driver = PiperSimDriver(menagerie=self.menagerie, max_render=max_render, **driver_cfg)
            cameras = {
                role: SimCamera(driver, role, cam.get("width", 640), cam.get("height", 480), cam.get("fps", 30))
                for role, cam in cfg.cameras.items()
            }
        driver.connect()
        stack.callback(driver.close)
        for camera in cameras.values():
            camera.start()
            stack.callback(camera.stop)
        return driver, cameras

    def _session(self, i: int) -> None:
        cfg, run = self.robot_cfg, self.config
        tel = self.telemetry
        tel.update(i, state="connecting", language=run.instruction, model=run.model, backend=self.gateway)
        try:
            with ExitStack() as stack:
                driver, cameras = self._driver_and_cameras(stack)
                self.cameras[i] = cameras
                observed = ObservedDriver(driver, tel, i)
                adapter = RobotAdapter(driver.spec, list(cameras), cfg.calibration)
                client = ObservedClient(
                    self.gateway,
                    run.model,
                    self.robot_id(i),
                    self.api_key,
                    embodiment=driver.spec.embodiment,
                    request_timeout_s=cfg.gateway.request_timeout_s,
                    telemetry=tel,
                    session=i,
                )
                server = client.connect(adapter.hello(self.robot_id(i), jpeg=cfg.control.jpeg_quality is not None))
                stack.callback(client.close)
                adapter.configure(server)
                overrides = {}
                if run.control_hz > 0:
                    overrides["control_hz"] = run.control_hz
                if run.execute_steps > 0:
                    overrides["execute_steps"] = run.execute_steps
                if run.prefetch >= 0:
                    overrides["prefetch_threshold"] = run.prefetch
                loop_cfg = dataclasses.replace(cfg.control, **overrides)
                hz = loop_cfg.control_hz or server.provides.control_hz
                safety = SafetyFilter(driver.spec, cfg.safety, hz)
                safety.mode = run.mode
                tel.attach_safety(i, safety)
                loop = ControlLoop(observed, cameras, adapter, safety, client, loop_cfg, log=logger.info)
                tel.update(i, state="running")
                tel.robot(
                    i,
                    robot_id=self.robot_id(i),
                    embodiment=driver.spec.embodiment,
                    dof=driver.spec.dof,
                    session_id=client.session_id,
                    control_hz=hz,
                    requires={"images": server.requires.images, "state": server.requires.state},
                    provides=asdict(server.provides),
                )
                for _ in range(run.episodes):
                    if self.stop_event.is_set():
                        break
                    stats = loop.run_episode(run.instruction, run.max_steps, self.stop_event)
                    tel.episode_done(i, stats.summary())
                    if stats.stopped and stats.stopped != "operator":
                        tel.update(i, state="error", error=stats.stopped)
                        return
                tel.update(i, state="done")
        except Exception as exc:  # one robot failing must not take the run down
            logger.exception("robot session %d failed", i)
            tel.update(i, state="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.cameras.pop(i, None)

    # -- frames -----------------------------------------------------------------

    def _frame_loop(self) -> None:
        if self.config.frame_fps <= 0:
            return
        period = 1.0 / self.config.frame_fps
        while not self.stop_event.is_set() or self.running:
            start = time.perf_counter()
            with self.watched_lock:
                watched = set(self.watched)
            for i, cameras in list(self.cameras.items()):
                try:
                    frames = [camera.read()[1] for camera in cameras.values()]
                except RuntimeError:
                    continue  # not started yet, or failed; the session reports it
                if not frames:
                    continue
                self._push(FRAME_KIND_THUMBNAIL, i, _encode(frames[0], width=THUMBNAIL_WIDTH, quality=70))
                if i in watched:
                    self._push(FRAME_KIND_DETAIL, i, _encode(_side_by_side(frames, DETAIL_HEIGHT), quality=80))
            time.sleep(max(0.0, period - (time.perf_counter() - start)))

    def _push(self, kind: int, i: int, jpeg: bytes) -> None:
        with self.frames_lock:
            self.frames.append((kind, i, jpeg))
