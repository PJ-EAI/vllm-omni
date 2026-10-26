# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-rate control loop over remote action chunks.

Each tick reads the robot, sends one command and, when few actions remain,
asks the gateway for the next chunk on a background thread. Chunks are
aligned to ticks: row ``i`` of a chunk is the target for ``obs_tick + i``, so
the rows that elapsed while the request was in flight are dropped instead of
replayed. Optionally the first ``blend_steps`` rows are blended with what was
left of the previous chunk.
"""

from __future__ import annotations

import collections
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from vla_maas_protocol import geometry as geo

from vla_maas_robot.adapter import RobotAdapter
from vla_maas_robot.cameras import Camera
from vla_maas_robot.client import PolicyClient
from vla_maas_robot.driver import Command, RobotDriver
from vla_maas_robot.safety import SafetyError, SafetyFilter


@dataclass
class LoopConfig:
    control_hz: float | None = None  # None = the model's control_hz from the handshake
    execute_steps: int | None = None  # use at most this many rows of each chunk
    prefetch_threshold: int = 10  # request the next chunk when this many rows remain
    blend: str = "none"  # none | linear
    blend_steps: int = 5
    jpeg_quality: int | None = None  # set (e.g. 90) to send JPEG over slow links
    max_frame_skew_ms: float = 50.0  # warn when a camera frame is older than the state by this much

    @classmethod
    def from_dict(cls, d: dict | None) -> LoopConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown control keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.blend not in ("none", "linear"):
            raise ValueError(f"blend must be none or linear, got {cfg.blend!r}")
        return cfg


def _blend(old: Command, new: Command, w: float) -> Command:
    a, b = np.asarray(old.target), np.asarray(new.target)
    if new.mode == "eef_pose":
        q = geo.quat_normalize((1 - w) * a[3:7] + w * (b[3:7] if np.dot(a[3:7], b[3:7]) >= 0 else -b[3:7]))
        target = np.concatenate(((1 - w) * a[:3] + w * b[:3], q))
    else:
        target = (1 - w) * a + w * b
    gripper = new.gripper if old.gripper is None or new.gripper is None else (1 - w) * old.gripper + w * new.gripper
    return Command(mode=new.mode, target=target, gripper=gripper, meta=new.meta)


class ChunkQueue:
    def __init__(self, blend_steps: int = 0) -> None:
        self.blend_steps = blend_steps
        self._rows: collections.deque[Command] = collections.deque()

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()

    def install(self, chunk: list[Command], obs_tick: int, now_tick: int) -> int:
        """Replace the queue with ``chunk`` aligned to ``now_tick``; returns rows dropped."""
        dropped = max(0, now_tick - obs_tick)
        rows = list(chunk[dropped:])
        old = list(self._rows)
        for i in range(min(self.blend_steps, len(old), len(rows))):
            if old[i].mode == rows[i].mode:
                rows[i] = _blend(old[i], rows[i], (i + 1) / (self.blend_steps + 1))
        self._rows = collections.deque(rows)
        return min(dropped, len(chunk))

    def pop(self) -> Command | None:
        return self._rows.popleft() if self._rows else None


@dataclass
class EpisodeStats:
    episode_id: str
    ticks: int = 0
    commands_sent: int = 0
    late_ticks: int = 0
    underruns: int = 0
    holds: int = 0
    inferences: int = 0
    dropped_rows: int = 0
    rtt_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    frame_skew_warnings: int = 0
    stopped: str | None = None  # why the episode ended early

    def summary(self) -> dict:
        def pct(xs: list[float], q: float) -> float | None:
            return round(float(np.percentile(xs, q)), 1) if xs else None

        net = [r - s for r, s in zip(self.rtt_ms, self.server_ms)]
        return {
            "episode_id": self.episode_id,
            "ticks": self.ticks,
            "commands_sent": self.commands_sent,
            "inferences": self.inferences,
            "late_tick_pct": round(100 * self.late_ticks / max(self.ticks, 1), 2),
            "underruns": self.underruns,
            "holds": self.holds,
            "dropped_rows": self.dropped_rows,
            "rtt_p50_ms": pct(self.rtt_ms, 50),
            "rtt_p99_ms": pct(self.rtt_ms, 99),
            "server_p50_ms": pct(self.server_ms, 50),
            "network_p50_ms": pct(net, 50),
            "frame_skew_warnings": self.frame_skew_warnings,
            "stopped": self.stopped,
        }


class ControlLoop:
    def __init__(
        self,
        driver: RobotDriver,
        cameras: dict[str, Camera],
        adapter: RobotAdapter,
        safety: SafetyFilter,
        client: PolicyClient,
        cfg: LoopConfig,
        log: Callable[[str], None] = print,
    ) -> None:
        self.driver, self.cameras, self.adapter = driver, cameras, adapter
        self.safety, self.client, self.cfg, self.log = safety, client, cfg, log
        provides = client.server.provides if client.server else None
        self.control_hz = cfg.control_hz or (provides.control_hz if provides else 10.0)
        self.queue = ChunkQueue(cfg.blend_steps if cfg.blend == "linear" else 0)
        self._seq = 0

    def _frames(self, stats: EpisodeStats, state_t: float) -> dict:
        frames = {role: cam.read() for role, cam in self.cameras.items()}
        if any((state_t - t) * 1e3 > self.cfg.max_frame_skew_ms for t, _ in frames.values()):
            stats.frame_skew_warnings += 1
        return frames

    def run_episode(self, instruction: str, max_steps: int, stop: threading.Event | None = None) -> EpisodeStats:
        stats = EpisodeStats(episode_id=str(uuid.uuid4()))
        stop = stop or threading.Event()
        period = 1.0 / self.control_hz
        self.queue.clear()
        self.safety.reset()
        pending: dict = {}
        thread: threading.Thread | None = None
        last_command_at = time.monotonic()
        primed = False

        def fetch(obs: dict, obs_tick: int) -> None:
            try:
                reply = self.client.request(obs)
                pending.update(reply=reply, obs_tick=obs_tick, rtt=self.client.last_rtt_ms)
            except BaseException as exc:  # handled on the control thread
                pending.update(error=exc)

        try:
            for tick in range(max_steps):
                if stop.is_set():
                    stats.stopped = "operator"
                    break
                tick_start = time.monotonic()
                state = self.driver.read_state()
                self.safety.check_state(state)
                if thread is None and len(self.queue) <= self.cfg.prefetch_threshold:
                    self._seq += 1
                    obs = self.adapter.build_obs(
                        state,
                        self._frames(stats, state.t),
                        seq=self._seq,
                        instruction=instruction,
                        episode_id=stats.episode_id,
                        jpeg_quality=self.cfg.jpeg_quality,
                    )
                    thread = threading.Thread(target=fetch, args=(obs, tick), daemon=True)
                    thread.start()
                if thread is not None and (not thread.is_alive() or (not primed and not len(self.queue))):
                    thread.join()  # the first chunk of an episode is waited for
                    thread = None
                    if "error" in pending:
                        raise pending["error"]
                    reply = pending.pop("reply")
                    if reply.get("type") == "stop":
                        stats.stopped = f"gateway stop: {reply.get('reason', '')}"
                        break
                    stats.inferences += 1
                    stats.rtt_ms.append(pending.pop("rtt"))
                    if reply.get("server_ms") is not None:
                        stats.server_ms.append(float(reply["server_ms"]))
                    chunk = self.adapter.to_commands(reply)
                    if self.cfg.execute_steps:
                        chunk = chunk[: self.cfg.execute_steps]
                    now_tick = tick if primed else pending["obs_tick"]
                    stats.dropped_rows += self.queue.install(chunk, pending.pop("obs_tick"), now_tick)
                    primed = True
                cmd = self.queue.pop()
                if cmd is None:
                    stats.underruns += 1
                    if (time.monotonic() - last_command_at) * 1e3 > self.safety.cfg.stale_ms:
                        self.driver.hold()
                        self.safety.reset()
                        stats.holds += 1
                        last_command_at = time.monotonic()
                else:
                    safe = self.safety.filter(cmd, state)
                    if self.safety.sends:
                        self.driver.command(safe)
                        stats.commands_sent += 1
                    last_command_at = time.monotonic()
                stats.ticks += 1
                elapsed = time.monotonic() - tick_start
                if elapsed > period:
                    stats.late_ticks += 1
                else:
                    time.sleep(period - elapsed)
        except SafetyError as exc:
            stats.stopped = f"safety: {exc}"
        except BaseException as exc:
            stats.stopped = f"{type(exc).__name__}: {exc}"
            self.driver.hold()
            if thread is not None:
                thread.join(timeout=self.client.request_timeout_s)
            raise
        finally:
            if thread is not None:
                thread.join(timeout=self.client.request_timeout_s)
        if stats.stopped:
            self.driver.hold()
            self.log(f"episode stopped: {stats.stopped}")
        return stats
