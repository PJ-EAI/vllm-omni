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
    in_flight: int = 1  # concurrent policy requests; >1 pipelines chunks to outpace a long RTT
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
        if cfg.in_flight < 1:
            raise ValueError(f"in_flight must be at least 1, got {cfg.in_flight}")
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
    """Rows stamped with absolute ticks. A new chunk supersedes only the future it
    covers (``tick >= obs_tick``); rows before that stay queued, so pipelined chunks
    merge instead of cannibalizing each other's unexecuted rows."""

    def __init__(self, blend_steps: int = 0) -> None:
        self.blend_steps = blend_steps
        self._rows: collections.deque[tuple[int, Command]] = collections.deque()

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()

    def install(self, chunk: list[Command], obs_tick: int, now_tick: int) -> int:
        """Merge ``chunk`` (row ``i`` is the target for ``obs_tick + i``); returns rows dropped as elapsed."""
        elapsed = max(0, min(now_tick - obs_tick, len(chunk)))
        fresh = [(obs_tick + i, cmd) for i, cmd in enumerate(chunk)][elapsed:]
        kept = [row for row in self._rows if row[0] < obs_tick]  # earlier future this chunk does not cover
        for i in range(min(self.blend_steps, len(kept), len(fresh))):
            k, old = kept[-self.blend_steps + i]
            tick, new = fresh[i]
            if old.mode == new.mode:
                fresh[i] = (tick, _blend(old, new, (i + 1) / (self.blend_steps + 1)))
        self._rows = collections.deque(kept + fresh)
        return elapsed

    def pop(self, at_tick: int) -> Command | None:
        if self._rows and self._rows[0][0] <= at_tick:
            return self._rows.popleft()[1]
        return None


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
        # The standard /act session is one websocket in strict request/response: the
        # gateway answers one observation at a time, so concurrent requests only race
        # on the socket. Pipelining needs a client that says it can serve them.
        self.in_flight = cfg.in_flight
        if self.in_flight > 1 and not getattr(client, "supports_pipelining", False):
            self.log("in_flight > 1 needs a pipelining client; clamping to 1")
            self.in_flight = 1

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
        inflight: list[dict] = []  # {"tick", "thread", "out"} in fire order
        done: list[dict] = []
        last_command_at = time.monotonic()
        primed = False
        last_fire = -(1 << 30)
        fire_spacing = 1  # ticks between fires; learned from RTT so chunk coverage stays contiguous

        def fetch(obs: dict, obs_tick: int, out: dict) -> None:
            try:
                reply = self.client.request(obs)
                out.update(reply=reply, rtt=self.client.last_rtt_ms)
            except BaseException as exc:  # handled on the control thread
                out.update(error=exc)

        def maybe_fire(tick: int, state) -> None:
            """One new request per call, spacing- and queue-gated (see the loop body)."""
            nonlocal last_fire
            if (
                tick - last_fire >= fire_spacing
                and len(inflight) < self.in_flight
                and len(self.queue) <= self.cfg.prefetch_threshold
            ):
                self._seq += 1
                obs = self.adapter.build_obs(
                    state,
                    self._frames(stats, state.t),
                    seq=self._seq,
                    instruction=instruction,
                    episode_id=stats.episode_id,
                    jpeg_quality=self.cfg.jpeg_quality,
                )
                out: dict = {}
                thread = threading.Thread(target=fetch, args=(obs, tick, out), daemon=True)
                inflight.append({"tick": tick, "thread": thread, "out": out})
                thread.start()
                last_fire = tick

        try:
            for tick in range(max_steps):
                if stop.is_set():
                    stats.stopped = "operator"
                    break
                tick_start = time.monotonic()
                state = self.driver.read_state()
                self.safety.check_state(state)
                # At most one new request per call keeps observations distinct; with
                # in_flight > 1 several are pipelined and installed in observation
                # order below, so the queue never rewinds to an older plan. The
                # spacing staggers fires so each chunk's surviving coverage
                # (horizon - RTT rows) abuts the previous instead of clumping.
                maybe_fire(tick, state)
                if not primed and not len(self.queue) and inflight:
                    inflight[0]["thread"].join()  # the first chunk of an episode is waited for
                still: list[dict] = []
                for entry in inflight:
                    if entry["thread"].is_alive():
                        still.append(entry)
                    else:
                        entry["thread"].join()
                        done.append(entry)
                inflight = still
                gateway_stop = False
                for entry in sorted(done, key=lambda e: e["tick"]):
                    out = entry["out"]
                    if "error" in out:
                        raise out["error"]
                    reply = out["reply"]
                    if reply.get("type") == "stop":
                        stats.stopped = f"gateway stop: {reply.get('reason', '')}"
                        gateway_stop = True
                        break
                    stats.inferences += 1
                    stats.rtt_ms.append(out["rtt"])
                    if reply.get("server_ms") is not None:
                        stats.server_ms.append(float(reply["server_ms"]))
                    chunk = self.adapter.to_commands(reply)
                    if self.cfg.execute_steps:
                        chunk = chunk[: self.cfg.execute_steps]
                    # A chunk spends rtt ticks in flight and its elapsed rows are dropped,
                    # so spacing fires one tick tighter than the surviving span keeps
                    # coverage contiguous with a small buffer for RTT jitter.
                    fresh_span = len(chunk) - int(out["rtt"] * self.control_hz / 1e3)
                    fire_spacing = max(1, fresh_span - 1)
                    now_tick = tick if primed else entry["tick"]
                    stats.dropped_rows += self.queue.install(chunk, entry["tick"], now_tick)
                    primed = True
                done = []
                if gateway_stop:
                    break
                maybe_fire(tick, state)  # refire on the install tick itself, not the next one
                cmd = self.queue.pop(tick)
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
            raise
        finally:
            for entry in inflight:
                entry["thread"].join(timeout=self.client.request_timeout_s)
        if stats.stopped:
            self.driver.hold()
            self.log(f"episode stopped: {stats.stopped}")
        return stats
