# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Every command passes here before it reaches a driver.

Modes: ``dry_run`` (filter and log, never send), ``slow`` (step limits scaled
by ``slow_scale``), ``live``. Limits are in the driver's own base frame.
A run of ``max_violations`` consecutive clamped commands raises
``SafetyError``; the control loop then holds the arm and stops.
This is a software layer. It does not replace the hardware e-stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vla_maas_robot.driver import Command, RobotSpec, RobotState

MODES = ("dry_run", "slow", "live")


class SafetyError(RuntimeError):
    pass


@dataclass
class SafetyConfig:
    start_mode: str = "dry_run"
    slow_scale: float = 0.2
    joint_margin_rad: float = 0.05  # kept clear of the driver's joint limits
    max_joint_step_rad: float | None = None  # per tick; None = max_joint_vel / control_hz
    max_eef_step_m: float = 0.02  # per tick
    workspace: dict[str, list[float]] | None = None  # {"x": [lo, hi], "y": ..., "z": ...}, metres
    max_violations: int = 5
    stale_ms: float = 300.0  # no fresh action for this long -> hold

    @classmethod
    def from_dict(cls, d: dict | None) -> SafetyConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown safety keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.start_mode not in MODES:
            raise ValueError(f"start_mode must be one of {MODES}, got {cfg.start_mode!r}")
        return cfg


@dataclass
class SafetyStats:
    commands: int = 0
    clamped: int = 0
    worst_streak: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


class SafetyFilter:
    def __init__(self, spec: RobotSpec, cfg: SafetyConfig, control_hz: float) -> None:
        self.spec, self.cfg, self.mode = spec, cfg, cfg.start_mode
        self.control_hz = control_hz
        limits = np.asarray(spec.joint_limits, dtype=np.float64)
        self.joint_lo = limits[:, 0] + cfg.joint_margin_rad
        self.joint_hi = limits[:, 1] - cfg.joint_margin_rad
        self.stats = SafetyStats()
        self._streak = 0
        self._last: np.ndarray | None = None  # last sent target; step limits apply to it
        if cfg.workspace:
            ws = cfg.workspace
            self.ws_lo = np.array([ws["x"][0], ws["y"][0], ws["z"][0]], dtype=np.float64)
            self.ws_hi = np.array([ws["x"][1], ws["y"][1], ws["z"][1]], dtype=np.float64)
        else:
            self.ws_lo = self.ws_hi = None

    @property
    def sends(self) -> bool:
        return self.mode != "dry_run"

    def _scale(self) -> float:
        return self.cfg.slow_scale if self.mode == "slow" else 1.0

    def joint_step(self) -> np.ndarray:
        if self.cfg.max_joint_step_rad is not None:
            step = np.full(self.spec.dof, self.cfg.max_joint_step_rad)
        else:
            step = np.asarray(self.spec.max_joint_vel, dtype=np.float64) / self.control_hz
        return step * self._scale()

    def _clamp(self, value: np.ndarray, lo: np.ndarray, hi: np.ndarray, reason: str, hits: list) -> np.ndarray:
        out = np.clip(value, lo, hi)
        if not np.allclose(out, value, atol=1e-9):
            hits.append(reason)
        return out

    def reset(self) -> None:
        """Forget the last target, e.g. after a hold; the next step is measured from the robot."""
        self._last = None
        self._streak = 0

    def check_state(self, state: RobotState) -> None:
        """Stop outright if the measured end effector already left the workspace."""
        if self.ws_lo is not None and state.eef_pose is not None:
            xyz = np.asarray(state.eef_pose[:3])
            if np.any(xyz < self.ws_lo - 1e-3) or np.any(xyz > self.ws_hi + 1e-3):
                raise SafetyError(f"end effector at {np.round(xyz, 3).tolist()} is outside the workspace")

    def filter(self, cmd: Command, state: RobotState) -> Command:
        hits: list[str] = []
        target = np.asarray(cmd.target, dtype=np.float64)
        if cmd.mode == "joint_pos":
            target = self._clamp(target, self.joint_lo, self.joint_hi, "joint_limit", hits)
            q = np.asarray(state.joint_pos, dtype=np.float64) if self._last is None else self._last
            step = self.joint_step()
            target = self._clamp(target, q - step, q + step, "joint_step", hits)
        elif cmd.mode == "eef_pose":
            xyz = target[:3]
            if self.ws_lo is not None:
                xyz = self._clamp(xyz, self.ws_lo, self.ws_hi, "workspace", hits)
            cur = None if state.eef_pose is None else np.asarray(state.eef_pose[:3], dtype=np.float64)
            cur = self._last[:3] if self._last is not None and len(self._last) == len(target) else cur
            if cur is not None:
                step = self.cfg.max_eef_step_m * self._scale()
                xyz = self._clamp(xyz, cur - step, cur + step, "eef_step", hits)
            target = np.concatenate((xyz, target[3:]))
        else:
            raise SafetyError(f"unsupported command mode {cmd.mode!r}")
        if cmd.mode not in self.spec.modes:
            raise SafetyError(f"{self.spec.embodiment} does not accept {cmd.mode} commands")
        gripper = cmd.gripper
        if gripper is not None:
            lo, hi = sorted(self.spec.gripper_range)
            gripper = float(np.clip(gripper, lo, hi))
        self.stats.commands += 1
        for reason in hits:
            self.stats.reasons[reason] = self.stats.reasons.get(reason, 0) + 1
        if hits:
            self.stats.clamped += 1
            self._streak += 1
            self.stats.worst_streak = max(self.stats.worst_streak, self._streak)
            if self._streak >= self.cfg.max_violations:
                raise SafetyError(f"{self._streak} consecutive clamped commands ({', '.join(sorted(set(hits)))})")
        else:
            self._streak = 0
        self._last = target
        return Command(mode=cmd.mode, target=target, gripper=gripper, meta={**cmd.meta, "clamped": hits})
