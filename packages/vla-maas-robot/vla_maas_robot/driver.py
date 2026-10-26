# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hardware abstraction: one ``RobotDriver`` per arm model.

A driver speaks SI units and absolute targets only. Its vendor SDK (and any
high-rate interpolation, e.g. Franka's 1 kHz loop) lives behind ``command``;
the control loop above it runs at the model's ``control_hz``. Vendor SDKs are
imported lazily so the package installs without them.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Mode = Literal["joint_pos", "eef_pose"]


@dataclass
class RobotSpec:
    """Static facts about an arm, used by the safety filter and the handshake."""

    embodiment: str
    dof: int
    joint_limits: np.ndarray  # [dof, 2] rad
    max_joint_vel: np.ndarray  # [dof] rad/s
    modes: tuple[str, ...] = ("joint_pos",)
    has_eef_pose: bool = True
    has_gripper: bool = True
    gripper_range: tuple[float, float] = (0.0, 1.0)  # raw driver units, closed -> open
    native_hz: float = 200.0

    def state_fields(self) -> list[str]:
        fields = ["joint_pos", "joint_vel"]
        if self.has_eef_pose:
            fields.append("eef_pose")
        if self.has_gripper:
            fields.append("gripper")
        return fields


@dataclass
class RobotState:
    t: float  # monotonic seconds
    joint_pos: np.ndarray  # [dof] rad
    joint_vel: np.ndarray | None = None  # [dof] rad/s
    eef_pose: np.ndarray | None = None  # [7] xyz (m) + quat xyzw, base frame
    gripper: float | None = None  # raw driver units


@dataclass
class Command:
    mode: Mode
    target: np.ndarray  # absolute: [dof] rad, or [7] eef pose
    gripper: float | None = None  # raw driver units
    meta: dict[str, Any] = field(default_factory=dict)


class RobotDriver:
    """Base class; subclasses wrap one vendor SDK."""

    spec: RobotSpec

    def connect(self) -> None:
        raise NotImplementedError

    def read_state(self) -> RobotState:
        raise NotImplementedError

    def command(self, cmd: Command) -> None:
        """Set a new absolute target; must not block for longer than one control tick."""
        raise NotImplementedError

    def hold(self) -> None:
        """Stop moving and hold the current pose."""
        raise NotImplementedError

    def home(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


# kind -> "module:Class"; hardware drivers land here as they are added.
DRIVERS: dict[str, str] = {
    "mock": "vla_maas_robot.drivers.mock:MockDriver",
}


def make_driver(cfg: dict) -> RobotDriver:
    cfg = dict(cfg)
    kind = cfg.pop("kind", None)
    if kind not in DRIVERS:
        raise ValueError(f"unknown driver kind {kind!r}; available: {sorted(DRIVERS)}")
    module, _, name = DRIVERS[kind].partition(":")
    return getattr(importlib.import_module(module), name)(**cfg)
