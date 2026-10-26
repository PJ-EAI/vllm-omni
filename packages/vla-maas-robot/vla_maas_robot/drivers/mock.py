# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A CPU-only arm: joints track the last target at bounded speed.

Forward kinematics is a planar chain, enough to give ``eef_pose`` that moves
with the joints. Used by tests and ``robotctl --driver mock`` dry runs.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from vla_maas_robot.driver import Command, RobotDriver, RobotSpec, RobotState


class MockDriver(RobotDriver):
    def __init__(
        self,
        dof: int = 6,
        embodiment: str = "mock",
        max_joint_vel: float = 2.0,
        link_m: float = 0.15,
        modes: tuple[str, ...] | list[str] = ("joint_pos",),
    ) -> None:
        self.spec = RobotSpec(
            embodiment=embodiment,
            dof=dof,
            joint_limits=np.tile([-2.8, 2.8], (dof, 1)),
            max_joint_vel=np.full(dof, max_joint_vel),
            modes=tuple(modes),
            gripper_range=(0.0, 0.08),
        )
        self.link_m = link_m
        self._lock = threading.Lock()
        self._q = np.zeros(dof)
        self._target = np.zeros(dof)
        self._gripper = 0.08
        self._t = time.monotonic()
        self.commands: list[Command] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True
        self._t = time.monotonic()

    def _advance(self) -> None:
        now = time.monotonic()
        step = self.spec.max_joint_vel * (now - self._t)
        self._q = self._q + np.clip(self._target - self._q, -step, step)
        self._t = now

    def _fk(self, q: np.ndarray) -> np.ndarray:
        angles = np.cumsum(q[: min(3, len(q))])
        x, y = self.link_m * np.cos(angles).sum(), self.link_m * np.sin(angles).sum()
        yaw = angles[-1]
        return np.array([x, y, 0.2, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])

    def read_state(self) -> RobotState:
        with self._lock:
            self._advance()
            return RobotState(
                t=self._t,
                joint_pos=self._q.copy(),
                joint_vel=np.zeros_like(self._q),
                eef_pose=self._fk(self._q),
                gripper=self._gripper,
            )

    def command(self, cmd: Command) -> None:
        if cmd.mode != "joint_pos":
            raise ValueError(f"MockDriver supports joint_pos only, got {cmd.mode}")
        with self._lock:
            self._advance()
            self._target = np.asarray(cmd.target, dtype=np.float64).copy()
            if cmd.gripper is not None:
                self._gripper = float(cmd.gripper)
            self.commands.append(cmd)

    def hold(self) -> None:
        with self._lock:
            self._advance()
            self._target = self._q.copy()

    def home(self) -> None:
        with self._lock:
            self._q = np.zeros(self.spec.dof)
            self._target = np.zeros(self.spec.dof)
