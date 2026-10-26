# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hardware data <-> standard Robot protocol. Knows nothing about any model.

Observation side: camera frames under their protocol roles, resized to what
the server asked for; state in SI units, the gripper mapped to ``[0, 1]`` and
``eef_pose`` optionally moved into a calibrated base frame. Action side: the
inverse mapping, one ``Command`` per chunk row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vla_maas_protocol import RobotHello, ServerHello, encode_jpeg, resize_with_pad
from vla_maas_protocol import geometry as geo

from vla_maas_robot.driver import Command, RobotSpec, RobotState


@dataclass
class Calibration:
    gripper_raw_range: tuple[float, float] | None = None  # raw closed, raw open; None = the driver's
    gripper_invert: bool = False
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)  # driver base frame in the standard base frame
    base_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, d: dict | None) -> Calibration:
        d = dict(d or {})
        gripper = d.pop("gripper", {}) or {}
        base = d.pop("base_frame", {}) or {}
        if d:
            raise ValueError(f"unknown calibration keys: {sorted(d)}")
        raw = gripper.get("raw_range")
        return cls(
            gripper_raw_range=tuple(raw) if raw is not None else None,
            gripper_invert=bool(gripper.get("invert", False)),
            base_xyz=tuple(base.get("xyz", (0.0, 0.0, 0.0))),
            base_rpy=tuple(base.get("rpy", (0.0, 0.0, 0.0))),
        )


class RobotAdapter:
    def __init__(self, spec: RobotSpec, roles: list[str], calibration: Calibration | None = None) -> None:
        self.spec = spec
        self.roles = list(roles)
        self.cal = calibration or Calibration()
        lo, hi = self.cal.gripper_raw_range or spec.gripper_range
        self._grip_lo, self._grip_hi = float(lo), float(hi)
        self._base_q = geo.rpy_to_quat(np.asarray(self.cal.base_rpy, dtype=np.float64))
        self._base_t = np.asarray(self.cal.base_xyz, dtype=np.float64)
        self.server: ServerHello | None = None

    # -- handshake ------------------------------------------------------------

    def hello(self, robot_id: str, jpeg: bool) -> RobotHello:
        return RobotHello(
            robot_id=robot_id,
            embodiment=self.spec.embodiment,
            dof=self.spec.dof,
            images=self.roles,
            state=self.spec.state_fields(),
            action_spaces=list(self.spec.modes),
            image_encodings=["raw", "jpeg"] if jpeg else ["raw"],
        )

    def configure(self, server: ServerHello) -> None:
        self.server = server

    # -- units ----------------------------------------------------------------

    def gripper_to_std(self, raw: float) -> float:
        g = (raw - self._grip_lo) / (self._grip_hi - self._grip_lo)
        g = 1.0 - g if self.cal.gripper_invert else g
        return float(np.clip(g, 0.0, 1.0))

    def gripper_from_std(self, g: float) -> float:
        g = float(np.clip(g, 0.0, 1.0))
        g = 1.0 - g if self.cal.gripper_invert else g
        return self._grip_lo + g * (self._grip_hi - self._grip_lo)

    def pose_to_std(self, pose: np.ndarray) -> np.ndarray:
        xyz = geo.quat_rotate(self._base_q, pose[:3]) + self._base_t
        return np.concatenate((xyz, geo.quat_normalize(geo.quat_mul(self._base_q, pose[3:7]))))

    def pose_from_std(self, pose: np.ndarray) -> np.ndarray:
        inv = geo.quat_conj(self._base_q)
        xyz = geo.quat_rotate(inv, np.asarray(pose[:3]) - self._base_t)
        return np.concatenate((xyz, geo.quat_normalize(geo.quat_mul(inv, pose[3:7]))))

    # -- frames ---------------------------------------------------------------

    def build_obs(
        self,
        state: RobotState,
        frames: dict[str, tuple[float, np.ndarray]],
        *,
        seq: int,
        instruction: str,
        episode_id: str,
        jpeg_quality: int | None = None,
    ) -> dict:
        sizes = self.server.requires.images if self.server else {}
        images = {}
        for role, (_, frame) in frames.items():
            if role in sizes:
                frame = resize_with_pad(frame, *sizes[role])
            elif self.server is not None:
                continue  # the model does not use this camera; do not send it
            images[role] = encode_jpeg(frame, jpeg_quality) if jpeg_quality else np.ascontiguousarray(frame)
        std_state = {"joint_pos": np.asarray(state.joint_pos, dtype=np.float32)}
        if state.joint_vel is not None:
            std_state["joint_vel"] = np.asarray(state.joint_vel, dtype=np.float32)
        if state.eef_pose is not None:
            std_state["eef_pose"] = self.pose_to_std(np.asarray(state.eef_pose)).astype(np.float32)
        if state.gripper is not None:
            std_state["gripper"] = np.array([self.gripper_to_std(state.gripper)], dtype=np.float32)
        return {
            "type": "obs",
            "seq": seq,
            "t": state.t,
            "episode_id": episode_id,
            "instruction": instruction,
            "images": images,
            "state": std_state,
        }

    def to_commands(self, msg: dict) -> list[Command]:
        if msg.get("mode", "absolute") != "absolute":
            raise ValueError(f"expected absolute actions, got {msg.get('mode')!r}")
        space, actions = msg["space"], msg["actions"]
        arm = np.asarray(actions[space], dtype=np.float64)
        grip = actions.get("gripper")
        grip = None if grip is None else np.asarray(grip, dtype=np.float64).reshape(len(arm), -1)[:, 0]
        if space == "eef_pose":
            arm = np.stack([self.pose_from_std(row) for row in arm])
        elif arm.shape[1] != self.spec.dof:
            raise ValueError(f"joint_pos actions have width {arm.shape[1]}, robot dof is {self.spec.dof}")
        return [
            Command(
                mode=space,
                target=row,
                gripper=None if grip is None or not self.spec.has_gripper else self.gripper_from_std(grip[i]),
            )
            for i, row in enumerate(arm)
        ]
