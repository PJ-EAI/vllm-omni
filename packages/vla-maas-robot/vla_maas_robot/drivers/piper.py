# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AgileX Piper (6-DoF arm + parallel gripper) over CAN, through ``piper_sdk``.

Install with ``pip install "vla-maas-robot[piper]"`` and bring the CAN port up
first (``bash can_activate.sh can0 1000000`` from the SDK). The SDK speaks
0.001 degrees, 0.001 mm and 0.001 rad/s; this driver converts to and from SI.

Joint targets go out as ``JointCtrl`` in MOVE J; the arm's own controller
interpolates to each target at ``speed_pct`` of its maximum speed, so the
safety filter's per-tick step limit stays the real bound on motion. End-pose
targets (``modes: [joint_pos, eef_pose]``) go out as ``EndPoseCtrl`` in MOVE P.

``close`` never disables the motors by default: a disabled Piper drops under
gravity. Set ``disable_on_close: true`` only with the arm resting on a support.
"""

from __future__ import annotations

import importlib
import threading
import time

import numpy as np
from vla_maas_protocol import geometry as geo

from vla_maas_robot.driver import Command, RobotDriver, RobotSpec, RobotState

# Joint limits from the piper_sdk JointCtrl docstring, in rad.
JOINT_LIMITS = np.array(
    [
        [-2.6179, 2.6179],
        [0.0, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-2.09439, 2.09439],
    ]
)
GRIPPER_RANGE_M = (0.0, 0.07)  # piper_sdk default gripper stroke, closed -> open
MDEG_PER_RAD = 180e3 / np.pi
CTRL_CAN, MOVE_P, MOVE_J, POS_VEL = 0x01, 0x00, 0x01, 0x00
GRIPPER_ENABLE = 0x01  # sent with the first gripper target, so connect never moves the jaws


class PiperDriver(RobotDriver):
    def __init__(
        self,
        can_name: str = "can0",
        embodiment: str = "piper",
        speed_pct: int = 30,
        gripper_effort_nm: float = 1.0,
        max_joint_vel: float = 3.0,
        modes: tuple[str, ...] | list[str] = ("joint_pos",),
        home_joints: list[float] | None = None,
        home_timeout_s: float = 10.0,
        enable_timeout_s: float = 5.0,
        feedback_timeout_s: float = 0.5,
        disable_on_close: bool = False,
        dof: int = 6,
    ) -> None:
        if dof != 6:
            raise ValueError(f"Piper has 6 joints, got dof={dof}")
        unknown = set(modes) - {"joint_pos", "eef_pose"}
        if unknown:
            raise ValueError(f"PiperDriver modes are joint_pos and eef_pose, got {sorted(unknown)}")
        if not 1 <= speed_pct <= 100:
            raise ValueError(f"speed_pct must be 1-100, got {speed_pct}")
        self.spec = RobotSpec(
            embodiment=embodiment,
            dof=6,
            joint_limits=JOINT_LIMITS.copy(),
            max_joint_vel=np.full(6, max_joint_vel),
            modes=tuple(modes),
            gripper_range=GRIPPER_RANGE_M,
            native_hz=200.0,
        )
        self.can_name = can_name
        self.speed_pct = int(speed_pct)
        self.gripper_effort = int(round(gripper_effort_nm * 1e3))
        self.home_joints = np.asarray(home_joints if home_joints is not None else np.zeros(6), dtype=np.float64)
        self.home_timeout_s = home_timeout_s
        self.enable_timeout_s = enable_timeout_s
        self.feedback_timeout_s = feedback_timeout_s
        self.disable_on_close = disable_on_close
        self._arm = None
        self._move_mode: int | None = None
        self._lock = threading.Lock()
        self._last_stamp: float | None = None
        self._last_change = 0.0

    # -- lifecycle --------------------------------------------------------------

    def connect(self) -> None:
        try:
            sdk = importlib.import_module("piper_sdk")
        except ImportError as exc:
            raise ImportError('PiperDriver needs piper_sdk: pip install "vla-maas-robot[piper]"') from exc
        self._arm = sdk.C_PiperInterface_V2(self.can_name)
        self._arm.ConnectPort()
        deadline = time.monotonic() + self.enable_timeout_s
        while not self._arm.EnablePiper():
            if time.monotonic() > deadline:
                raise RuntimeError(f"Piper on {self.can_name} did not enable within {self.enable_timeout_s}s")
            time.sleep(0.01)
        self._set_move_mode(MOVE_J)
        self._last_stamp, self._last_change = None, time.monotonic()

    def close(self) -> None:
        if self._arm is None:
            return
        try:
            if self.disable_on_close:
                self._arm.DisablePiper()
            self._arm.DisconnectPort()
        finally:
            self._arm = None

    # -- state ------------------------------------------------------------------

    def _require(self):
        if self._arm is None:
            raise RuntimeError("PiperDriver is not connected")
        return self._arm

    def _check_fresh(self, stamp: float, now: float) -> None:
        # The SDK keeps the last frame forever; a stamp that stops moving means the CAN link is gone.
        if stamp != self._last_stamp:
            self._last_stamp, self._last_change = stamp, now
        elif now - self._last_change > self.feedback_timeout_s:
            raise RuntimeError(f"no Piper joint feedback on {self.can_name} for {now - self._last_change:.2f}s")

    def read_state(self) -> RobotState:
        arm = self._require()
        now = time.monotonic()
        joints = arm.GetArmJointMsgs()
        self._check_fresh(float(joints.time_stamp), now)
        js = joints.joint_state
        q = np.array([js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]) / MDEG_PER_RAD
        hs = arm.GetArmHighSpdInfoMsgs()
        motors = [hs.motor_1, hs.motor_2, hs.motor_3, hs.motor_4, hs.motor_5, hs.motor_6]
        qd = np.array([m.motor_speed for m in motors], dtype=np.float64) * 1e-3
        ep = arm.GetArmEndPoseMsgs().end_pose
        xyz = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=np.float64) * 1e-6
        rpy = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=np.float64) / MDEG_PER_RAD
        gripper = arm.GetArmGripperMsgs().gripper_state.grippers_angle * 1e-6
        return RobotState(
            t=now,
            joint_pos=q,
            joint_vel=qd,
            eef_pose=np.concatenate((xyz, geo.rpy_to_quat(rpy))),
            gripper=float(gripper),
        )

    # -- commands ---------------------------------------------------------------

    def _set_move_mode(self, move_mode: int) -> None:
        if move_mode != self._move_mode:
            self._require().MotionCtrl_2(CTRL_CAN, move_mode, self.speed_pct, POS_VEL)
            self._move_mode = move_mode

    def _send_joints(self, q: np.ndarray) -> None:
        self._set_move_mode(MOVE_J)
        self._require().JointCtrl(*(int(round(v)) for v in np.asarray(q, dtype=np.float64) * MDEG_PER_RAD))

    def command(self, cmd: Command) -> None:
        if cmd.mode not in self.spec.modes:
            raise ValueError(f"PiperDriver is configured for {self.spec.modes}, got {cmd.mode}")
        with self._lock:
            arm = self._require()
            if cmd.mode == "joint_pos":
                self._send_joints(cmd.target)
            else:
                pose = np.asarray(cmd.target, dtype=np.float64)
                xyz = pose[:3] * 1e6
                rpy = geo.quat_to_rpy(pose[3:7]) * MDEG_PER_RAD
                self._set_move_mode(MOVE_P)
                arm.EndPoseCtrl(*(int(round(v)) for v in np.concatenate((xyz, rpy))))
            if cmd.gripper is not None:
                lo, hi = GRIPPER_RANGE_M
                opening = float(np.clip(cmd.gripper, lo, hi))
                arm.GripperCtrl(int(round(opening * 1e6)), self.gripper_effort, GRIPPER_ENABLE, 0)

    def hold(self) -> None:
        with self._lock:
            self._send_joints(self.read_state().joint_pos)

    def home(self) -> None:
        with self._lock:
            self._send_joints(self.home_joints)
        deadline = time.monotonic() + self.home_timeout_s
        while time.monotonic() < deadline:
            if np.max(np.abs(self.read_state().joint_pos - self.home_joints)) < 0.02:
                return
            time.sleep(0.05)
        raise RuntimeError(f"Piper did not reach home within {self.home_timeout_s}s")
