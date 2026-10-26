# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MuJoCo twin of the AgileX Piper for the studio's Robot-protocol mode.

``PiperSimDriver`` is a ``vla_maas_robot`` ``RobotDriver`` that takes the same
``driver:`` block as the real ``PiperDriver`` (``robots/piper.yaml``) and
copies its ``RobotSpec``: joint limits, gripper stroke, modes and embodiment
are the real driver's, so the handshake, the gateway adapter and the safety
filter see exactly what they will see on hardware. Only the SDK is replaced:

* the arm starts at ``home_joints``, where ``robotctl home`` leaves the real arm;
* joint targets are tracked kinematically at ``speed_pct`` of the joint speed
  limit, like the arm's own MOVE J interpolation;
* end-pose targets (``modes: [joint_pos, eef_pose]``) go through
  damped-least-squares IK on ``link6``, like MOVE P;
* ``eef_pose`` is the ``link6`` frame in the base frame, the gripper is the jaw
  opening in metres (0 - 0.07).

``SimCamera`` renders the scene under a protocol camera role (``front`` is a
fixed view of the table, ``wrist`` rides on ``link6``) and plugs into the
control loop like any other ``vla_maas_robot`` camera.

The model comes from MuJoCo Menagerie (``agilex_piper/piper.xml``). This is a
kinematic twin for checking the protocol, the route and the safety settings
before the real arm moves; it says nothing about grasping or about how a
checkpoint behaves on real images (lessons L002).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from vla_maas_robot.cameras import Camera  # noqa: E402
from vla_maas_robot.driver import Command, RobotDriver, RobotState  # noqa: E402
from vla_maas_robot.drivers.piper import PiperDriver  # noqa: E402

PIPER_MODEL = "agilex_piper/piper.xml"
DEFAULT_MENAGERIE = os.environ.get("MUJOCO_MENAGERIE", "~/.cache/mujoco_menagerie")
CLONE_HINT = (
    "git clone --depth 1 --filter=blob:none --sparse "
    "https://github.com/google-deepmind/mujoco_menagerie ~/.cache/mujoco_menagerie && "
    "cd ~/.cache/mujoco_menagerie && git sparse-checkout set agilex_piper"
)
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ("joint7", "joint8")  # each jaw travels half of the opening
EEF_BODY = "link6"
GRIPPER_SPEED_M_S = 0.1
CAMERAS = ("front", "wrist")


def piper_model_path(menagerie: str = "") -> Path:
    return Path(menagerie or DEFAULT_MENAGERIE).expanduser() / PIPER_MODEL


def _build_model(path: Path, max_width: int, max_height: int) -> mujoco.MjModel:
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo Menagerie Piper not found: {path}\nClone it with:\n  {CLONE_HINT}")
    spec = mujoco.MjSpec.from_file(str(path))
    world = spec.worldbody
    world.add_light(pos=[0.3, 0.0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.add_geom(name="studio_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], rgba=[0.3, 0.32, 0.36, 1])
    world.add_geom(
        name="studio_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.2, 0.3, 0.002],
        pos=[0.35, 0.0, 0.002],
        rgba=[0.62, 0.56, 0.46, 1],
    )
    world.add_geom(
        name="studio_cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        pos=[0.32, 0.08, 0.024],
        rgba=[0.85, 0.15, 0.15, 1],
    )
    # Front view from in front of the table, looking back at the arm.
    eye, look = np.array([0.95, -0.35, 0.55]), np.array([0.25, 0.0, 0.15])
    forward = (look - eye) / np.linalg.norm(look - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world.add_camera(name="studio_front", pos=eye.tolist(), xyaxes=[*right, *up], fovy=50)
    # Wrist view: beside the gripper housing, looking along link6 +z (between the jaws).
    hand = spec.body(EEF_BODY)
    hand.add_camera(name="studio_wrist", pos=[-0.06, 0.0, 0.05], xyaxes=[0, -1, 0, -0.928, 0, 0.371], fovy=75)
    spec.visual.global_.offwidth = max(int(max_width), 640)
    spec.visual.global_.offheight = max(int(max_height), 480)
    return spec.compile()


class PiperSimDriver(RobotDriver):
    """``PiperDriver``'s spec and config keys over a kinematic MuJoCo Piper."""

    def __init__(self, menagerie: str = "", max_render: tuple[int, int] = (640, 480), **driver_cfg) -> None:
        # Validates the YAML keys and yields the real driver's spec; nothing touches CAN.
        reference = PiperDriver(**driver_cfg)
        self.spec = reference.spec
        self.home_joints = reference.home_joints.copy()
        self.home_timeout_s = reference.home_timeout_s
        self.joint_speed = np.asarray(self.spec.max_joint_vel, dtype=np.float64) * reference.speed_pct / 100.0
        self.model = _build_model(piper_model_path(menagerie), *max_render)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self._qadr = np.array([self.model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self._vadr = np.array([self.model.joint(name).dofadr[0] for name in ARM_JOINTS])
        self._fadr = np.array([self.model.joint(name).qposadr[0] for name in FINGER_JOINTS])
        self._hand = self.model.body(EEF_BODY).id
        self._q = self.home_joints.copy()
        self._target = self._q.copy()
        self._gripper = self.spec.gripper_range[1]
        self._gripper_target = self._gripper
        self._qd = np.zeros(6)
        self._t = time.monotonic()
        self._connected = False
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # -- lifecycle --------------------------------------------------------------

    def connect(self) -> None:
        with self.lock:
            self._connected = True
            self._t = time.monotonic()
            self._apply()

    def close(self) -> None:
        self._connected = False

    # -- kinematics -------------------------------------------------------------

    def _apply(self) -> None:
        self.data.qpos[self._qadr] = self._q
        half = self._gripper / 2.0
        self.data.qpos[self._fadr] = (half, -half)
        mujoco.mj_kinematics(self.model, self.data)

    def _advance(self) -> None:
        now = time.monotonic()
        dt = max(now - self._t, 0.0)
        step = self.joint_speed * dt
        previous = self._q
        self._q = previous + np.clip(self._target - previous, -step, step)
        self._qd = (self._q - previous) / dt if dt > 0 else np.zeros(6)
        grip_step = GRIPPER_SPEED_M_S * dt
        self._gripper += float(np.clip(self._gripper_target - self._gripper, -grip_step, grip_step))
        self._t = now
        self._apply()

    def qpos(self) -> np.ndarray:
        """Current joint positions (arm and jaws) for cameras."""
        with self.lock:
            self._advance()
            return self.data.qpos.copy()

    def _ik(self, pose: np.ndarray) -> np.ndarray:
        """Joint target for a ``link6`` pose (xyz + quat xyzw), from the current joints."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.data.qpos
        goal_quat = np.array([pose[6], pose[3], pose[4], pose[5]])  # wxyz
        goal_quat /= np.linalg.norm(goal_quat)
        lo, hi = self.spec.joint_limits[:, 0], self.spec.joint_limits[:, 1]
        error = np.zeros(6)
        for _ in range(50):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            error[:3] = pose[:3] - data.xpos[self._hand]
            mujoco.mju_subQuat(error[3:], goal_quat, data.xquat[self._hand])
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
                break
            mujoco.mj_jacBody(self.model, data, self._jacp, self._jacr, self._hand)
            jacobian = np.vstack((self._jacp[:, self._vadr], self._jacr[:, self._vadr]))
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), error)
            data.qpos[self._qadr] = np.clip(data.qpos[self._qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
        return data.qpos[self._qadr].copy()

    # -- RobotDriver ------------------------------------------------------------

    def _require(self) -> None:
        if not self._connected:
            raise RuntimeError("PiperSimDriver is not connected")

    def read_state(self) -> RobotState:
        with self.lock:
            self._require()
            self._advance()
            wxyz = self.data.xquat[self._hand]
            return RobotState(
                t=self._t,
                joint_pos=self._q.copy(),
                joint_vel=self._qd.copy(),
                eef_pose=np.concatenate((self.data.xpos[self._hand], [wxyz[1], wxyz[2], wxyz[3], wxyz[0]])),
                gripper=float(self._gripper),
            )

    def command(self, cmd: Command) -> None:
        if cmd.mode not in self.spec.modes:
            raise ValueError(f"PiperSimDriver is configured for {self.spec.modes}, got {cmd.mode}")
        with self.lock:
            self._require()
            self._advance()
            if cmd.mode == "joint_pos":
                target = np.asarray(cmd.target, dtype=np.float64)
            else:
                target = self._ik(np.asarray(cmd.target, dtype=np.float64))
            self._target = np.clip(target, self.spec.joint_limits[:, 0], self.spec.joint_limits[:, 1])
            if cmd.gripper is not None:
                lo, hi = self.spec.gripper_range
                self._gripper_target = float(np.clip(cmd.gripper, lo, hi))

    def hold(self) -> None:
        with self.lock:
            self._advance()
            self._target = self._q.copy()
            self._gripper_target = self._gripper

    def home(self) -> None:
        with self.lock:
            self._target = self.home_joints.copy()
        deadline = time.monotonic() + self.home_timeout_s
        while time.monotonic() < deadline:
            if np.max(np.abs(self.read_state().joint_pos - self.home_joints)) < 0.02:
                return
            time.sleep(0.02)
        raise RuntimeError(f"simulated Piper did not reach home within {self.home_timeout_s}s")


class SimCamera(Camera):
    """Renders ``role``'s view of a ``PiperSimDriver`` at ``fps``.

    The renderer is created on the grab thread because an OpenGL context is
    bound to the thread that made it.
    """

    def __init__(self, sim: PiperSimDriver, role: str, width: int = 640, height: int = 480, fps: float = 30.0) -> None:
        super().__init__()
        if role not in CAMERAS:
            raise ValueError(f"the Piper twin renders {CAMERAS}, not {role!r}")
        self.sim, self.role = sim, role
        self.size, self.period = (int(height), int(width)), 1.0 / float(fps)
        self._camera = sim.model.camera(f"studio_{role}").id
        self._data = mujoco.MjData(sim.model)
        self._renderer: mujoco.Renderer | None = None
        self._next = 0.0

    def grab(self) -> np.ndarray:
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = time.monotonic() + self.period
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.sim.model, self.size[0], self.size[1])
        self._data.qpos[:] = self.sim.qpos()
        mujoco.mj_kinematics(self.sim.model, self._data)
        mujoco.mj_camlight(self.sim.model, self._data)
        self._renderer.update_scene(self._data, camera=self._camera)
        return self._renderer.render().copy()

    def _run(self) -> None:
        try:
            super()._run()
        finally:
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:  # GL teardown can fail at exit; the frames are already out
                    pass
                self._renderer = None
