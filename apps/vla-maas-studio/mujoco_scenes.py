# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MuJoCo Menagerie scenes for the VLA MaaS studio.

The task is a tabletop reach: the selected arm tracks a Cartesian target with
damped-least-squares IK, and a red sphere marks the target. The studio's mock
policy is a visual servo, so the rendered policy view gets exact red/blue
pixel markers at the target and end-effector positions. That keeps the default
CPU policy meaningful while the human sees a real articulated arm.

The MuJoCo backend is a visualization and load source. A policy trained on
LIBERO is out of distribution here; use it to inspect the arm and the serving
loop, not to claim policy quality.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_DIR = REPO_ROOT / "examples" / "online_serving" / "vla_maas"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from mock_libero import EEF_RGB, GOAL_RGB  # noqa: E402
from scenes import CLONE_HINT, DEFAULT_MENAGERIE, MAX_STEPS, MUJOCO_ARMS, state_vector  # noqa: E402

ARM_BY_ID = {arm["id"]: arm for arm in MUJOCO_ARMS}


class MujocoScene:
    """A kinematic MuJoCo reach scene that matches the LIBERO obs contract."""

    def __init__(
        self,
        arm_id: str,
        task_id: int,
        *,
        resolution: int = 256,
        action_scale: float = 0.005,
        seed: int = 0,
        menagerie: str = "",
    ) -> None:
        arm = ARM_BY_ID.get(arm_id)
        if arm is None:
            raise ValueError(f"unknown MuJoCo arm {arm_id!r}; available: {sorted(ARM_BY_ID)}")
        self.arm = arm
        self.arm_id = arm_id
        self.task_id = int(task_id)
        self.language = f"reach the red target ({self.task_id})"
        self.max_steps = MAX_STEPS["reach"]
        self.action_dim = 7
        self.state_dim = 8
        self.robot = arm_id
        self.resolution = int(resolution)
        self.action_scale = float(action_scale)
        self.seed = int(seed)
        root = Path(menagerie).expanduser() if menagerie else Path(DEFAULT_MENAGERIE).expanduser()
        self.model_path = root / arm["model"]
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo Menagerie model not found: {self.model_path}\nClone the assets with:\n  {CLONE_HINT}"
            )
        self._step = 0
        self._render_step = -1
        self._agent_image: np.ndarray | None = None
        self._policy_image: np.ndarray | None = None
        self._build()

    # -- scene construction ---------------------------------------------------

    def _build(self) -> None:
        probe = mujoco.MjModel.from_xml_path(str(self.model_path))
        probe_data = mujoco.MjData(probe)
        if probe.nkey:
            mujoco.mj_resetDataKeyframe(probe, probe_data, 0)
        mujoco.mj_forward(probe, probe_data)
        probe_hand = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_BODY, self.arm["eef_body"])
        if probe_hand < 0:
            raise ValueError(f"body {self.arm['eef_body']!r} not found in {self.model_path}")
        self.center = probe_data.xpos[probe_hand].copy()

        spec = mujoco.MjSpec.from_file(str(self.model_path))
        world = spec.worldbody
        center = self.center
        world.add_geom(
            name="studio_table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.28, 0.34, 0.02],
            pos=[center[0], center[1], center[2] - 0.09],
            rgba=[0.55, 0.5, 0.42, 1],
        )
        target = world.add_body(
            name="studio_target",
            pos=[center[0], center[1], center[2]],
            mocap=True,
        )
        target.add_geom(
            name="studio_target_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025],
            rgba=[0.86, 0.12, 0.12, 1],
            contype=0,
            conaffinity=0,
        )
        eye = center + np.asarray(self.arm["view"], dtype=np.float64)
        forward = center - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        world.add_camera(
            name="studio_agentview",
            pos=eye.tolist(),
            xyaxes=[*right.tolist(), *up.tolist()],
            fovy=float(self.arm["fovy"]),
        )

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.arm["eef_body"])
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "studio_target")
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "studio_agentview")
        self.mocap_id = int(self.model.body_mocapid[target_body])
        if self.hand_id < 0 or target_body < 0 or self.cam_id < 0 or self.mocap_id < 0:
            raise RuntimeError("failed to add the studio table/target/camera to the MuJoCo model")

        joint_ids = list(range(self.arm["dof"]))
        self.qpos_adr = np.array([self.model.jnt_qposadr[joint] for joint in joint_ids])
        self.dof_adr = np.array([self.model.jnt_dofadr[joint] for joint in joint_ids])
        self.qpos_low = self.model.jnt_range[joint_ids, 0].copy()
        self.qpos_high = self.model.jnt_range[joint_ids, 1].copy()
        unlimited = self.model.jnt_limited[joint_ids] == 0
        self.qpos_low[unlimited] = -6.28
        self.qpos_high[unlimited] = 6.28

        extra = [joint for joint in range(self.model.njnt) if joint >= self.arm["dof"]]
        self.gripper_qadr = np.array([self.model.jnt_qposadr[joint] for joint in extra[:2]], dtype=int)

        camera_mat = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        self._right = np.array([camera_mat[0, 0], camera_mat[0, 1]])
        self._right /= np.linalg.norm(self._right)
        self._down = -np.array([camera_mat[1, 0], camera_mat[1, 1]])
        self._down /= np.linalg.norm(self._down)

        self._ik_target = self.data.xpos[self.hand_id].copy()
        self._jacp = np.zeros((3, self.model.nv))
        self.renderer = mujoco.Renderer(self.model, self.resolution, self.resolution)

    # -- scene interface ------------------------------------------------------

    def init_state(self, episode: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 1_000_003 + self.task_id * 1009 + episode)
        offset = rng.uniform(-0.08, 0.08, 2)
        if np.linalg.norm(offset) < 0.02:
            offset += np.array([0.05, 0.05])
        return np.array([self.center[0] + offset[0], self.center[1] + offset[1], self.center[2]])

    def reset(self, init_state: np.ndarray) -> dict:
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = np.asarray(init_state, dtype=np.float64)[:3]
        mujoco.mj_forward(self.model, self.data)
        self._ik_target = self.data.xpos[self.hand_id].copy()
        self._step += 1
        return self._raw()

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)[:7]
        if action.size < 7:
            action = np.pad(action, (0, 7 - action.size))
        delta = self.action_scale * (action[0] * self._down + action[1] * self._right)
        self._ik_target[:2] = np.clip(
            self._ik_target[:2] + delta,
            self.center[:2] - 0.12,
            self.center[:2] + 0.12,
        )
        self._ik_target[2] = np.clip(
            self._ik_target[2] + self.action_scale * action[2],
            self.center[2] - 0.05,
            self.center[2] + 0.05,
        )
        self._solve_ik()
        self._step += 1
        raw = self._raw()
        distance = np.linalg.norm(self.data.xpos[self.hand_id] - self.data.mocap_pos[self.mocap_id])
        return raw, bool(distance < 0.03)

    def policy_obs(
        self,
        raw: dict,
        prompt: str,
        session_id: str,
        base_camera_key: str,
        wrist_camera_key: str,
    ) -> dict:
        self._ensure_render()
        state = state_vector(raw)
        # The wrist view duplicates the agent view until a hand camera is
        # modelled; the mock policy ignores it, and real LIBERO checkpoints are
        # out of distribution here anyway.
        return {
            base_camera_key: self._policy_image.copy(),
            wrist_camera_key: self._policy_image.copy(),
            "state": state,
            "prompt": prompt,
            "session_id": session_id,
        }

    def render(self, raw: dict) -> np.ndarray:
        self._ensure_render()
        return self._agent_image

    def close(self) -> None:
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:  # EGL teardown can raise at process exit
                pass

    # -- helpers --------------------------------------------------------------

    def _raw(self) -> dict:
        quat_wxyz = self.data.xquat[self.hand_id]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        if self.gripper_qadr.size:
            gripper = self.data.qpos[self.gripper_qadr].copy()
        else:
            gripper = np.zeros(2)
        return {
            "robot0_eef_pos": self.data.xpos[self.hand_id].copy(),
            "robot0_eef_quat": quat_xyzw,
            "robot0_gripper_qpos": gripper,
        }

    def _solve_ik(self) -> None:
        for _ in range(8):
            mujoco.mj_forward(self.model, self.data)
            error = self._ik_target - self.data.xpos[self.hand_id]
            if np.linalg.norm(error) < 5e-4:
                break
            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.hand_id)
            jacobian = self._jacp[:, self.dof_adr]
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + (0.02**2) * np.eye(3),
                error,
            )
            self.data.qpos[self.qpos_adr] = np.clip(
                self.data.qpos[self.qpos_adr] + np.clip(dq, -0.2, 0.2),
                self.qpos_low,
                self.qpos_high,
            )
        mujoco.mj_forward(self.model, self.data)

    def _ensure_render(self) -> None:
        if self._render_step == self._step and self._agent_image is not None:
            return
        self.renderer.update_scene(self.data, camera=self.cam_id)
        image = self.renderer.render()
        self._agent_image = image.copy()
        policy = image.copy()
        self._mark(policy, self._project(self.data.mocap_pos[self.mocap_id]), GOAL_RGB, 4)
        self._mark(policy, self._project(self.data.xpos[self.hand_id]), EEF_RGB, 3)
        self._policy_image = policy
        self._render_step = self._step

    def _project(self, point: np.ndarray) -> tuple[int, int] | None:
        position = self.data.cam_xpos[self.cam_id]
        matrix = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        view = matrix.T @ (point - position)
        if view[2] >= -1e-6:
            return None
        fovy = np.deg2rad(self.model.cam_fovy[self.cam_id])
        x = (view[0] / -view[2]) / np.tan(fovy / 2)
        y = (view[1] / -view[2]) / np.tan(fovy / 2)
        col = int(round((x + 1) * 0.5 * (self.resolution - 1)))
        row = int(round((1 - (y + 1) * 0.5) * (self.resolution - 1)))
        if not (0 <= row < self.resolution and 0 <= col < self.resolution):
            return None
        return row, col

    @staticmethod
    def _mark(image: np.ndarray, xy: tuple[int, int] | None, rgb: tuple[int, int, int], radius: int) -> None:
        if xy is None:
            return
        row, col = xy
        image[max(0, row - radius) : row + radius + 1, max(0, col - radius) : col + radius + 1] = rgb
