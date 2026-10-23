# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Simulator adapters behind one interface for the MaaS fleet client.

Every adapter exposes robosuite-style raw observations (the keys LIBERO, which
is built on robosuite, also uses), so one ``policy_obs`` builds the OpenPI
request for all of them:

  * ``libero``    - real LIBERO suites (Franka Panda); real π0.5 LIBERO
                    fine-tunes give a meaningful success rate here.
  * ``robosuite`` - robosuite tasks with any single-arm robot (Panda, UR5e,
                    Sawyer, Kinova3, IIWA, Jaco). Only Panda matches a LIBERO
                    fine-tune; other arms exercise serving, not policy quality.
  * ``mock``      - ``mock_libero`` (no simulator), for CPU tests of the client.

Rendering is headless. Set ``MUJOCO_GL=egl`` on GPU servers (default here) and
``MUJOCO_GL=osmesa`` on CPU-only hosts; ``render_gpu`` picks the EGL device.
Import this module only in the process that will own the env: MuJoCo reads
``MUJOCO_GL`` once at import time.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ACTION_DIM = 7
DUMMY_ACTION = [0.0] * 6 + [-1.0]
# Gripper qpos is 2-dim on Panda/Sawyer and 6-dim on Robotiq grippers; LIBERO
# policies expect 2, so the state is always eef pos(3) + axis-angle(3) + 2.
GRIPPER_STATE_DIM = 2

LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
ROBOSUITE_PROMPTS = {
    "Lift": "pick up the red cube",
    "Stack": "stack the red cube on top of the green cube",
    "PickPlaceCan": "pick up the can and place it in the bin",
    "NutAssemblySquare": "fit the square nut onto the square peg",
    "Door": "open the door",
    "Wipe": "wipe the table",
}
ROBOSUITE_ROBOTS = ("Panda", "UR5e", "Sawyer", "Kinova3", "IIWA", "Jaco")


@dataclass
class EnvSpec:
    """What one robot instance simulates. Parsed from a scenario group's ``env``."""

    kind: str = "mock"  # libero | robosuite | mock
    robot: str = "Panda"
    task: str = "Lift"  # robosuite env name
    suite: str = "libero_spatial"  # libero / mock suite
    task_ids: list = field(default_factory=list)  # libero / mock; empty = all
    resolution: int = 256
    max_steps: int | None = None
    control_freq: int = 20  # robosuite controller rate (LIBERO uses 20 Hz)
    # Extra camera rendered only for the video (e.g. robosuite "frontview",
    # which shows the whole arm); None records what the policy sees.
    video_camera: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> EnvSpec:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown env keys: {sorted(unknown)}")
        spec = cls(**d)
        if spec.kind not in ("libero", "robosuite", "mock"):
            raise ValueError(f"env kind must be libero, robosuite or mock, got {spec.kind!r}")
        if spec.kind == "libero" and spec.robot != "Panda":
            raise ValueError("LIBERO scenes are built for the Franka Panda; use kind: robosuite for other arms")
        if spec.kind == "robosuite" and spec.task not in ROBOSUITE_PROMPTS:
            raise ValueError(f"robosuite task must be one of {sorted(ROBOSUITE_PROMPTS)}, got {spec.task!r}")
        return spec


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite quat (x, y, z, w) -> axis-angle, as in the OpenPI LIBERO example."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def robot_state(raw: dict) -> np.ndarray:
    gripper = np.zeros(GRIPPER_STATE_DIM)
    qpos = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64)[:GRIPPER_STATE_DIM]
    gripper[: qpos.size] = qpos
    return np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), gripper)).astype(np.float32)


def upright(image: np.ndarray) -> np.ndarray:
    # robosuite/LIBERO render with the OpenGL convention; OpenPI LIBERO clients
    # (and the LIBERO fine-tunes) rotate both cameras 180 degrees.
    return np.ascontiguousarray(image[::-1, ::-1])


def policy_obs(raw: dict, prompt: str, session_id: str, base_key: str, wrist_key: str) -> dict:
    return {
        base_key: upright(raw["agentview_image"]),
        wrist_key: upright(raw["robot0_eye_in_hand_image"]),
        "state": robot_state(raw),
        "prompt": prompt,
        "session_id": session_id,
    }


def video_frame(raw: dict, camera: str | None = None) -> np.ndarray:
    """Scene camera and wrist view side by side, as a person would see them.

    Only the OpenGL vertical flip is undone; the extra horizontal flip in
    ``upright`` exists to match the policy's training data, not for viewing.
    """
    scene = raw[f"{camera or 'agentview'}_image"]
    return np.ascontiguousarray(np.concatenate((scene[::-1], raw["robot0_eye_in_hand_image"][::-1]), axis=1))


def camera_names(spec: EnvSpec) -> list:
    names = ["agentview", "robot0_eye_in_hand"]
    if spec.video_camera and spec.video_camera not in names:
        names.append(spec.video_camera)
    return names


class SimEnv:
    """One simulated robot. ``reset`` picks a task, ``step`` returns (raw, done)."""

    spec: EnvSpec
    prompt: str = ""
    max_steps: int = 300
    task_name: str = ""

    def reset(self, rng: np.random.Generator) -> dict:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _SuiteEnv(SimEnv):
    """LIBERO or mock LIBERO: a benchmark suite of language-conditioned tasks."""

    def __init__(self, spec: EnvSpec, render_gpu: int) -> None:
        self.spec = spec
        if spec.kind == "mock":
            import mock_libero as benchmark

            self._make_env = lambda task: benchmark.MockOffScreenRenderEnv(
                camera_heights=spec.resolution, camera_widths=spec.resolution
            )
        else:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv

            def _make_env(task):
                bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                return OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=spec.resolution,
                    camera_widths=spec.resolution,
                    camera_names=camera_names(spec),
                    render_gpu_device_id=render_gpu,
                )

            self._make_env = _make_env
        self._suite = benchmark.get_benchmark_dict()[spec.suite]()
        self._task_ids = list(spec.task_ids) or list(range(self._suite.n_tasks))
        self.max_steps = spec.max_steps or LIBERO_MAX_STEPS.get(spec.suite, 300)
        self._env = None
        self._env_task: int | None = None

    def reset(self, rng: np.random.Generator) -> dict:
        task_id = int(rng.choice(self._task_ids))
        task = self._suite.get_task(task_id)
        if self._env_task != task_id:  # LIBERO builds one MuJoCo scene per task
            if self._env is not None:
                self._env.close()
            self._env = self._make_env(task)
            self._env_task = task_id
        init_states = self._init_states(task_id)
        self.prompt = task.language
        self.task_name = f"{self.spec.suite}/{task_id}"
        self._env.reset()
        raw = self._env.set_init_state(init_states[int(rng.integers(len(init_states)))])
        if self.spec.kind == "libero":
            for _ in range(10):  # objects settle after placement, as in OpenPI
                raw, _, _, _ = self._env.step(DUMMY_ACTION)
        return raw

    def _init_states(self, task_id: int) -> np.ndarray:
        try:
            return self._suite.get_task_init_states(task_id)
        except Exception as exc:
            # torch >= 2.6 defaults torch.load to weights_only=True, which rejects
            # the numpy arrays in LIBERO's bundled init-state files. Retry the
            # same file (shipped in the LIBERO repo) the way LIBERO intended.
            if self.spec.kind != "libero" or "weights_only" not in str(exc):
                raise
            import torch
            from libero.libero import get_libero_path

            task = self._suite.tasks[task_id]
            path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
            return torch.load(path, weights_only=False)

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        raw, _, done, _ = self._env.step(np.asarray(action[:ACTION_DIM]).tolist())
        return raw, bool(done)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()


def _robosuite_controller(robot: str) -> dict:
    try:  # robosuite >= 1.5: composite controllers (arm OSC_POSE + gripper)
        from robosuite.controllers import load_composite_controller_config

        return load_composite_controller_config(controller="BASIC", robot=robot)
    except ImportError:  # robosuite 1.4.x, the version LIBERO pins
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller="OSC_POSE")


class RobosuiteEnv(SimEnv):
    def __init__(self, spec: EnvSpec, render_gpu: int) -> None:
        import robosuite

        self.spec = spec
        self.prompt = ROBOSUITE_PROMPTS[spec.task]
        self.task_name = f"{spec.task}/{spec.robot}"
        self.max_steps = spec.max_steps or 400
        self._env = robosuite.make(
            spec.task,
            robots=spec.robot,
            controller_configs=_robosuite_controller(spec.robot),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names(spec),
            camera_heights=spec.resolution,
            camera_widths=spec.resolution,
            control_freq=spec.control_freq,
            horizon=self.max_steps + 1,  # the client ends episodes, not robosuite
            ignore_done=True,
            render_gpu_device_id=render_gpu,
        )
        if self._env.action_dim != ACTION_DIM:
            raise ValueError(f"{spec.robot} on {spec.task} has action_dim {self._env.action_dim}, need {ACTION_DIM}")

    def reset(self, rng: np.random.Generator) -> dict:
        # robosuite samples placements from numpy's global RNG.
        np.random.seed(int(rng.integers(2**31)))
        return self._env.reset()

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        raw, _, _, _ = self._env.step(np.asarray(action[:ACTION_DIM], dtype=np.float64))
        return raw, bool(self._env._check_success())

    def close(self) -> None:
        self._env.close()


def configure_rendering(render_gpu: int) -> None:
    """Call before importing mujoco/robosuite/libero in this process."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    if os.environ["MUJOCO_GL"] == "egl":
        # ``render_gpu`` is an EGL (physical) device index. The sim process does
        # no CUDA work, and robosuite rejects an EGL id outside a job's
        # CUDA_VISIBLE_DEVICES, so drop the CUDA mask here.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu)


def make_sim_env(spec: EnvSpec, render_gpu: int = 0) -> SimEnv:
    if spec.kind != "mock":
        configure_rendering(render_gpu)
    if spec.kind == "robosuite":
        return RobosuiteEnv(spec, render_gpu)
    return _SuiteEnv(spec, render_gpu)


def describe(spec: EnvSpec) -> dict[str, Any]:
    if spec.kind == "robosuite":
        return {"env": "robosuite", "task": spec.task, "robot": spec.robot}
    return {"env": spec.kind, "task": spec.suite, "robot": spec.robot}
