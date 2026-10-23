# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A dependency-free stand-in for LIBERO with the same client-facing surface.

It reproduces what ``libero_eval.py`` touches: the benchmark/suite API, the
``OffScreenRenderEnv`` methods, and the observation keys, shapes and dtypes
(two 180-degree-flipped HxWx3 uint8 cameras, eef pos/quat, 2-dim gripper qpos,
7-dim delta action). The task is a 2-D reach: drive the end effector to a goal
drawn in the agent-view camera. The goal is visible only in pixels, so a policy
has to read the image to succeed, exactly like the real benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Workspace (metres) mapped onto the agent-view image.
WORKSPACE_LOW = np.array([-0.3, -0.3])
WORKSPACE_HIGH = np.array([0.3, 0.3])
SUCCESS_RADIUS = 0.03
ACTION_SCALE = 0.02  # metres moved per unit of delta action, like OSC_POSE
TABLE_Z = 0.9

GOAL_RGB = (220, 30, 30)
EEF_RGB = (30, 30, 220)


@dataclass(frozen=True)
class MockTask:
    language: str
    problem_folder: str = "mock"
    bddl_file: str = "mock.bddl"


_TASK_LANGUAGE = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]


class MockSuite:
    def __init__(self, name: str, n_tasks: int = 10, n_init_states: int = 50) -> None:
        self.name = name
        self.n_tasks = n_tasks
        self._n_init_states = n_init_states

    def get_task(self, task_id: int) -> MockTask:
        return MockTask(language=_TASK_LANGUAGE[task_id % len(_TASK_LANGUAGE)])

    def get_task_init_states(self, task_id: int) -> np.ndarray:
        # Each init state is [eef_x, eef_y, goal_x, goal_y], deterministic per task.
        rng = np.random.default_rng(1000 + task_id)
        eef = rng.uniform(WORKSPACE_LOW * 0.8, WORKSPACE_HIGH * 0.8, (self._n_init_states, 2))
        goal = rng.uniform(WORKSPACE_LOW * 0.8, WORKSPACE_HIGH * 0.8, (self._n_init_states, 2))
        return np.concatenate([eef, goal], axis=1)


def get_benchmark_dict() -> dict:
    names = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    return {name: (lambda name=name: MockSuite(name, n_tasks=90 if name == "libero_90" else 10)) for name in names}


def world_to_pixel(xy: np.ndarray, resolution: int) -> np.ndarray:
    """Map workspace metres to (row, col) in the upright agent-view image."""
    frac = (xy - WORKSPACE_LOW) / (WORKSPACE_HIGH - WORKSPACE_LOW)
    return np.clip(frac * (resolution - 1), 0, resolution - 1)


class MockOffScreenRenderEnv:
    def __init__(
        self,
        camera_heights: int = 256,
        camera_widths: int = 256,
        action_scale: float = ACTION_SCALE,
        **_: object,
    ) -> None:
        if camera_heights != camera_widths:
            raise ValueError("MockOffScreenRenderEnv renders square cameras only.")
        self.resolution = camera_heights
        # Smaller steps = longer episodes; the policy server must use the same value.
        self.action_scale = action_scale
        self._eef = np.zeros(2)
        self._goal = np.zeros(2)
        self._gripper = 0.04

    def seed(self, seed: int) -> None:
        pass

    def reset(self) -> dict:
        return self._obs()

    def set_init_state(self, init_state: np.ndarray) -> dict:
        self._eef = np.asarray(init_state[:2], dtype=np.float64).copy()
        self._goal = np.asarray(init_state[2:4], dtype=np.float64).copy()
        return self._obs()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"LIBERO actions are 7-dim, got shape {action.shape}.")
        self._eef = np.clip(self._eef + np.clip(action[:2], -1, 1) * self.action_scale, WORKSPACE_LOW, WORKSPACE_HIGH)
        self._gripper = 0.0 if action[6] > 0 else 0.04
        done = bool(np.linalg.norm(self._eef - self._goal) < SUCCESS_RADIUS)
        return self._obs(), float(done), done, {}

    def close(self) -> None:
        pass

    def _draw(self, image: np.ndarray, xy: np.ndarray, rgb: tuple[int, int, int], radius: int) -> None:
        row, col = world_to_pixel(xy, self.resolution).round().astype(int)
        image[max(0, row - radius) : row + radius + 1, max(0, col - radius) : col + radius + 1] = rgb

    def _obs(self) -> dict:
        res = self.resolution
        agent = np.full((res, res, 3), 180, dtype=np.uint8)
        self._draw(agent, self._goal, GOAL_RGB, radius=4)
        self._draw(agent, self._eef, EEF_RGB, radius=3)
        wrist = np.full((res, res, 3), 90, dtype=np.uint8)
        # Real LIBERO renders upside down; the client flips both cameras back.
        return {
            "agentview_image": agent[::-1, ::-1].copy(),
            "robot0_eye_in_hand_image": wrist,
            "robot0_eef_pos": np.array([self._eef[0], self._eef[1], TABLE_Z]),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "robot0_gripper_qpos": np.array([self._gripper, -self._gripper]),
        }
