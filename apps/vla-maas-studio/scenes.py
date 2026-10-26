# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scene catalog and adapters for the VLA MaaS studio.

The studio treats a scene as an observation source: ``reset``, ``step``,
``render`` and ``policy_obs``. Two backends are provided:

* ``mock``   -- the dependency-free ``mock_libero`` env (CPU only).
* ``libero`` -- the real LIBERO/robosuite ``OffScreenRenderEnv``.

Adding another backend (for example MuJoCo Menagerie arms) means implementing
``SceneSession`` and listing its suites in ``build_catalog``. The runner only
uses this interface, so arm/scene selection stays independent of the transport
to the MaaS gateway.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_DIR = REPO_ROOT / "examples" / "online_serving" / "vla_maas"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from libero_eval import quat2axisangle  # noqa: E402

ACTION_DIM = 7
NUM_SETTLE_STEPS = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    "reach": 200,
}

# Every LIBERO backend exposes Franka Panda. The MuJoCo backend below adds
# more arms; both lists share the same runner interface.
ROBOTS = [
    {"id": "panda", "label": "Franka Emika Panda", "physics": ["mock", "libero"], "dof": 7},
]

# MuJoCo Menagerie arms. ``model`` is relative to the --menagerie checkout and
# ``eef_body`` is the body whose frame the reach task drives.
MUJOCO_ARMS = [
    {
        "id": "panda",
        "label": "Franka Emika Panda",
        "physics": ["mujoco"],
        "dof": 7,
        "model": "franka_emika_panda/panda.xml",
        "eef_body": "hand",
        "view": [0.55, -0.75, 0.55],
        "fovy": 45,
    },
    {
        "id": "ur5e",
        "label": "Universal Robots UR5e",
        "physics": ["mujoco"],
        "dof": 6,
        "model": "universal_robots_ur5e/ur5e.xml",
        "eef_body": "wrist_3_link",
        "view": [0.75, -1.0, 0.7],
        "fovy": 50,
    },
    {
        "id": "xarm7",
        "label": "UFactory xArm 7",
        "physics": ["mujoco"],
        "dof": 7,
        "model": "ufactory_xarm7/xarm7.xml",
        "eef_body": "link7",
        "view": [0.6, -0.8, 0.6],
        "fovy": 48,
    },
]

REACH_TASKS = [{"id": task_id, "language": f"reach the red target ({task_id})"} for task_id in range(10)]

DEFAULT_MENAGERIE = os.environ.get("MUJOCO_MENAGERIE", "~/.cache/mujoco_menagerie")
CLONE_HINT = (
    "git clone --depth 1 --filter=blob:none --sparse "
    "https://github.com/google-deepmind/mujoco_menagerie ~/.cache/mujoco_menagerie && "
    "cd ~/.cache/mujoco_menagerie && git sparse-checkout set "
    "franka_emika_panda universal_robots_ur5e ufactory_xarm7"
)


def _benchmark_module(physics: str):
    if physics == "mock":
        import mock_libero

        return mock_libero
    if physics == "libero":
        from libero.libero import benchmark

        return benchmark
    raise ValueError(f"unknown physics backend {physics!r} (expected 'mock' or 'libero')")


def build_catalog(physics: str, menagerie: str = "") -> dict:
    """List robots, suites and tasks for the configured simulator."""
    if physics == "mujoco":
        root = Path(menagerie).expanduser() if menagerie else Path(DEFAULT_MENAGERIE).expanduser()
        arms = [arm for arm in MUJOCO_ARMS if (root / arm["model"]).is_file()]
        if not arms:
            raise RuntimeError(f"no MuJoCo Menagerie arms found under {root}\nClone them with:\n  {CLONE_HINT}")
        return {
            "physics": physics,
            "max_robots": 16,
            "robots": arms,
            "suites": {"reach": {"max_steps": MAX_STEPS["reach"], "tasks": REACH_TASKS}},
        }

    benchmark = _benchmark_module(physics)
    suites: dict[str, dict] = {}
    for name, factory in benchmark.get_benchmark_dict().items():
        if name not in MAX_STEPS:
            continue
        suite = factory()
        tasks = [{"id": task_id, "language": suite.get_task(task_id).language} for task_id in range(suite.n_tasks)]
        suites[name] = {"max_steps": MAX_STEPS[name], "tasks": tasks}
    if not suites:
        raise RuntimeError(f"no usable LIBERO suites found for physics={physics!r}")
    return {"physics": physics, "robots": ROBOTS, "suites": suites}


def create_scene(
    physics: str,
    suite_name: str,
    task_id: int,
    *,
    robot: str = "panda",
    resolution: int = 256,
    action_scale: float = 0.005,
    seed: int = 0,
    menagerie: str = "",
):
    """Build the backend-specific scene while keeping the runner interface."""
    if physics == "mujoco":
        from mujoco_scenes import MujocoScene

        return MujocoScene(
            arm_id=robot,
            task_id=task_id,
            resolution=resolution,
            action_scale=action_scale,
            seed=seed,
            menagerie=menagerie,
        )
    return SceneSession(
        physics=physics,
        suite_name=suite_name,
        task_id=task_id,
        robot=robot,
        resolution=resolution,
        action_scale=action_scale,
        seed=seed,
    )


class SceneSession:
    """One task instance: an env, its init states and the policy observation."""

    def __init__(
        self,
        physics: str,
        suite_name: str,
        task_id: int,
        *,
        robot: str = "panda",
        resolution: int = 256,
        action_scale: float = 0.005,
        seed: int = 0,
    ) -> None:
        if robot != "panda":
            raise ValueError(f"robot {robot!r} is not implemented by the {physics!r} scene backend")
        self.physics = physics
        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.seed = seed
        self.resolution = resolution
        self.robot = robot

        benchmark = _benchmark_module(physics)
        if suite_name not in benchmark.get_benchmark_dict():
            raise ValueError(f"unknown task suite {suite_name!r}")
        self.suite = benchmark.get_benchmark_dict()[suite_name]()
        if not 0 <= self.task_id < self.suite.n_tasks:
            raise ValueError(f"task {self.task_id} out of range for {suite_name!r}")
        self.task = self.suite.get_task(self.task_id)
        self.language = self.task.language
        self.init_states = np.asarray(self.suite.get_task_init_states(self.task_id))
        self.max_steps = MAX_STEPS[suite_name]
        self.action_dim = ACTION_DIM
        self.state_dim = 8
        self.env = self._make_env(action_scale)

    def _make_env(self, action_scale: float):
        if self.physics == "mock":
            from mock_libero import MockOffScreenRenderEnv

            env = MockOffScreenRenderEnv(
                camera_heights=self.resolution,
                camera_widths=self.resolution,
                action_scale=action_scale,
            )
        else:
            from libero.libero import get_libero_path
            from libero.libero.envs import OffScreenRenderEnv

            bddl = os.path.join(
                get_libero_path("bddl_files"),
                self.task.problem_folder,
                self.task.bddl_file,
            )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=self.resolution,
                camera_widths=self.resolution,
            )
        env.seed(self.seed)
        return env

    def init_state(self, episode: int) -> np.ndarray:
        return self.init_states[episode % len(self.init_states)]

    def reset(self, init_state: np.ndarray) -> dict:
        self.env.reset()
        raw = self.env.set_init_state(init_state)
        for _ in range(NUM_SETTLE_STEPS):
            raw, _, _, _ = self.env.step(DUMMY_ACTION)
        return raw

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        raw, _, done, _ = self.env.step(np.asarray(action, dtype=np.float32).tolist())
        return raw, bool(done)

    def policy_obs(
        self,
        raw: dict,
        prompt: str,
        session_id: str,
        base_camera_key: str,
        wrist_camera_key: str,
    ) -> dict:
        state = np.concatenate(
            (raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])
        ).astype(np.float32)
        return {
            base_camera_key: np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
            wrist_camera_key: np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]),
            "state": state,
            "prompt": prompt,
            "session_id": session_id,
        }

    @staticmethod
    def render(raw: dict) -> np.ndarray:
        """Upright agent view, i.e. what the policy sees (LIBERO renders flipped)."""
        return np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])

    def close(self) -> None:
        self.env.close()
