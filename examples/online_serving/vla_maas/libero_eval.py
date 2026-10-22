#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Closed-loop LIBERO evaluation against a VLA served over the OpenPI websocket.

Mirrors the OpenPI LIBERO example (``openpi/examples/libero/main.py``): two
cameras flipped 180 degrees, 8-dim state (eef pos + axis-angle + gripper qpos),
7-dim action, 10 settle steps per episode. Reports success rate and inference
latency per suite.

Two execution modes:
  * sync  (default): execute ``--replan-steps`` actions, then block on the next
    inference; matches the published OpenPI LIBERO protocol.
  * async (``--async-prefetch N``): request the next chunk when N actions are
    left, so inference overlaps execution; ``--control-hz`` paces env steps in
    wall-clock time to expose stalls (reported as ``underruns``).

Pass ``--mock`` to swap in ``mock_libero`` (no simulator, CPU only).

Usage:
    python libero_eval.py --host 127.0.0.1 --port 8000 \
        --task-suite libero_spatial --episodes-per-task 10 \
        --base-camera-key observation.images.image \
        --wrist-camera-key observation.images.image2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np
from vla_client import DEFAULT_PATH, AsyncChunkExecutor, connect_policy

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_SETTLE_STEPS = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]
ACTION_DIM = 7


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite quat (x, y, z, w) -> axis-angle, as in the OpenPI example."""
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_obs(raw: dict, args: argparse.Namespace, prompt: str, session_id: str) -> dict:
    state = np.concatenate(
        (raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])
    ).astype(np.float32)
    # The server-side processor resizes with padding; only undo LIBERO's flip here.
    return {
        args.base_camera_key: np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
        args.wrist_camera_key: np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": state,
        "prompt": prompt,
        "session_id": session_id,
    }


def make_env(task, resolution: int, seed: int, mock: bool):
    if mock:
        from mock_libero import MockOffScreenRenderEnv

        env = MockOffScreenRenderEnv(camera_heights=resolution, camera_widths=resolution)
        env.seed(seed)
        return env
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def run_episode(env, init_state, prompt: str, executor: AsyncChunkExecutor, args, max_steps: int) -> bool:
    env.reset()
    raw = env.set_init_state(init_state)
    for _ in range(NUM_SETTLE_STEPS):
        raw, _, _, _ = env.step(DUMMY_ACTION)

    executor.reset()
    session_id = str(uuid.uuid4())
    period = 1.0 / args.control_hz if args.control_hz else 0.0
    for _ in range(max_steps):
        tick = time.perf_counter()
        action = executor.next_action(lambda: make_obs(raw, args, prompt, session_id))
        raw, _, done, _ = env.step(action.tolist())
        if done:
            return True
        if period:
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="LIBERO closed-loop eval over OpenPI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--task-suite", default="libero_spatial", choices=sorted(MAX_STEPS))
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=None, help="Evaluate only the first N tasks.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--base-camera-key", default="observation.images.image")
    parser.add_argument("--wrist-camera-key", default="observation.images.image2")
    parser.add_argument("--replan-steps", type=int, default=5, help="Actions executed per chunk.")
    parser.add_argument(
        "--async-prefetch",
        type=int,
        default=0,
        help="Request the next chunk when this many actions remain (0 = sync).",
    )
    parser.add_argument("--control-hz", type=float, default=0.0, help="Pace env steps in wall-clock (0 = off).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mock", action="store_true", help="Use mock_libero instead of the real LIBERO.")
    parser.add_argument(
        "--report-episodes",
        action="store_true",
        help="Send each episode's outcome to the MaaS gateway dashboard (gateway only).",
    )
    parser.add_argument("--api-key", default=None, help="Sent as `Authorization: Api-Key <key>`.")
    parser.add_argument("--output", type=Path, default=None, help="Write the summary as JSON.")
    args = parser.parse_args()

    if args.mock:
        import mock_libero as benchmark
    else:
        from libero.libero import benchmark

    policy, metadata = connect_policy(args.host, args.port, args.path, api_key=args.api_key)
    print(f"server metadata: {metadata}")
    executor = AsyncChunkExecutor(
        policy,
        action_dim=ACTION_DIM,
        prefetch_threshold=args.async_prefetch,
        execute_steps=args.replan_steps,
    )

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    num_tasks = min(suite.n_tasks, args.max_tasks or suite.n_tasks)
    per_task = {}
    for task_id in range(num_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = make_env(task, args.resolution, args.seed, args.mock)
        successes = 0
        for ep in range(args.episodes_per_task):
            ok = run_episode(env, init_states[ep], task.language, executor, args, MAX_STEPS[args.task_suite])
            successes += int(ok)
            if args.report_episodes:
                executor.wait()  # the socket carries one request at a time
                policy.infer({"endpoint": "episode_result", "success": ok})
            print(f"[task {task_id} ep {ep}] {'success' if ok else 'fail'}: {task.language}")
        env.close()
        per_task[task.language] = successes / args.episodes_per_task

    lat = np.asarray(executor.latencies_ms)
    summary = {
        "task_suite": args.task_suite + (" (mock)" if args.mock else ""),
        "mode": "async" if args.async_prefetch else "sync",
        "success_rate": float(np.mean(list(per_task.values()))),
        "per_task": per_task,
        "inference_p50_ms": float(np.percentile(lat, 50)),
        "inference_p99_ms": float(np.percentile(lat, 99)),
        "num_inferences": int(lat.size),
        "underruns": executor.underruns,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
