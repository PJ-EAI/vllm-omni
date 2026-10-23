#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Simulate a fleet of robots hitting one VLA endpoint concurrently.

Each robot has its own websocket session and mock-LIBERO env and runs its own
real-time control loop at ``--control-hz`` with async chunk prefetching, so
requests from different robots arrive interleaved, as they would from a real
fleet. Robots run back-to-back episodes for ``--duration-s`` and the run
reports fleet throughput, inference latency percentiles, control-loop stalls
and success rate.

Robots are spread over ``--procs`` processes (threads inside each) to keep
client-side GIL contention out of the latency numbers.

Usage:
    python fleet_sim.py --port 9000 --path /v1/robot/pi05/openpi --api-key dev-key-change-me \
        --robots 16 --duration-s 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from libero_eval import make_obs
from mock_libero import MockOffScreenRenderEnv, get_benchmark_dict
from vla_client import DEFAULT_PATH, AsyncChunkExecutor, connect_policy

ACTION_DIM = 7
MAX_EPISODE_STEPS = 600


def run_robot(robot_id: int, args: argparse.Namespace, out: list) -> None:
    rng = np.random.default_rng(robot_id)
    suite = get_benchmark_dict()["libero_spatial"]()
    policy, _ = connect_policy(args.host, args.port, args.path, api_key=args.api_key)
    executor = AsyncChunkExecutor(
        policy,
        action_dim=ACTION_DIM,
        prefetch_threshold=args.async_prefetch,
        execute_steps=args.replan_steps,
    )
    env = MockOffScreenRenderEnv(camera_heights=256, camera_widths=256, action_scale=args.action_scale)
    period = 1.0 / args.control_hz
    episodes = successes = ticks = late_ticks = 0
    stop_at = time.perf_counter() + args.duration_s
    while time.perf_counter() < stop_at:
        task_id = int(rng.integers(suite.n_tasks))
        task = suite.get_task(task_id)
        init = suite.get_task_init_states(task_id)[int(rng.integers(50))]
        env.reset()
        raw = env.set_init_state(init)
        executor.reset()
        session_id = str(uuid.uuid4())
        success = False
        for _ in range(MAX_EPISODE_STEPS):
            tick = time.perf_counter()
            action = executor.next_action(lambda: make_obs(raw, args, task.language, session_id))
            raw, _, done, _ = env.step(action.tolist())
            ticks += 1
            elapsed = time.perf_counter() - tick
            if elapsed > period:
                late_ticks += 1
            if done:
                success = True
                break
            time.sleep(max(0.0, period - elapsed))
        episodes += 1
        successes += int(success)
        if args.report_episodes:
            executor.wait()  # the socket carries one request at a time
            policy.infer({"endpoint": "episode_result", "success": success})
    executor.wait()
    out.append(
        {
            "robot": robot_id,
            "latencies_ms": executor.latencies_ms,
            "underruns": executor.underruns,
            "episodes": episodes,
            "successes": successes,
            "ticks": ticks,
            "late_ticks": late_ticks,
        }
    )


def run_proc(robot_ids: list[int], args: argparse.Namespace, queue: mp.Queue) -> None:
    out: list[dict] = []
    threads = [threading.Thread(target=run_robot, args=(rid, args, out)) for rid in robot_ids]
    for t in threads:
        t.start()
        time.sleep(args.stagger_s)  # robots do not all boot on the same tick
    for t in threads:
        t.join()
    queue.put(out)


def summarize(results: list[dict], args: argparse.Namespace, wall_s: float) -> dict:
    lat = np.concatenate([np.asarray(r["latencies_ms"]) for r in results if r["latencies_ms"]])
    ticks = sum(r["ticks"] for r in results)
    episodes = sum(r["episodes"] for r in results)
    return {
        "label": args.label,
        "robots": args.robots,
        "duration_s": round(wall_s, 1),
        "inferences": int(lat.size),
        "throughput_rps": round(lat.size / wall_s, 1),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "latency_p90_ms": round(float(np.percentile(lat, 90)), 1),
        "latency_p99_ms": round(float(np.percentile(lat, 99)), 1),
        "late_tick_pct": round(100 * sum(r["late_ticks"] for r in results) / max(ticks, 1), 2),
        "underruns": sum(r["underruns"] for r in results),
        "episodes": episodes,
        "success_rate": round(sum(r["successes"] for r in results) / max(episodes, 1), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent robot fleet against an OpenPI endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--procs", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--replan-steps", type=int, default=25, help="Actions executed per chunk.")
    parser.add_argument("--async-prefetch", type=int, default=10, help="Prefetch when this many actions remain.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.005,
        help="Mock env step size; smaller = longer episodes. Start the mock server with the same value.",
    )
    parser.add_argument("--stagger-s", type=float, default=0.05)
    parser.add_argument("--base-camera-key", default="observation.images.image")
    parser.add_argument("--wrist-camera-key", default="observation.images.image2")
    parser.add_argument(
        "--report-episodes", action="store_true", help="Send outcomes to the MaaS gateway dashboard (gateway only)."
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path, default=None, help="Append the summary as a JSON line.")
    args = parser.parse_args()

    procs = max(1, min(args.procs, args.robots))
    shards = [list(range(i, args.robots, procs)) for i in range(procs)]
    queue: mp.Queue = mp.Queue()
    workers = [mp.Process(target=run_proc, args=(shard, args, queue)) for shard in shards]
    start = time.perf_counter()
    for w in workers:
        w.start()
    results = [r for _ in workers for r in queue.get()]
    for w in workers:
        w.join()
    summary = summarize(results, args, time.perf_counter() - start)
    print(json.dumps(summary, ensure_ascii=False))
    if args.output:
        with args.output.open("a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
