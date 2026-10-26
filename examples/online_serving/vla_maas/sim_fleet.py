#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drive many simulated robots, of several arm types, against the VLA MaaS.

A scenario YAML lists robot groups. Each group names a simulator (LIBERO,
robosuite or the mock), an arm, a model route on the gateway, and how many
robots run it. Every robot is its own process with its own simulator, OpenPI
session and real-time control loop, and robots boot at random offsets and
pause between episodes, so requests reach the service unsynchronised and
mixed across models, the way independent users would.

Per-episode MP4s (agent + wrist camera, captioned with arm, model, step and
inference latency) are recorded headless through the simulator's offscreen
renderer: ``MUJOCO_GL=egl`` on A100/A800 nodes, ``--render-gpus`` spreads the
robots' EGL contexts over GPUs.

Outputs:
  * stdout and ``--output``: one JSON line per group plus an ``all`` line
    (throughput, latency percentiles, late ticks, success rate, sim step time).
  * ``--episodes-out``: one JSON line per finished episode.
  * ``--video-dir``: ``<group>/r<robot>_ep<n>_<success|fail>.mp4``.

Usage:
    python sim_fleet.py --scenario scenarios/sim_fleet_mock.yaml --port 9000 \
        --api-key dev-key-change-me --duration-s 30 --video-dir videos
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_lib
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sim_envs import ACTION_DIM, EnvSpec, describe

DEFAULTS: dict[str, Any] = {
    "control_hz": 20.0,
    "replan_steps": 5,
    "async_prefetch": 0,
    "start_jitter_s": 2.0,
    "think_time_s": [0.0, 0.0],
    "record_episodes": 0,
    "video_stride": 1,
    "report_episodes": False,
    "base_camera_key": "observation.images.image",
    "wrist_camera_key": "observation.images.image2",
    "seed": 0,
}


@dataclass
class Group:
    name: str
    path: str
    model: str
    robots: int
    env: EnvSpec
    opts: dict = field(default_factory=dict)


def load_scenario(path: Path, defaults_override: dict | None = None) -> list[Group]:
    data = yaml.safe_load(Path(path).read_text())
    defaults = {**DEFAULTS, **(data.get("defaults") or {}), **(defaults_override or {})}
    unknown = set(defaults) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown scenario defaults: {sorted(unknown)}")
    groups = []
    for g in data.get("groups") or []:
        g = dict(g)
        name = g.pop("name")
        model = g.pop("model", None)
        path_ = g.pop("path", None) or (f"/v1/robot/{model}/openpi" if model else None)
        if path_ is None:
            raise ValueError(f"group {name!r} needs `model` (gateway route) or `path`")
        robots = int(g.pop("robots", 1))
        env = EnvSpec.from_dict(g.pop("env", {}))
        unknown = set(g) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"group {name!r} has unknown keys: {sorted(unknown)}")
        groups.append(Group(name, path_, model or path_, robots, env, {**defaults, **g}))
    if not groups:
        raise ValueError(f"{path} defines no groups")
    names = [g.name for g in groups]
    if len(set(names)) != len(names):
        raise ValueError(f"group names must be unique: {names}")
    return groups


def run_robot(job: dict, results: mp.Queue, go: Any, stop_at: Any) -> None:
    """One robot: own process, simulator, websocket session and control loop.

    Builds its simulator, reports ``ready``, then waits on ``go`` so that slow
    scene construction does not eat into the measured window.
    """
    group: Group = job["group"]
    opts, rid = group.opts, job["robot_id"]
    # Imports that touch MuJoCo happen here, after the render device is chosen.
    from sim_envs import make_sim_env, policy_obs, video_frame
    from vla_client import AsyncChunkExecutor, close_policy, connect_policy

    rng = np.random.default_rng([opts["seed"], job["global_id"]])
    record = {"type": "robot", "group": group.name, "robot_id": rid, "latencies_ms": [], "sim_step_ms": []}
    env = recorder = policy = None
    try:
        env = make_sim_env(group.env, render_gpu=job["render_gpu"])
        if opts["record_episodes"] and job["video_dir"]:
            from video_recorder import EpisodeRecorder

            recorder = EpisodeRecorder(
                Path(job["video_dir"]) / group.name, fps=opts["control_hz"], stride=opts["video_stride"]
            )
        results.put({"type": "ready"})
        go.wait()
        time.sleep(float(rng.uniform(0, opts["start_jitter_s"])))
        policy, _ = connect_policy(job["host"], job["port"], group.path, api_key=job["api_key"])
        executor = AsyncChunkExecutor(
            policy,
            action_dim=ACTION_DIM,
            prefetch_threshold=opts["async_prefetch"],
            execute_steps=opts["replan_steps"],
        )
        period = 1.0 / opts["control_hz"] if opts["control_hz"] else 0.0
        ticks = late_ticks = episode = 0
        while time.time() < stop_at.value and (not job["episodes"] or episode < job["episodes"]):
            raw = env.reset(rng)
            executor.reset()
            session_id = str(uuid.uuid4())
            n_inferences = len(executor.latencies_ms)
            name = f"r{rid}_ep{episode}"
            recording = recorder is not None and episode < opts["record_episodes"]
            if recording:
                header = f"{group.name} r{rid} {group.env.robot} -> {group.model}"
                recorder.start(header, name)
                recorder.add(video_frame(raw, group.env.video_camera), 0, None)
            started, success, steps = time.perf_counter(), False, 0
            for steps in range(1, env.max_steps + 1):
                tick = time.perf_counter()
                action = executor.next_action(
                    lambda raw=raw: policy_obs(
                        raw, env.prompt, session_id, opts["base_camera_key"], opts["wrist_camera_key"]
                    )
                )
                step_start = time.perf_counter()
                raw, success = env.step(action)
                record["sim_step_ms"].append((time.perf_counter() - step_start) * 1e3)
                if recording:
                    latency = executor.latencies_ms[-1] if executor.latencies_ms else None
                    recorder.add(video_frame(raw, group.env.video_camera), steps, latency)
                ticks += 1
                elapsed = time.perf_counter() - tick
                if period and elapsed > period:
                    late_ticks += 1
                if success:
                    break
                if period:
                    time.sleep(max(0.0, period - elapsed))
            video = recorder.finish(f"{name}_{'success' if success else 'fail'}", success) if recording else None
            if opts["report_episodes"]:
                executor.wait()  # the socket carries one request at a time
                policy.infer({"endpoint": "episode_result", "success": success})
            results.put(
                {
                    "type": "episode",
                    "group": group.name,
                    "model": group.model,
                    **describe(group.env),
                    "robot_id": rid,
                    "episode": episode,
                    "prompt": env.prompt,
                    "task_name": env.task_name,
                    "success": success,
                    "steps": steps,
                    "wall_s": round(time.perf_counter() - started, 2),
                    "inferences": len(executor.latencies_ms) - n_inferences,
                    "video": str(video) if video else None,
                }
            )
            episode += 1
            lo, hi = opts["think_time_s"]
            time.sleep(float(rng.uniform(lo, hi)))  # the "user" pauses before the next task
        executor.wait()
        record.update(
            latencies_ms=executor.latencies_ms,
            underruns=executor.underruns,
            ticks=ticks,
            late_ticks=late_ticks,
        )
    except BaseException as exc:  # report and let the rest of the fleet finish
        if recorder is not None:
            recorder.abort()
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        if policy is not None:
            close_policy(policy)
        if env is not None:
            env.close()
        results.put(record)


def _pct(values: list, q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if len(values) else None


def summarize(label: str, group_name: str, robots: list[dict], episodes: list[dict], wall_s: float) -> dict:
    lat = [x for r in robots for x in r.get("latencies_ms", [])]
    steps = [x for r in robots for x in r.get("sim_step_ms", [])]
    ticks = sum(r.get("ticks", 0) for r in robots)
    successes = sum(e["success"] for e in episodes)
    return {
        "label": label,
        "group": group_name,
        "robots": len(robots),
        "duration_s": round(wall_s, 1),
        "inferences": len(lat),
        "throughput_rps": round(len(lat) / wall_s, 2) if wall_s else 0.0,
        "latency_p50_ms": _pct(lat, 50),
        "latency_p90_ms": _pct(lat, 90),
        "latency_p99_ms": _pct(lat, 99),
        "late_tick_pct": round(100 * sum(r.get("late_ticks", 0) for r in robots) / max(ticks, 1), 2),
        "underruns": sum(r.get("underruns", 0) for r in robots),
        "episodes": len(episodes),
        "success_rate": round(successes / len(episodes), 3) if episodes else None,
        "sim_step_p50_ms": _pct(steps, 50),
        "videos": sum(1 for e in episodes if e.get("video")),
        "errors": sum(1 for r in robots if "error" in r),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulated multi-arm robot fleet against the VLA MaaS")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--duration-s", type=float, default=60.0, help="No new episodes start after this.")
    parser.add_argument("--episodes", type=int, default=0, help="Stop each robot after N episodes (0 = no cap).")
    parser.add_argument("--render-gpus", default="0", help="EGL device ids, robots assigned round-robin.")
    parser.add_argument("--video-dir", type=Path, default=None, help="Record MP4s here (record_episodes > 0).")
    parser.add_argument(
        "--record-episodes", type=int, default=None, help="Override the scenario: record the first N per robot."
    )
    parser.add_argument(
        "--video-grid", action="store_true", help="Also tile the clips into <video-dir>/grid.mp4 (one per group first)."
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path, default=None, help="Append the group summaries as JSON lines.")
    parser.add_argument("--episodes-out", type=Path, default=None, help="Write one JSON line per episode.")
    args = parser.parse_args()

    override = {} if args.record_episodes is None else {"record_episodes": args.record_episodes}
    groups = load_scenario(args.scenario, override)
    gpus = [int(x) for x in str(args.render_gpus).split(",") if x != ""] or [0]
    jobs = []
    for group in groups:
        for rid in range(group.robots):
            jobs.append(
                {
                    "group": group,
                    "robot_id": rid,
                    "global_id": len(jobs),
                    "render_gpu": gpus[len(jobs) % len(gpus)],
                    "host": args.host,
                    "port": args.port,
                    "api_key": args.api_key,
                    "episodes": args.episodes,
                    "video_dir": str(args.video_dir) if args.video_dir else None,
                }
            )
    print(
        f"{len(jobs)} robots in {len(groups)} groups: "
        + ", ".join(f"{g.name}={g.robots}x{g.env.robot}@{g.env.kind}->{g.model}" for g in groups),
        flush=True,
    )

    # spawn: each robot gets a fresh interpreter, so MuJoCo/EGL state is never forked.
    ctx = mp.get_context("spawn")
    results: mp.Queue = ctx.Queue()
    go, stop_at = ctx.Event(), ctx.Value("d", 0.0)
    procs = [ctx.Process(target=run_robot, args=(job, results, go, stop_at), daemon=True) for job in jobs]
    for p in procs:
        p.start()
    ready, start = 0, time.perf_counter()
    robots: list[dict] = []
    episodes: list[dict] = []
    episodes_f = args.episodes_out.open("w") if args.episodes_out else None
    while len(robots) < len(jobs):
        try:
            item = results.get(timeout=5.0)
        except queue_lib.Empty:
            if not any(p.is_alive() for p in procs) and results.empty():
                print(f"warning: {len(jobs) - len(robots)} robot processes exited without a report", flush=True)
                break
            continue
        if item["type"] == "ready" or (item["type"] == "robot" and not go.is_set()):
            ready += 1
            if item["type"] == "robot":
                robots.append(item)
                print(f"[{item['group']} r{item['robot_id']}] failed to start:\n{item.get('traceback')}", flush=True)
            if ready == len(jobs):
                print(f"all {len(jobs)} robots ready; running for {args.duration_s:.0f}s", flush=True)
                stop_at.value = time.time() + args.duration_s
                start = time.perf_counter()
                go.set()
            continue
        if item["type"] == "episode":
            episodes.append(item)
            if episodes_f:
                episodes_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                episodes_f.flush()
            print(
                f"[{item['group']} r{item['robot_id']} ep{item['episode']}] "
                f"{'success' if item['success'] else 'fail'} in {item['steps']} steps: {item['prompt']}",
                flush=True,
            )
        else:
            robots.append(item)
            if "error" in item:
                print(f"[{item['group']} r{item['robot_id']}] error: {item['error']}\n{item['traceback']}", flush=True)
    for p in procs:
        p.join(timeout=10)
    wall_s = time.perf_counter() - start
    if episodes_f:
        episodes_f.close()

    summaries = []
    for g in groups:
        group_robots = [r for r in robots if r["group"] == g.name]
        group_episodes = [e for e in episodes if e["group"] == g.name]
        summary = summarize(args.label, g.name, group_robots, group_episodes, wall_s)
        summaries.append({**summary, "model": g.model, **describe(g.env)})
    summaries.append(summarize(args.label, "all", robots, episodes, wall_s))
    if args.video_grid and args.video_dir and any(e.get("video") for e in episodes):
        from video_grid import find_clips, make_grid

        fps = max(g.opts["control_hz"] / g.opts["video_stride"] for g in groups) or 20.0
        grid = make_grid(find_clips(args.video_dir, 16), args.video_dir / "grid.mp4", cols=4, fps=fps)
        print(f"grid video: {grid}", flush=True)
    lines = [json.dumps(s, ensure_ascii=False) for s in summaries]
    print("\n".join(lines))
    if args.output:
        with args.output.open("a") as f:
            f.write("\n".join(lines) + "\n")
    return 1 if any("error" in r for r in robots) or len(robots) < len(jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
