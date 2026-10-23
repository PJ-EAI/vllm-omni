#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client E2E latency benchmark for a VLA served over the OpenPI websocket.

Measures the round trip a robot client sees (msgpack pack, send, engine,
denoise, unpack) with synthetic observations, and reports p50/p90/p99 plus the
action-chunk rate it implies for an async 50 Hz control loop.

Usage:
    python latency_bench.py --host 127.0.0.1 --port 8000 \
        --cameras observation.images.base_0_rgb observation.images.left_wrist_0_rgb \
        --state-dim 32 --num-inference-steps 10 --warmup 3 --iters 50
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
from vla_client import DEFAULT_PATH, connect_policy, infer_actions


def _make_obs(args: argparse.Namespace, session_id: str, rng: np.random.Generator) -> dict:
    size = args.image_size
    obs = {cam: rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for cam in args.cameras}
    obs["state"] = rng.uniform(-1, 1, args.state_dim).astype(np.float32)
    obs["prompt"] = args.prompt
    obs["session_id"] = session_id
    if args.num_inference_steps is not None:
        obs["sampling_params"] = {"num_inference_steps": args.num_inference_steps}
    return obs


def main() -> int:
    parser = argparse.ArgumentParser(description="VLA OpenPI latency benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--cameras", nargs="+", default=["observation.images.base_0_rgb"])
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--control-hz", type=float, default=50.0, help="Robot control rate for the budget check.")
    parser.add_argument("--api-key", default=None, help="Sent as `Authorization: Api-Key <key>`.")
    parser.add_argument("--output", type=Path, default=None, help="Write the summary as JSON.")
    args = parser.parse_args()

    policy, metadata = connect_policy(args.host, args.port, args.path, api_key=args.api_key)
    horizon = int(metadata.get("action_horizon", 0))
    print(f"server metadata: {metadata}")

    rng = np.random.default_rng(0)
    session_id = str(uuid.uuid4())
    for _ in range(args.warmup):
        infer_actions(policy, _make_obs(args, session_id, rng))

    latencies_ms = []
    for _ in range(args.iters):
        obs = _make_obs(args, session_id, rng)
        start = time.perf_counter()
        actions = infer_actions(policy, obs)
        latencies_ms.append((time.perf_counter() - start) * 1e3)
    if not np.isfinite(actions).all():
        raise RuntimeError("Server returned non-finite actions.")

    lat = np.asarray(latencies_ms)
    chunk_ms = horizon / args.control_hz * 1e3 if horizon else float("nan")
    summary = {
        "cameras": len(args.cameras),
        "num_inference_steps": args.num_inference_steps,
        "iters": args.iters,
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p99_ms": float(np.percentile(lat, 99)),
        "max_ms": float(lat.max()),
        "action_shape": list(actions.shape),
        # An async client stalls only when inference outlasts the chunk it executes.
        "chunk_duration_ms_at_control_hz": chunk_ms,
        "meets_100ms_p99": bool(np.percentile(lat, 99) <= 100.0),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
