#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only mock of the vllm-omni OpenPI policy endpoint.

Speaks the same websocket protocol as ``/v1/realtime/robot/openpi`` (metadata
on connect, msgpack obs in, msgpack ``[action_horizon, max_action_dim]`` out,
``{"endpoint": "reset"}`` control message) so clients and the MaaS gateway can
be built and tested before GPUs are in the loop. Inference is a scripted
visual servo on ``mock_libero`` images plus an injected latency that stands in
for the GPU forward pass.

Usage:
    python mock_policy_server.py --port 8000 --latency-ms 80 --jitter-ms 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np
from mock_libero import EEF_RGB, GOAL_RGB, WORKSPACE_HIGH, WORKSPACE_LOW
from openpi_client import msgpack_numpy
from vla_client import DEFAULT_PATH
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

logger = logging.getLogger("mock_policy_server")


def _centroid(image: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray | None:
    mask = np.all(image == np.asarray(rgb, dtype=np.uint8), axis=-1)
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    return np.array([rows.mean(), cols.mean()])


class MockPolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.metadata = {
            "image_resolution": [224, 224],
            "n_external_cameras": 1,
            "needs_wrist_camera": True,
            "needs_stereo_camera": False,
            "needs_session_id": False,
            "action_space": "eef_delta",
            "action_horizon": args.action_horizon,
            "action_dim": 7,
            "max_action_dim": args.max_action_dim,
            "max_cameras": 2,
            "mock": True,
        }

    def infer(self, obs: dict) -> np.ndarray:
        image = obs.get(self.args.base_camera_key)
        if image is None:
            raise ValueError(f"missing camera {self.args.base_camera_key!r}")
        chunk = np.zeros((self.args.action_horizon, self.args.max_action_dim), dtype=np.float32)
        goal, eef = _centroid(image, GOAL_RGB), _centroid(image, EEF_RGB)
        chunk[:, 6] = -1.0  # gripper open
        if goal is None or eef is None:
            return chunk
        # Pixels -> metres, then a saturating proportional step per action.
        metres_per_px = (WORKSPACE_HIGH - WORKSPACE_LOW) / (image.shape[0] - 1)
        remaining = (goal - eef) * metres_per_px
        for t in range(self.args.action_horizon):
            step = np.clip(remaining / 0.02, -1.0, 1.0)  # ACTION_SCALE in mock_libero
            chunk[t, :2] = step
            remaining = remaining - step * 0.02
        return chunk

    async def infer_with_latency(self, obs: dict) -> np.ndarray:
        start = time.perf_counter()
        actions = self.infer(obs)
        target = (self.args.latency_ms + self.rng.normal(0, self.args.jitter_ms)) / 1e3
        await asyncio.sleep(max(0.0, target - (time.perf_counter() - start)))
        return actions


async def handle(ws: ServerConnection, policy: MockPolicy) -> None:
    packer = msgpack_numpy.Packer()
    await ws.send(packer.pack(policy.metadata))
    async for message in ws:
        try:
            obs = msgpack_numpy.unpackb(message)
            if isinstance(obs, dict) and obs.get("endpoint") == "reset":
                await ws.send(packer.pack({"status": "ok"}))
                continue
            await ws.send(packer.pack(await policy.infer_with_latency(obs)))
        except Exception as exc:
            logger.exception("inference failed")
            await ws.send(packer.pack({"type": "error", "message": str(exc)}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock OpenPI policy server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--latency-ms", type=float, default=80.0)
    parser.add_argument("--jitter-ms", type=float, default=10.0)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--base-camera-key", default="observation.images.image")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    policy = MockPolicy(args)

    def check_path(_: ServerConnection, request: Request) -> Response | None:
        if request.path != args.path:
            return Response(404, "Not Found", Headers(), b"unknown path\n")
        return None

    async def run() -> None:
        async with serve(
            lambda ws: handle(ws, policy),
            args.host,
            args.port,
            process_request=check_path,
            compression=None,
            max_size=None,
        ) as server:
            logger.info("mock policy on ws://%s:%d%s", args.host, args.port, args.path)
            await server.serve_forever()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
