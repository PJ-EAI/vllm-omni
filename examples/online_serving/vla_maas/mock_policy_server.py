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

All connections share ONE simulated GPU worker, so concurrent robots contend
for it the way they would for a single vllm-omni replica:
  * ``--max-batch 1`` (default): requests run one at a time, like a replica
    deployed with ``max_num_seqs: 1``.
  * ``--max-batch N``: dynamic batching like vllm-omni request batching; the
    worker waits up to ``--max-wait-ms`` after the first queued request, runs up
    to N together, and a batch of B takes ``latency + (B - 1) * per-item``.

Usage:
    python mock_policy_server.py --port 8000 --latency-ms 80 --jitter-ms 10
    python mock_policy_server.py --port 8000 --max-batch 32 --max-wait-ms 5 --per-item-ms 4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np
from mock_libero import ACTION_SCALE, EEF_RGB, GOAL_RGB, WORKSPACE_HIGH, WORKSPACE_LOW
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
            step = np.clip(remaining / self.args.action_scale, -1.0, 1.0)
            chunk[t, :2] = step
            remaining = remaining - step * self.args.action_scale
        return chunk

    def batch_latency_s(self, batch_size: int) -> float:
        base = self.args.latency_ms + (batch_size - 1) * self.args.per_item_ms
        return max(0.0, base + self.rng.normal(0, self.args.jitter_ms)) / 1e3


class GpuWorker:
    """One simulated accelerator: a queue drained in (optionally batched) steps."""

    def __init__(self, policy: MockPolicy, max_batch: int, max_wait_ms: float) -> None:
        self.policy = policy
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1e3
        self.queue: asyncio.Queue[tuple[dict, asyncio.Future]] = asyncio.Queue()
        self.batch_sizes: list[int] = []

    async def submit(self, obs: dict) -> np.ndarray:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((obs, future))
        return await future

    async def run(self) -> None:
        while True:
            batch = [await self.queue.get()]
            deadline = time.perf_counter() + self.max_wait_s
            while len(batch) < self.max_batch:
                timeout = deadline - time.perf_counter()
                if timeout <= 0 and self.queue.empty():
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), max(timeout, 0)))
                except TimeoutError:
                    break
            start = time.perf_counter()
            results = []
            for obs, _ in batch:
                try:
                    results.append(self.policy.infer(obs))
                except Exception as exc:
                    results.append(exc)
            target = self.policy.batch_latency_s(len(batch))
            await asyncio.sleep(max(0.0, target - (time.perf_counter() - start)))
            self.batch_sizes.append(len(batch))
            for (_, future), result in zip(batch, results):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)

    async def log_stats(self, every_s: float = 10.0) -> None:
        while True:
            await asyncio.sleep(every_s)
            if self.batch_sizes:
                sizes = np.asarray(self.batch_sizes)
                logger.info("batches=%d mean_batch=%.2f max_batch=%d", sizes.size, sizes.mean(), sizes.max())
                self.batch_sizes.clear()


async def handle(ws: ServerConnection, policy: MockPolicy, worker: GpuWorker) -> None:
    packer = msgpack_numpy.Packer()
    await ws.send(packer.pack(policy.metadata))
    async for message in ws:
        try:
            obs = msgpack_numpy.unpackb(message)
            if isinstance(obs, dict) and obs.get("endpoint") == "reset":
                await ws.send(packer.pack({"status": "ok"}))
                continue
            await ws.send(packer.pack(await worker.submit(obs)))
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
    parser.add_argument("--max-batch", type=int, default=1, help="1 = serial, like max_num_seqs: 1.")
    parser.add_argument("--max-wait-ms", type=float, default=5.0, help="Batch collection window.")
    parser.add_argument("--per-item-ms", type=float, default=4.0, help="Added latency per extra batch item.")
    parser.add_argument(
        "--action-scale", type=float, default=ACTION_SCALE, help="Must match the mock env's action_scale."
    )
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
        worker = GpuWorker(policy, args.max_batch, args.max_wait_ms)
        # Keep references so the background tasks are not garbage-collected.
        background = [asyncio.create_task(worker.run()), asyncio.create_task(worker.log_stats())]
        async with serve(
            lambda ws: handle(ws, policy, worker),
            args.host,
            args.port,
            process_request=check_path,
            compression=None,
            max_size=None,
        ) as server:
            logger.info("mock policy on ws://%s:%d%s", args.host, args.port, args.path)
            await server.serve_forever()
        del background

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
