# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared OpenPI websocket helpers and an async action-chunk executor.

``AsyncChunkExecutor`` decouples the control loop from inference: the robot
pops one action per control tick from a queue, and a background thread asks the
server for the next chunk once the queue drops below ``prefetch_threshold``.
With a 50-step chunk at 50 Hz the loop has ~1 s of actions buffered, so a
<100 ms inference never stalls it.
"""

from __future__ import annotations

import collections
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PATH = "/v1/realtime/robot/openpi"


def _import_websocket_policy():
    # Avoid shadowing the installed ``openpi_client`` package with sibling files.
    example_dir = str(Path(__file__).resolve().parent)
    removed = sys.path and sys.path[0] == example_dir
    if removed:
        sys.path.pop(0)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("VLA MaaS clients require `openpi-client`.") from exc
    finally:
        if removed:
            sys.path.insert(0, example_dir)
    return WebsocketClientPolicy


def ws_uri(host: str, port: int, path: str) -> str:
    if host.startswith(("ws://", "wss://")):
        return host
    return f"ws://{host}:{port}{path}"


def connect_policy(host: str, port: int, path: str = DEFAULT_PATH, api_key: str | None = None) -> tuple[Any, dict]:
    policy = _import_websocket_policy()(host=ws_uri(host, port, path), api_key=api_key)
    return policy, dict(policy.get_server_metadata())


def close_policy(policy: Any) -> None:
    """Close the session's websocket so the server sees a clean 1000 close."""
    ws = getattr(policy, "_ws", None)
    if ws is not None:
        ws.close()


def infer_actions(policy: Any, obs: dict) -> np.ndarray:
    response = policy.infer(obs)
    if isinstance(response, dict) and response.get("type") == "error":
        raise RuntimeError(f"Inference failed: {response.get('message', response)}")
    if isinstance(response, dict) and "actions" in response:
        response = response["actions"]
    return np.asarray(response, dtype=np.float32)


class AsyncChunkExecutor:
    """Background-prefetching action queue for a fixed-rate control loop."""

    def __init__(
        self,
        policy: Any,
        action_dim: int,
        prefetch_threshold: int,
        execute_steps: int | None = None,
    ) -> None:
        self.policy = policy
        self.action_dim = action_dim
        self.prefetch_threshold = prefetch_threshold
        self.execute_steps = execute_steps
        self._queue: collections.deque[np.ndarray] = collections.deque()
        self._lock = threading.Lock()
        self._pending: threading.Thread | None = None
        self._error: BaseException | None = None
        self.latencies_ms: list[float] = []
        # Async-only: ticks that found the queue empty after the first chunk.
        self.underruns = 0
        self._primed = False

    def reset(self) -> None:
        self.wait()
        with self._lock:
            self._queue.clear()
        self._primed = False

    def wait(self) -> None:
        if self._pending is not None:
            self._pending.join()
            self._pending = None
        if self._error is not None:
            raise self._error

    def _fetch(self, obs: dict) -> None:
        try:
            start = time.perf_counter()
            chunk = infer_actions(self.policy, obs)[:, : self.action_dim]
            self.latencies_ms.append((time.perf_counter() - start) * 1e3)
            if self.execute_steps:
                chunk = chunk[: self.execute_steps]
            with self._lock:
                # Fresh predictions replace whatever is left of the stale chunk.
                self._queue.clear()
                self._queue.extend(chunk)
        except BaseException as exc:  # surfaced on the control thread
            self._error = exc

    def next_action(self, make_obs: Callable[[], dict]) -> np.ndarray:
        """Return the action for this tick, prefetching the next chunk if due."""
        if self._error is not None:
            raise self._error
        with self._lock:
            remaining = len(self._queue)
        if remaining <= self.prefetch_threshold and self._pending is None:
            self._pending = threading.Thread(target=self._fetch, args=(make_obs(),), daemon=True)
            self._pending.start()
        if remaining == 0:
            if self._primed and self.prefetch_threshold > 0:
                self.underruns += 1
            self.wait()
            self._primed = True
        elif self._pending is not None and not self._pending.is_alive():
            self.wait()
        with self._lock:
            return self._queue.popleft()
