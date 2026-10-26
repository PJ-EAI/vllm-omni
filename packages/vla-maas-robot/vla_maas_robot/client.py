# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Blocking client for the standard Robot protocol (one request in flight)."""

from __future__ import annotations

import time
from contextlib import ExitStack
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from vla_maas_protocol import ProtocolError, RobotHello, ServerHello, packb, unpackb
from websockets.sync.client import connect


def act_url(gateway: str, model: str, robot_id: str, embodiment: str | None = None) -> str:
    """``https://gw`` / ``wss://gw`` / ``gw:9000`` -> ``wss://gw/v1/robot/<model>/act?robot_id=..``.

    ``embodiment`` lets the gateway pick per-embodiment model settings before
    it sends its handshake.
    """
    if "://" not in gateway:
        gateway = f"ws://{gateway}"
    parts = urlsplit(gateway)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    if scheme not in ("ws", "wss"):
        raise ValueError(f"gateway URL must be ws://, wss://, http:// or https://, got {gateway!r}")
    path = f"{parts.path.rstrip('/')}/v1/robot/{quote(model, safe='')}/act"
    query = urlencode({"robot_id": robot_id, **({"embodiment": embodiment} if embodiment else {})})
    return urlunsplit((scheme, parts.netloc, path, query, ""))


class PolicyClient:
    def __init__(
        self,
        gateway: str,
        model: str,
        robot_id: str,
        api_key: str | None,
        embodiment: str | None = None,
        open_timeout_s: float = 10.0,
        request_timeout_s: float = 5.0,
    ) -> None:
        self.url = act_url(gateway, model, robot_id, embodiment)
        self.api_key = api_key
        self.open_timeout_s = open_timeout_s
        self.request_timeout_s = request_timeout_s
        self._ws = None
        self._stack = ExitStack()
        self.server: ServerHello | None = None
        self.session_id: str | None = None
        self.last_rtt_ms: float | None = None
        self.last_server_ms: float | None = None

    def connect(self, hello: RobotHello) -> ServerHello:
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else {}
        self._ws = self._stack.enter_context(
            connect(
                self.url,
                additional_headers=headers,
                open_timeout=self.open_timeout_s,
                max_size=None,
                compression=None,
            )
        )
        self.server = ServerHello.from_wire(self._recv())
        self._ws.send(packb(hello.to_wire()))
        reply = self._recv()
        if reply.get("type") == "error":
            self.close()
            raise ProtocolError(reply)
        if reply.get("type") != "ready":
            raise ValueError(f"expected 'ready', got {reply.get('type')!r}")
        self.session_id = reply.get("session_id")
        return self.server

    def _recv(self) -> dict:
        msg = unpackb(self._ws.recv(timeout=self.request_timeout_s))
        if not isinstance(msg, dict):
            raise ValueError(f"expected a map, got {type(msg).__name__}")
        return msg

    def request(self, msg: dict) -> dict:
        """Send one message and return the reply; ``error`` replies raise."""
        start = time.perf_counter()
        self._ws.send(packb(msg))
        reply = self._recv()
        self.last_rtt_ms = (time.perf_counter() - start) * 1e3
        self.last_server_ms = reply.get("server_ms")
        if reply.get("type") == "error":
            raise ProtocolError(reply)
        return reply

    def episode_result(self, success: bool) -> None:
        self.request({"type": "episode_result", "success": bool(success)})

    def reset(self) -> None:
        self.request({"type": "reset"})

    def close(self) -> None:
        self._stack.close()
        self._ws = None
