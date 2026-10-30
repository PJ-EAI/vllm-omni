# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model routes and their revisions: what checkpoint and model type a route serves, and swapping it live.

A route (``models.<name>`` in the gateway YAML) is served by one *revision* at a
time: a set of backends, an optional ``/act`` adapter and the labels
``checkpoint`` and ``model_type``. Changing any of these and reloading the
config (``GET /v1/admin/reload`` or ``SIGHUP``) builds a new revision:

1. every backend of a changed route is *probed*: the gateway opens its OpenPI
   socket, reads the handshake and checks that the adapter accepts it and that
   the checkpoint / model type the backend reports (vllm-omni and the mock send
   them under ``server``) match the YAML labels. One failure rejects the whole
   reload and nothing changes;
2. new sessions go to the new revision at once;
3. sessions already open keep their backend and adapter until they disconnect
   (the old backends show as ``draining``), so no robot is cut mid-episode.
   ``?cutover=stop`` instead sends ``stop`` to every ``/act`` session still on an
   old revision.

The revision id is ``r<n>-<hash>``: ``n`` counts revisions of the route since the
gateway started, the hash is over the route's spec and what its backends
reported, so the same deployment gets the same hash across restarts.

``backends: []`` pauses a route: new sessions get 503 until a reload adds
backends, which is how a swap without spare GPUs stops the old servers first.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from model_adapters import make_adapter
from monitor import Backend
from openpi_client import msgpack_numpy
from websockets.asyncio.client import connect

PROBE_TIMEOUT_S = 15.0


def normalize_model_type(value: str | None) -> str | None:
    """``Pi05Pipeline`` / ``pi05`` / ``PI0.5`` -> ``pi05``; labels and backend class names compare equal."""
    if value is None:
        return None
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return text.removesuffix("pipeline") or text


@dataclass(frozen=True)
class RouteSpec:
    backends: tuple[str, ...]
    adapter: dict | None = None
    checkpoint: str | None = None
    model_type: str | None = None

    @classmethod
    def parse(cls, name: str, spec: Any) -> RouteSpec:
        if not isinstance(spec, dict) or not isinstance(spec.get("backends"), list):
            raise ValueError(f"model {name!r} needs a backends list")
        return cls(
            backends=tuple(str(url) for url in spec["backends"]),
            adapter=spec.get("adapter") or None,
            checkpoint=str(spec["checkpoint"]) if spec.get("checkpoint") else None,
            model_type=str(spec["model_type"]) if spec.get("model_type") else None,
        )

    def wire(self) -> dict:
        return {
            "backends": list(self.backends),
            "adapter": self.adapter,
            "checkpoint": self.checkpoint,
            "model_type": self.model_type,
        }


@dataclass(eq=False)
class Revision:
    model: str
    n: int
    spec: RouteSpec
    backends: list[Backend]
    checkpoint: str | None = None  # the label, or what every backend reported
    model_type: str | None = None
    created_at: float = field(default_factory=time.time)
    id: str = ""

    @classmethod
    def build(cls, model: str, n: int, spec: RouteSpec, reported: list[dict] | None = None) -> Revision:
        """``reported``: each backend's ``server`` handshake block, in ``spec.backends`` order (probe result)."""
        reported = reported or [{} for _ in spec.backends]
        checkpoint = spec.checkpoint or _common(r.get("checkpoint") for r in reported)
        model_type = spec.model_type or _common(r.get("model_type") for r in reported)
        digest = hashlib.sha1(json.dumps([spec.wire(), reported], sort_keys=True, default=str).encode()).hexdigest()
        rev = cls(model, n, spec, [], checkpoint, model_type)
        rev.id = f"r{n}-{digest[:8]}"
        rev.backends = [
            Backend(url, revision=rev.id, checkpoint=r.get("checkpoint") or checkpoint, model_type=model_type)
            for url, r in zip(spec.backends, reported)
        ]
        return rev

    def summary(self) -> dict:
        return {
            "revision": self.id,
            "checkpoint": self.checkpoint,
            "model_type": self.model_type,
            "backends": list(self.spec.backends),
            "act": self.spec.adapter is not None,
            "created_at": self.created_at,
        }


def _common(values) -> str | None:
    values = {v for v in values if v}
    return values.pop() if len(values) == 1 else None


def check_reported(spec: RouteSpec, url: str, metadata: Any) -> tuple[dict, list[str]]:
    """The backend's ``server`` block and why it cannot serve ``spec`` (empty = it can)."""
    if not isinstance(metadata, dict):
        return {}, [f"{url}: handshake is {type(metadata).__name__}, not a map"]
    server = metadata.get("server") if isinstance(metadata.get("server"), dict) else {}
    problems = []
    if spec.checkpoint and server.get("checkpoint") and server["checkpoint"] != spec.checkpoint:
        problems.append(f"{url}: serves checkpoint {server['checkpoint']!r}, config says {spec.checkpoint!r}")
    if (
        spec.model_type
        and server.get("model_type")
        and normalize_model_type(server["model_type"]) != normalize_model_type(spec.model_type)
    ):
        problems.append(f"{url}: serves model type {server['model_type']!r}, config says {spec.model_type!r}")
    if spec.adapter is not None:
        try:
            make_adapter(spec.adapter, metadata, None)
        except Exception as exc:
            problems.append(f"{url}: adapter rejects the handshake: {exc}")
    return server, problems


async def probe(url: str, timeout_s: float = PROBE_TIMEOUT_S) -> Any:
    """The backend's OpenPI handshake metadata."""
    async with connect(url, compression=None, max_size=None, open_timeout=timeout_s) as ws:
        return msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(), timeout_s))


async def probe_route(spec: RouteSpec, timeout_s: float = PROBE_TIMEOUT_S) -> tuple[list[dict], list[str]]:
    """Probe every backend of ``spec``; returns the reported ``server`` blocks and all problems."""

    async def one(url: str) -> tuple[dict, list[str]]:
        try:
            metadata = await probe(url, timeout_s)
        except Exception as exc:  # unreachable, refused, timed out, not OpenPI
            return {}, [f"{url}: probe failed: {exc!r}"]
        return check_reported(spec, url, metadata)

    results = await asyncio.gather(*(one(url) for url in spec.backends))
    reported = [server for server, _ in results]
    problems = [p for _, ps in results for p in ps]
    for key in ("checkpoint", "model_type"):
        seen = {r[key] for r in reported if r.get(key)}
        if len(seen) > 1 and not getattr(spec, key):
            problems.append(f"backends report different {key}s {sorted(seen)}; one revision serves one {key}")
    return reported, problems
