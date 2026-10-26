# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MessagePack framing with NumPy arrays, wire-compatible with ``openpi-client``.

Arrays are encoded as ``{"__ndarray__": True, "data", "dtype", "shape"}`` maps
and NumPy scalars as ``{"__npgeneric__": True, "data", "dtype"}``, the markers
``openpi_client.msgpack_numpy`` uses, so either side may decode the other's
frames. Object and complex arrays are rejected.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _pack_default(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": np.ascontiguousarray(obj).tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"cannot encode {type(obj).__name__}")


def _unpack_hook(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_default, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_hook, raw=False)
