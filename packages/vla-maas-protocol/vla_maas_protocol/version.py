# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Protocol versioning: ``maas-robot/<major>``.

A server accepts its own major version and the previous one, so robots and
the cloud side can be upgraded independently.
"""

from __future__ import annotations

PROTOCOL_NAME = "maas-robot"
PROTOCOL_MAJOR = 1
PROTOCOL = f"{PROTOCOL_NAME}/{PROTOCOL_MAJOR}"


def parse(protocol: str) -> int:
    name, _, major = str(protocol).partition("/")
    if name != PROTOCOL_NAME or not major.isdigit():
        raise ValueError(f"not a {PROTOCOL_NAME} protocol string: {protocol!r}")
    return int(major)


def server_accepts(protocol: str, server_major: int = PROTOCOL_MAJOR) -> bool:
    try:
        major = parse(protocol)
    except ValueError:
        return False
    return server_major - 1 <= major <= server_major
