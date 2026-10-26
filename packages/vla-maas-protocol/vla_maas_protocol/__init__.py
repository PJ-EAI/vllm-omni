# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standard Robot protocol of the VLA MaaS: the only thing robots and the cloud share."""

from vla_maas_protocol.codec import packb, unpackb
from vla_maas_protocol.images import decode_image, encode_jpeg, resize_with_pad
from vla_maas_protocol.messages import (
    CLOSE_DUPLICATE_ROBOT,
    CLOSE_INCOMPATIBLE,
    ProtocolError,
    Provides,
    Requirements,
    RobotHello,
    ServerHello,
    compatibility_problems,
    error,
    obs_problems,
)
from vla_maas_protocol.version import PROTOCOL, PROTOCOL_MAJOR
from vla_maas_protocol.vocab import ACTION_SPACES, IMAGE_ROLES, STATE_FIELDS, field_dim

__all__ = [
    "ACTION_SPACES",
    "CLOSE_DUPLICATE_ROBOT",
    "CLOSE_INCOMPATIBLE",
    "IMAGE_ROLES",
    "PROTOCOL",
    "PROTOCOL_MAJOR",
    "STATE_FIELDS",
    "ProtocolError",
    "Provides",
    "Requirements",
    "RobotHello",
    "ServerHello",
    "compatibility_problems",
    "decode_image",
    "encode_jpeg",
    "error",
    "field_dim",
    "obs_problems",
    "packb",
    "resize_with_pad",
    "unpackb",
]
