# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Message shapes of the standard Robot protocol and the checks both sides run.

Messages are plain dicts (msgpack maps) tagged by ``type``; see
``spec/robot_protocol.md`` for the full sequence. The dataclasses here build
and parse the two handshake messages; observation and action frames stay
dicts because they carry arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from vla_maas_protocol import version
from vla_maas_protocol.vocab import ACTION_SPACES, IMAGE_ROLES, STATE_FIELDS, field_dim

# Close codes the gateway uses on the robot websocket.
CLOSE_INCOMPATIBLE = 4400
CLOSE_DUPLICATE_ROBOT = 4409


@dataclass
class Requirements:
    """What a model needs from every observation."""

    images: dict[str, list[int]] = field(default_factory=dict)  # role -> [height, width]
    state: list[str] = field(default_factory=list)


@dataclass
class Provides:
    """What a model returns: absolute targets for ``fields`` at ``control_hz``."""

    space: str = "joint_pos"
    fields: list[str] = field(default_factory=lambda: ["joint_pos", "gripper"])
    horizon: int = 1
    control_hz: float = 10.0


@dataclass
class ServerHello:
    model: str
    requires: Requirements
    provides: Provides
    protocol: str = version.PROTOCOL

    def to_wire(self) -> dict:
        return {"type": "server_hello", **asdict(self)}

    @classmethod
    def from_wire(cls, msg: dict) -> ServerHello:
        _expect(msg, "server_hello")
        return cls(
            model=msg["model"],
            requires=Requirements(**msg["requires"]),
            provides=Provides(**msg["provides"]),
            protocol=msg["protocol"],
        )


@dataclass
class RobotHello:
    robot_id: str
    embodiment: str
    dof: int
    images: list[str]
    state: list[str]
    action_spaces: list[str]
    image_encodings: list[str] = field(default_factory=lambda: ["raw"])
    protocol: str = version.PROTOCOL

    def to_wire(self) -> dict:
        return {"type": "robot_hello", **asdict(self)}

    @classmethod
    def from_wire(cls, msg: dict) -> RobotHello:
        _expect(msg, "robot_hello")
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in msg.items() if k in fields})

    def problems(self) -> list[str]:
        out = [f"unknown camera role {r!r}" for r in self.images if r not in IMAGE_ROLES]
        out += [f"unknown state field {f!r}" for f in self.state if f not in STATE_FIELDS]
        out += [f"unknown action space {s!r}" for s in self.action_spaces if s not in ACTION_SPACES]
        if int(self.dof) <= 0:
            out.append(f"dof must be positive, got {self.dof}")
        return out


def compatibility_problems(server: ServerHello, robot: RobotHello) -> list[str]:
    """Everything that stops ``robot`` from running ``server``'s model; empty when compatible."""
    problems = robot.problems()
    if not version.server_accepts(robot.protocol):
        problems.append(f"protocol {robot.protocol!r} not accepted by {server.protocol!r}")
    problems += [f"model needs camera {r!r}" for r in server.requires.images if r not in robot.images]
    problems += [f"model needs state field {f!r}" for f in server.requires.state if f not in robot.state]
    if server.provides.space not in robot.action_spaces:
        problems.append(f"model outputs {server.provides.space!r}, robot accepts {robot.action_spaces}")
    return problems


def obs_problems(obs: dict, requires: Requirements, dof: int) -> list[str]:
    """Shape checks on one observation frame against the model's requirements."""
    problems = []
    images = obs.get("images") or {}
    for role in requires.images:
        if role not in images:
            problems.append(f"missing image {role!r}")
    state = obs.get("state") or {}
    for name in requires.state:
        value = state.get(name)
        if value is None:
            problems.append(f"missing state {name!r}")
            continue
        want = field_dim(name, dof)
        if np.asarray(value).shape != (want,):
            problems.append(f"state {name!r} has shape {np.asarray(value).shape}, want ({want},)")
    return problems


def error(code: str, message: str, problems: list[str] | None = None) -> dict:
    return {"type": "error", "code": code, "message": message, "problems": problems or []}


def _expect(msg: Any, kind: str) -> None:
    if not isinstance(msg, dict) or msg.get("type") != kind:
        got = msg.get("type") if isinstance(msg, dict) else type(msg).__name__
        if isinstance(msg, dict) and got == "error":
            raise ProtocolError(msg)
        raise ValueError(f"expected {kind!r} message, got {got!r}")


class ProtocolError(RuntimeError):
    """An ``{"type": "error"}`` message from the peer."""

    def __init__(self, msg: dict) -> None:
        self.code = msg.get("code", "error")
        self.problems = list(msg.get("problems") or [])
        detail = "; ".join(self.problems)
        super().__init__(f"{self.code}: {msg.get('message', '')}" + (f" ({detail})" if detail else ""))
