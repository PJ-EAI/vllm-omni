# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-model translation between the standard Robot protocol and a model's native frames.

Robots speak one protocol (``vla_maas_protocol``); every model keeps its own
observation keys, state vector, action layout and normalization. A
``ModelAdapter`` sits in the gateway between the two, so neither the robots nor
the model servers change when the other side does.

``GenericAdapter`` covers models whose native frame is "camera keys + one
state vector + prompt" in and "an ``[horizon, width]`` array" out (π0, π0.5,
the mock backend). It is configured under ``models.<name>.adapter`` in the
gateway YAML::

    adapter:
      kind: generic
      images: {front: observation.images.image, wrist: observation.images.image2}  # role -> native key
      image_size: [224, 224]          # or per role: {front: [224, 224], wrist: [224, 224]}
      state: [joint_pos, {field: gripper, range: [-1, 1]}]  # concatenated, in order
      state_dim: 32                   # optional zero padding
      action:
        space: joint_pos              # what the robot receives
        layout: [{field: joint_pos, mode: delta}, {field: gripper, range: [-1, 1]}]  # native columns
      control_hz: 20
      horizon: null                   # null = the backend's action_horizon
      native_extra: {}                # constant keys added to every native frame
      session_id: false               # add the session id for stateful models
      embodiments: [piper, franka]    # optional allow-list
      per_embodiment: {franka: {state: [...]}}  # overrides picked by ?embodiment=

Segments: ``field`` is a protocol field. ``range: [closed, open]`` maps the
protocol gripper's ``[0, 1]`` onto the model's scale. ``eef_pose`` segments
take ``rotation: quat | axis_angle | rpy`` (default ``quat``). ``mode: delta``
means the model predicts ``target - observed state`` (OpenPI's relative-action
convention: every row is relative to the observation, rotations composed as
``delta * observed``); the adapter returns absolute targets.

Models with other native shapes (dict outputs, extra session state) subclass
``ModelAdapter`` and register a ``kind`` in ``ADAPTERS``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from vla_maas_protocol import (
    ACTION_SPACES,
    Provides,
    Requirements,
    RobotHello,
    decode_image,
    field_dim,
    resize_with_pad,
)
from vla_maas_protocol import geometry as geo


@dataclass
class Segment:
    field: str
    mode: str = "absolute"  # absolute | delta
    rotation: str = "quat"  # eef_pose only
    range: tuple[float, float] | None = None  # gripper only: native value at closed, at open

    @classmethod
    def parse(cls, spec: Any) -> Segment:
        spec = {"field": spec} if isinstance(spec, str) else dict(spec)
        seg = cls(**spec)
        if seg.field not in ("joint_pos", "joint_vel", "eef_pose", "gripper"):
            raise ValueError(f"unknown segment field {seg.field!r}")
        if seg.mode not in ("absolute", "delta"):
            raise ValueError(f"segment mode must be absolute or delta, got {seg.mode!r}")
        if seg.rotation not in geo.ROTATIONS:
            raise ValueError(f"rotation must be one of {sorted(geo.ROTATIONS)}, got {seg.rotation!r}")
        if seg.range is not None:
            seg.range = (float(seg.range[0]), float(seg.range[1]))
        return seg

    def width(self, dof: int) -> int:
        if self.field == "eef_pose":
            return 3 + geo.rotation_dim(self.rotation)
        return field_dim(self.field, dof)


class ModelAdapter:
    """Translate one robot session's frames for one model backend."""

    def __init__(self, cfg: dict, backend_metadata: dict) -> None:
        self.cfg = cfg
        self.backend_metadata = backend_metadata

    def requirements(self) -> Requirements:
        raise NotImplementedError

    def provides(self) -> Provides:
        raise NotImplementedError

    def problems(self, robot: RobotHello) -> list[str]:
        """Model-specific reasons this robot cannot use the model (beyond the protocol checks)."""
        allowed = self.cfg.get("embodiments")
        if allowed and robot.embodiment not in allowed:
            return [f"model supports embodiments {allowed}, robot is {robot.embodiment!r}"]
        return []

    def to_native(self, obs: dict, robot: RobotHello, session_id: str) -> dict:
        raise NotImplementedError

    def from_native(self, native: Any, obs: dict, robot: RobotHello) -> dict[str, np.ndarray]:
        raise NotImplementedError


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


class GenericAdapter(ModelAdapter):
    KEYS = {
        "kind",
        "images",
        "image_size",
        "state",
        "state_key",
        "state_dim",
        "prompt_key",
        "action",
        "control_hz",
        "horizon",
        "native_extra",
        "session_id",
        "embodiments",
        "per_embodiment",
    }

    def __init__(self, cfg: dict, backend_metadata: dict) -> None:
        super().__init__(cfg, backend_metadata)
        unknown = set(cfg) - self.KEYS
        if unknown:
            raise ValueError(f"generic adapter: unknown keys {sorted(unknown)}")
        self.images: dict[str, str] = dict(cfg.get("images") or {})
        size = cfg.get("image_size") or backend_metadata.get("image_resolution") or [224, 224]
        self.image_sizes = (
            {role: list(size[role]) for role in self.images}
            if isinstance(size, dict)
            else {role: list(size) for role in self.images}
        )
        self.state = [Segment.parse(s) for s in cfg.get("state") or []]
        self.state_key = cfg.get("state_key", "state")
        self.state_dim = cfg.get("state_dim")
        self.prompt_key = cfg.get("prompt_key", "prompt")
        action = dict(cfg.get("action") or {})
        self.space = action.get("space", "joint_pos")
        if self.space not in ACTION_SPACES:
            raise ValueError(f"action.space must be one of {ACTION_SPACES}, got {self.space!r}")
        self.layout = [Segment.parse(s) for s in action.get("layout") or [self.space]]
        fields = [s.field for s in self.layout]
        if self.space not in fields or not set(fields) <= {self.space, "gripper"}:
            raise ValueError(f"action.layout must hold {self.space!r} and optionally 'gripper', got {fields}")
        self.control_hz = float(cfg.get("control_hz") or 10.0)
        self.horizon = cfg.get("horizon") or backend_metadata.get("action_horizon")
        self.native_extra = dict(cfg.get("native_extra") or {})
        self.with_session_id = bool(cfg.get("session_id", False))

    @classmethod
    def for_embodiment(cls, cfg: dict, backend_metadata: dict, embodiment: str | None) -> GenericAdapter:
        override = (cfg.get("per_embodiment") or {}).get(embodiment) if embodiment else None
        return cls(_merge(cfg, override) if override else cfg, backend_metadata)

    # -- handshake ------------------------------------------------------------

    def requirements(self) -> Requirements:
        state = list(dict.fromkeys(s.field for s in self.state))
        # Delta actions are converted against the observed state, so it must be sent too.
        for seg in self.layout:
            if seg.mode == "delta" and seg.field not in state:
                state.append(seg.field)
        return Requirements(images=dict(self.image_sizes), state=state)

    def provides(self) -> Provides:
        return Provides(
            space=self.space,
            fields=[s.field for s in self.layout],
            horizon=int(self.horizon or 1),
            control_hz=self.control_hz,
        )

    def problems(self, robot: RobotHello) -> list[str]:
        problems = super().problems(robot)
        if self.state_dim is not None:
            width = sum(s.width(robot.dof) for s in self.state)
            if width > self.state_dim:
                problems.append(f"state needs {width} values for dof={robot.dof}, model takes {self.state_dim}")
        return problems

    # -- frames ---------------------------------------------------------------

    def _state_part(self, seg: Segment, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if seg.field == "gripper" and seg.range is not None:
            closed, opened = seg.range
            return closed + value * (opened - closed)
        if seg.field == "eef_pose":
            return np.concatenate((value[:3], geo.quat_to(seg.rotation, value[3:7])))
        return value

    def to_native(self, obs: dict, robot: RobotHello, session_id: str) -> dict:
        native: dict[str, Any] = dict(self.native_extra)
        for role, key in self.images.items():
            image = decode_image(obs["images"][role])
            native[key] = np.ascontiguousarray(resize_with_pad(image, *self.image_sizes[role]))
        if self.state:
            parts = [self._state_part(seg, obs["state"][seg.field]) for seg in self.state]
            vector = np.concatenate(parts).astype(np.float32)
            if self.state_dim is not None:
                vector = np.pad(vector, (0, int(self.state_dim) - vector.size))
            native[self.state_key] = vector
        native[self.prompt_key] = obs.get("instruction", "")
        if self.with_session_id:
            native["session_id"] = session_id
        return native

    def _absolute(self, seg: Segment, cols: np.ndarray, obs: dict) -> np.ndarray:
        if seg.field == "gripper":
            if seg.range is not None:
                closed, opened = seg.range
                cols = (cols - closed) / (opened - closed)
            return np.clip(cols, 0.0, 1.0)
        if seg.field == "eef_pose":
            pos, rot = cols[:, :3], geo.quat_from(seg.rotation, cols[:, 3:])
            if seg.mode == "delta":
                cur = np.asarray(obs["state"]["eef_pose"], dtype=np.float64)
                pos, rot = pos + cur[:3], geo.quat_mul(rot, cur[3:7])
            return np.concatenate((pos, geo.quat_normalize(rot)), axis=1)
        if seg.mode == "delta":
            return cols + np.asarray(obs["state"][seg.field], dtype=np.float64)
        return cols

    def from_native(self, native: Any, obs: dict, robot: RobotHello) -> dict[str, np.ndarray]:
        if isinstance(native, dict):
            native = native.get("actions")
        chunk = np.asarray(native, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"expected a [horizon, width] action array, got shape {chunk.shape}")
        if self.horizon:
            chunk = chunk[: int(self.horizon)]
        need = sum(s.width(robot.dof) for s in self.layout)
        if chunk.shape[1] < need:
            raise ValueError(f"action width {chunk.shape[1]} < {need} needed by layout for dof={robot.dof}")
        out, col = {}, 0
        for seg in self.layout:
            width = seg.width(robot.dof)
            out[seg.field] = self._absolute(seg, chunk[:, col : col + width], obs).astype(np.float32)
            col += width
        return out


ADAPTERS: dict[str, type[GenericAdapter]] = {"generic": GenericAdapter}


def make_adapter(cfg: dict, backend_metadata: dict, embodiment: str | None) -> ModelAdapter:
    kind = cfg.get("kind", "generic")
    if kind not in ADAPTERS:
        raise ValueError(f"unknown adapter kind {kind!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[kind].for_embodiment(cfg, backend_metadata, embodiment)
