# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One YAML file per physical robot. It names no model-specific keys."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from vla_maas_protocol import IMAGE_ROLES

from vla_maas_robot.adapter import Calibration
from vla_maas_robot.loop import LoopConfig
from vla_maas_robot.safety import SafetyConfig

TOP_LEVEL = {"robot_id", "gateway", "driver", "cameras", "calibration", "control", "safety"}


@dataclass
class GatewayConfig:
    url: str
    model: str
    api_key: str | None = None
    request_timeout_s: float = 5.0


@dataclass
class RobotConfig:
    robot_id: str
    gateway: GatewayConfig
    driver: dict
    cameras: dict[str, dict]
    calibration: Calibration = field(default_factory=Calibration)
    control: LoopConfig = field(default_factory=LoopConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def load_robot_config(path: str | Path) -> RobotConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(data) - TOP_LEVEL
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    gw = dict(data.get("gateway") or {})
    key_env = gw.pop("api_key_env", None)
    if key_env and not gw.get("api_key"):
        gw["api_key"] = os.environ.get(key_env)
    cameras = data.get("cameras") or {}
    bad_roles = [r for r in cameras if r not in IMAGE_ROLES]
    if bad_roles:
        raise ValueError(f"{path}: camera keys must be protocol roles {IMAGE_ROLES}, got {bad_roles}")
    if "driver" not in data or "robot_id" not in data:
        raise ValueError(f"{path}: robot_id and driver are required")
    return RobotConfig(
        robot_id=str(data["robot_id"]),
        gateway=GatewayConfig(**gw),
        driver=dict(data["driver"]),
        cameras={role: dict(cfg) for role, cfg in cameras.items()},
        calibration=Calibration.from_dict(data.get("calibration")),
        control=LoopConfig.from_dict(data.get("control")),
        safety=SafetyConfig.from_dict(data.get("safety")),
    )
