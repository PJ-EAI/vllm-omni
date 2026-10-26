# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed vocabulary of the standard Robot protocol.

Robots and models never exchange model-specific keys: cameras are named by
role and state/action by field. All quantities are SI: radians, metres,
seconds. ``eef_pose`` is ``xyz (m) + quaternion (x, y, z, w)`` in the robot
base frame. ``gripper`` is opening in ``[0, 1]`` (0 closed, 1 open).
"""

from __future__ import annotations

IMAGE_ROLES = ("front", "top", "left", "right", "wrist", "wrist_left", "wrist_right")

# Field -> fixed width, or None when the width is the robot's joint count.
STATE_FIELDS: dict[str, int | None] = {
    "joint_pos": None,
    "joint_vel": None,
    "eef_pose": 7,
    "gripper": 1,
}

# An action chunk carries one arm field (the space) plus optionally ``gripper``.
ACTION_SPACES = ("joint_pos", "eef_pose")


def field_dim(field: str, dof: int) -> int:
    if field not in STATE_FIELDS:
        raise ValueError(f"unknown field {field!r}; expected one of {sorted(STATE_FIELDS)}")
    width = STATE_FIELDS[field]
    return dof if width is None else width
