# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rotation helpers shared by robot and model adapters (quaternions are x, y, z, w)."""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=np.float64)
    q[..., :3] *= -1
    return q


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    q = np.where(q[..., 3:4] < 0, -q, q)  # shortest rotation
    sin_half = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, q[..., 3:4])
    scale = np.where(sin_half > 1e-12, angle / np.maximum(sin_half, 1e-12), 2.0)
    return q[..., :3] * scale


def axis_angle_to_quat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = angle / 2.0
    scale = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate((v * scale, np.cos(half)), axis=-1)


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Extrinsic roll-pitch-yaw (x, then y, then z) to quaternion."""
    r, p, y = np.moveaxis(np.asarray(rpy, dtype=np.float64) / 2.0, -1, 0)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quat_normalize(q), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    vq = np.concatenate((v, np.zeros(v.shape[:-1] + (1,))), axis=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[..., :3]


ROTATIONS = {
    "quat": (4, lambda q: q, quat_normalize),
    "axis_angle": (3, quat_to_axis_angle, axis_angle_to_quat),
    "rpy": (3, quat_to_rpy, rpy_to_quat),
}


def rotation_dim(rotation: str) -> int:
    return ROTATIONS[rotation][0]


def quat_to(rotation: str, q: np.ndarray) -> np.ndarray:
    return ROTATIONS[rotation][1](q)


def quat_from(rotation: str, r: np.ndarray) -> np.ndarray:
    return ROTATIONS[rotation][2](r)
