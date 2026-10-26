# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import numpy as np
import pytest
from model_adapters import make_adapter
from vla_maas_protocol import RobotHello, encode_jpeg
from vla_maas_protocol import geometry as geo

CFG = {
    "kind": "generic",
    "images": {"front": "observation.images.base_0_rgb", "wrist": "observation.images.left_wrist_0_rgb"},
    "image_size": [32, 32],
    "state": ["joint_pos", {"field": "gripper", "range": [-1, 1]}],
    "state_dim": 10,
    "action": {"space": "joint_pos", "layout": [{"field": "joint_pos", "mode": "delta"}, {"field": "gripper"}]},
    "control_hz": 30,
    "per_embodiment": {"franka": {"state_dim": 12, "control_hz": 15}},
    "embodiments": ["piper", "franka"],
}


def _robot(**kw):
    base = dict(
        robot_id="r",
        embodiment="piper",
        dof=6,
        images=["front", "wrist"],
        state=["joint_pos", "gripper"],
        action_spaces=["joint_pos"],
    )
    return RobotHello(**{**base, **kw})


def _obs(dof=6):
    return {
        "type": "obs",
        "instruction": "pick",
        "images": {"front": np.zeros((48, 64, 3), np.uint8), "wrist": encode_jpeg(np.zeros((32, 32, 3), np.uint8))},
        "state": {"joint_pos": np.arange(dof, dtype=np.float32), "gripper": np.array([0.25], np.float32)},
    }


def test_handshake_and_problems():
    adapter = make_adapter(CFG, {"action_horizon": 50}, "piper")
    assert adapter.requirements().images == {"front": [32, 32], "wrist": [32, 32]}
    assert adapter.requirements().state == ["joint_pos", "gripper"]
    p = adapter.provides()
    assert (p.space, p.fields, p.horizon, p.control_hz) == ("joint_pos", ["joint_pos", "gripper"], 50, 30.0)
    assert adapter.problems(_robot()) == []
    assert adapter.problems(_robot(embodiment="ur5"))
    assert adapter.problems(_robot(dof=12))  # 13 state values do not fit state_dim 10
    franka = make_adapter(CFG, {}, "franka")
    assert franka.provides().control_hz == 15.0 and franka.state_dim == 12


def test_to_native_and_from_native():
    adapter = make_adapter(CFG, {}, "piper")
    native = adapter.to_native(_obs(), _robot(), "sess")
    assert native["observation.images.base_0_rgb"].shape == (32, 32, 3)
    assert native["observation.images.left_wrist_0_rgb"].shape == (32, 32, 3)
    np.testing.assert_allclose(native["state"], [0, 1, 2, 3, 4, 5, -0.5, 0, 0, 0])
    assert native["prompt"] == "pick" and "session_id" not in native
    chunk = np.zeros((4, 32), np.float32)
    chunk[:, :6] = 0.1
    chunk[:, 6] = [0.0, 0.5, 1.0, 2.0]
    out = adapter.from_native(chunk, _obs(), _robot())
    np.testing.assert_allclose(out["joint_pos"][0], np.arange(6) + 0.1, rtol=1e-6)
    np.testing.assert_allclose(out["gripper"][:, 0], [0.0, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError):
        adapter.from_native(np.zeros((4, 3)), _obs(), _robot())


def test_eef_delta_axis_angle():
    cfg = {
        "images": {},
        "state": [{"field": "eef_pose", "rotation": "axis_angle"}],
        "action": {"space": "eef_pose", "layout": [{"field": "eef_pose", "mode": "delta", "rotation": "axis_angle"}]},
    }
    adapter = make_adapter(cfg, {}, None)
    q0 = geo.rpy_to_quat(np.array([0.0, 0.0, 0.3]))
    obs = {"images": {}, "state": {"eef_pose": np.concatenate(([0.5, 0.0, 0.2], q0))}, "instruction": ""}
    robot = _robot(images=[], state=["eef_pose"], action_spaces=["eef_pose"])
    np.testing.assert_allclose(adapter.to_native(obs, robot, "s")["state"], [0.5, 0, 0.2, 0, 0, 0.3], atol=1e-9)
    delta = np.array([[0.01, 0.0, -0.02, 0.0, 0.0, 0.2]])
    out = adapter.from_native(delta, obs, robot)["eef_pose"][0]
    np.testing.assert_allclose(out[:3], [0.51, 0.0, 0.18], atol=1e-6)
    np.testing.assert_allclose(geo.quat_to_rpy(out[3:]), [0, 0, 0.5], atol=1e-6)


def test_config_errors():
    with pytest.raises(ValueError, match="unknown keys"):
        make_adapter({**CFG, "camera_map": {}}, {}, None)
    with pytest.raises(ValueError, match="layout"):
        make_adapter({**CFG, "action": {"space": "joint_pos", "layout": ["eef_pose"]}}, {}, None)
    with pytest.raises(ValueError, match="adapter kind"):
        make_adapter({"kind": "gr00t"}, {}, None)
