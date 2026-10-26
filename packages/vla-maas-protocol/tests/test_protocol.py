# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
from pathlib import Path

import numpy as np
import pytest
from vla_maas_protocol import (
    Provides,
    Requirements,
    RobotHello,
    ServerHello,
    compatibility_problems,
    decode_image,
    encode_jpeg,
    obs_problems,
    packb,
    unpackb,
)
from vla_maas_protocol import geometry as geo
from vla_maas_protocol.version import server_accepts

SCHEMA = json.loads((Path(__file__).parents[1] / "spec" / "schema.json").read_text())


def _server() -> ServerHello:
    return ServerHello(
        model="m",
        requires=Requirements(images={"front": [224, 224], "wrist": [224, 224]}, state=["joint_pos", "gripper"]),
        provides=Provides(space="joint_pos", fields=["joint_pos", "gripper"], horizon=10, control_hz=30.0),
    )


def _robot(**kw) -> RobotHello:
    base = dict(
        robot_id="r1",
        embodiment="piper",
        dof=6,
        images=["front", "wrist"],
        state=["joint_pos", "eef_pose", "gripper"],
        action_spaces=["joint_pos"],
    )
    return RobotHello(**{**base, **kw})


def test_codec_round_trip_and_openpi_markers():
    msg = {"type": "obs", "state": {"joint_pos": np.arange(6, dtype=np.float32)}, "n": np.int64(3), "s": "x"}
    out = unpackb(packb(msg))
    np.testing.assert_array_equal(out["state"]["joint_pos"], msg["state"]["joint_pos"])
    assert out["state"]["joint_pos"].dtype == np.float32
    assert out["n"] == 3 and out["s"] == "x"
    # openpi-client decodes our arrays and we decode theirs.
    openpi = pytest.importorskip("openpi_client.msgpack_numpy")
    np.testing.assert_array_equal(openpi.unpackb(packb(msg))["state"]["joint_pos"], np.arange(6))
    np.testing.assert_array_equal(unpackb(openpi.packb({"a": np.ones(3)}))["a"], np.ones(3))


def test_codec_rejects_object_arrays():
    with pytest.raises(ValueError):
        packb({"a": np.array([object()])})


def test_handshake_round_trip_matches_schema():
    server, robot = _server(), _robot()
    assert ServerHello.from_wire(unpackb(packb(server.to_wire()))) == server
    assert RobotHello.from_wire(unpackb(packb(robot.to_wire()))) == robot
    for name, msg in (("server_hello", server.to_wire()), ("robot_hello", robot.to_wire())):
        assert set(SCHEMA["$defs"][name]["required"]) <= set(msg)
        assert set(msg) <= set(SCHEMA["$defs"][name]["properties"])


def test_compatibility_problems():
    assert compatibility_problems(_server(), _robot()) == []
    problems = compatibility_problems(_server(), _robot(images=["front"], action_spaces=["eef_pose"]))
    assert any("wrist" in p for p in problems)
    assert any("outputs 'joint_pos'" in p for p in problems)
    assert compatibility_problems(_server(), _robot(protocol="maas-robot/9"))
    assert compatibility_problems(_server(), _robot(images=["front", "wrist", "belly"]))


def test_version_window():
    assert server_accepts("maas-robot/1")
    assert server_accepts("maas-robot/1", server_major=2)
    assert not server_accepts("maas-robot/1", server_major=3)
    assert not server_accepts("openpi/1")


def test_obs_problems():
    req = _server().requires
    obs = {"images": {"front": 0, "wrist": 0}, "state": {"joint_pos": np.zeros(6), "gripper": np.zeros(1)}}
    assert obs_problems(obs, req, dof=6) == []
    obs["state"]["joint_pos"] = np.zeros(7)
    del obs["images"]["wrist"]
    assert len(obs_problems(obs, req, dof=6)) == 2


def test_jpeg_round_trip():
    pytest.importorskip("PIL")
    image = np.full((32, 48, 3), 120, dtype=np.uint8)
    decoded = decode_image(unpackb(packb({"i": encode_jpeg(image)}))["i"])
    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(int) - 120).max() <= 3


@pytest.mark.parametrize("rotation", ["quat", "axis_angle", "rpy"])
def test_rotation_round_trip(rotation):
    q = geo.quat_normalize(np.array([0.1, -0.3, 0.2, 0.9]))
    back = geo.quat_from(rotation, geo.quat_to(rotation, q))
    assert min(np.abs(back - q).max(), np.abs(back + q).max()) < 1e-9


def test_quat_rotate():
    q = geo.rpy_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(geo.quat_rotate(q, np.array([1.0, 0, 0])), [0, 1, 0], atol=1e-12)
