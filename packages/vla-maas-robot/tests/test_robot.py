# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests of the robot client. They need only this package and vla-maas-protocol."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from vla_maas_protocol import Provides, Requirements, ServerHello
from vla_maas_robot.adapter import Calibration, RobotAdapter
from vla_maas_robot.cameras import MockCamera, make_camera
from vla_maas_robot.client import act_url
from vla_maas_robot.config import load_robot_config
from vla_maas_robot.driver import Command, make_driver
from vla_maas_robot.drivers.mock import MockDriver
from vla_maas_robot.loop import ChunkQueue, ControlLoop, LoopConfig
from vla_maas_robot.safety import SafetyConfig, SafetyError, SafetyFilter

ROBOTS = Path(__file__).parents[1] / "robots"


def _server(horizon=10, hz=50.0) -> ServerHello:
    return ServerHello(
        model="m",
        requires=Requirements(images={"front": [64, 64]}, state=["joint_pos", "gripper"]),
        provides=Provides(space="joint_pos", fields=["joint_pos", "gripper"], horizon=horizon, control_hz=hz),
    )


def _cmd(values, gripper=None) -> Command:
    return Command(mode="joint_pos", target=np.asarray(values, dtype=float), gripper=gripper)


def test_no_server_side_imports():
    import sys

    import vla_maas_robot.robotctl  # noqa: F401

    assert not {"vllm", "vllm_omni", "torch", "gateway", "model_adapters"} & set(sys.modules)


def test_act_url():
    assert act_url("https://gw.example.com", "pi05", "piper-01", "piper") == (
        "wss://gw.example.com/v1/robot/pi05/act?robot_id=piper-01&embodiment=piper"
    )
    assert act_url("127.0.0.1:9000", "m", "r") == "ws://127.0.0.1:9000/v1/robot/m/act?robot_id=r"
    with pytest.raises(ValueError):
        act_url("ftp://x", "m", "r")


def test_adapter_obs_and_commands():
    driver = MockDriver(dof=6)
    adapter = RobotAdapter(driver.spec, ["front", "wrist"], Calibration(gripper_raw_range=(0.0, 0.08)))
    adapter.configure(_server())
    state = driver.read_state()
    frames = {"front": (0.0, np.zeros((48, 96, 3), np.uint8)), "wrist": (0.0, np.zeros((48, 48, 3), np.uint8))}
    obs = adapter.build_obs(state, frames, seq=1, instruction="go", episode_id="e")
    assert set(obs["images"]) == {"front"}  # the model does not use the wrist camera
    assert obs["images"]["front"].shape == (64, 64, 3)
    assert obs["state"]["gripper"].tolist() == [1.0]  # 0.08 m is fully open
    assert obs["state"]["eef_pose"].shape == (7,)
    msg = {
        "space": "joint_pos",
        "mode": "absolute",
        "actions": {"joint_pos": np.ones((3, 6), np.float32), "gripper": np.array([[0.0], [0.5], [1.0]])},
    }
    cmds = adapter.to_commands(msg)
    assert [c.gripper for c in cmds] == pytest.approx([0.0, 0.04, 0.08])
    with pytest.raises(ValueError):
        adapter.to_commands({**msg, "actions": {"joint_pos": np.ones((3, 7))}})


def test_adapter_base_frame_round_trip():
    driver = MockDriver(dof=6)
    cal = Calibration(base_xyz=(0.1, -0.2, 0.3), base_rpy=(0.0, 0.0, np.pi / 2))
    adapter = RobotAdapter(driver.spec, [], cal)
    pose = np.array([0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
    std = adapter.pose_to_std(pose)
    np.testing.assert_allclose(std[:3], [0.1 - 0.1, -0.2 + 0.4, 0.5], atol=1e-9)
    np.testing.assert_allclose(adapter.pose_from_std(std), pose, atol=1e-9)


def test_safety_limits_and_streak():
    spec = MockDriver(dof=2).spec
    f = SafetyFilter(spec, SafetyConfig(start_mode="live", max_joint_step_rad=0.1, max_violations=3), 50.0)
    state = MockDriver(dof=2).read_state()
    out = f.filter(_cmd([0.05, -0.05], gripper=1.0), state)
    assert out.target.tolist() == [0.05, -0.05] and out.gripper == 0.08
    out = f.filter(_cmd([1.0, 0.0]), state)  # step limit from the last target
    assert out.target[0] == pytest.approx(0.15)
    f.filter(_cmd([1.0, 0.0]), state)
    with pytest.raises(SafetyError):
        f.filter(_cmd([1.0, 0.0]), state)
    slow = SafetyFilter(spec, SafetyConfig(start_mode="slow", max_joint_step_rad=0.1, slow_scale=0.5), 50.0)
    assert slow.filter(_cmd([1.0, 0.0]), state).target[0] == pytest.approx(0.05)
    limit = SafetyFilter(spec, SafetyConfig(start_mode="live", max_joint_step_rad=10), 50.0)
    assert limit.filter(_cmd([9.0, 0.0]), state).target[0] == pytest.approx(2.75)
    assert not SafetyFilter(spec, SafetyConfig(), 50.0).sends  # dry_run by default


def test_safety_workspace():
    driver = MockDriver(dof=6)
    state = driver.read_state()  # eef at x = 0.45
    cfg = SafetyConfig(start_mode="live", workspace={"x": [0.0, 0.3], "y": [-1, 1], "z": [0, 1]})
    with pytest.raises(SafetyError):
        SafetyFilter(driver.spec, cfg, 50.0).check_state(state)


def test_chunk_queue_alignment_and_blend():
    q = ChunkQueue()
    chunk = [_cmd([i]) for i in range(10)]
    assert q.install(chunk, obs_tick=3, now_tick=7) == 4
    assert q.pop().target[0] == 4
    blended = ChunkQueue(blend_steps=1)
    blended.install([_cmd([0.0])] * 5, 0, 0)
    blended.install([_cmd([1.0])] * 5, 0, 0)
    assert blended.pop().target[0] == pytest.approx(0.5)
    assert blended.pop().target[0] == pytest.approx(1.0)


def test_config_and_factories(tmp_path):
    cfg = load_robot_config(ROBOTS / "mock.yaml")
    assert cfg.robot_id == "mock-01" and set(cfg.cameras) == {"front", "wrist"}
    assert cfg.safety.start_mode == "dry_run"
    assert isinstance(make_driver(cfg.driver), MockDriver)
    assert isinstance(make_camera(cfg.cameras["front"]), MockCamera)
    bad = tmp_path / "bad.yaml"
    bad.write_text("robot_id: x\ndriver: {kind: mock}\ncameras: {base_0_rgb: {kind: mock}}\n")
    with pytest.raises(ValueError, match="protocol roles"):
        load_robot_config(bad)
    with pytest.raises(ValueError, match="unknown driver"):
        make_driver({"kind": "nope"})


class FakeClient:
    """Replies with absolute joint chunks after ``latency_s``; ``stop_after`` replies stop."""

    def __init__(self, server: ServerHello, latency_s: float = 0.0, stop_after: int | None = None) -> None:
        self.server, self.latency_s, self.stop_after = server, latency_s, stop_after
        self.request_timeout_s, self.last_rtt_ms, self.requests = 5.0, None, []

    def request(self, obs):
        self.requests.append(obs)
        time.sleep(self.latency_s)
        self.last_rtt_ms = self.latency_s * 1e3
        if self.stop_after is not None and len(self.requests) > self.stop_after:
            return {"type": "stop", "reason": "test"}
        h = self.server.provides.horizon
        target = obs["state"]["joint_pos"] + 0.01 * np.arange(1, h + 1)[:, None]
        return {
            "type": "actions",
            "space": "joint_pos",
            "mode": "absolute",
            "server_ms": 1.0,
            "actions": {"joint_pos": target.astype(np.float32), "gripper": np.ones((h, 1))},
        }


def _loop(client, mode="live", **control):
    driver = MockDriver(dof=6)
    driver.connect()
    cam = MockCamera(width=64, height=64, fps=200)
    cam.start()
    adapter = RobotAdapter(driver.spec, ["front"])
    adapter.configure(client.server)
    safety = SafetyFilter(driver.spec, SafetyConfig(start_mode=mode), client.server.provides.control_hz)
    loop = ControlLoop(driver, {"front": cam}, adapter, safety, client, LoopConfig(**control), log=lambda _: None)
    return loop, driver, cam


def test_loop_runs_aligns_and_counts():
    client = FakeClient(_server(horizon=10, hz=50.0), latency_s=0.05)
    loop, driver, cam = _loop(client, prefetch_threshold=4)
    try:
        stats = loop.run_episode("go", max_steps=40)
    finally:
        cam.stop()
    assert stats.stopped is None
    assert stats.commands_sent == len(driver.commands) > 20
    assert stats.inferences >= 3
    assert stats.dropped_rows > 0  # ~2-3 ticks elapse during each 50 ms request at 50 Hz
    assert stats.summary()["network_p50_ms"] == pytest.approx(49.0)


def test_loop_dry_run_and_stop():
    client = FakeClient(_server(), stop_after=1)
    loop, driver, cam = _loop(client, mode="dry_run", prefetch_threshold=8)
    try:
        stats = loop.run_episode("go", max_steps=100)
    finally:
        cam.stop()
    assert driver.commands == [] and stats.commands_sent == 0
    assert stats.stopped.startswith("gateway stop")


def test_loop_operator_stop():
    client = FakeClient(_server())
    loop, _, cam = _loop(client)
    stop = threading.Event()
    stop.set()
    try:
        assert loop.run_episode("go", max_steps=100, stop=stop).stopped == "operator"
    finally:
        cam.stop()
