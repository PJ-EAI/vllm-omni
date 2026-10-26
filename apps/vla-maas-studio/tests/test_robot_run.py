# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Studio Robot-protocol runs: studio -> gateway (/act) -> CPU mock backend, on localhost.

The MuJoCo Piper twin tests skip without ``mujoco``, the Menagerie Piper, or a
working offscreen GL backend (set ``MUJOCO_GL=osmesa`` on headless CPUs).
"""

import asyncio
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from gateway import Gateway
from robot_run import ActRun, build_robot_catalog, parse_act_config
from vla_maas_robot.config import load_robot_config
from websockets.asyncio.server import serve
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[3]
MOCK_SERVER = ROOT / "examples/online_serving/vla_maas/mock_policy_server.py"
ROBOTS = ROOT / "packages/vla-maas-robot/robots"
KEY = "op-key"
ADAPTER = {
    "kind": "generic",
    "images": {"front": "observation.images.image", "wrist": "observation.images.image2"},
    "image_size": [64, 64],
    "state": ["joint_pos", {"field": "gripper", "range": [-1, 1]}],
    "state_dim": 32,
    "action": {
        "space": "joint_pos",
        "layout": [{"field": "joint_pos", "mode": "delta"}, {"field": "gripper", "range": [-1, 1]}],
    },
    "control_hz": 50,
    "horizon": 20,
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gateway_url():
    backend_port, gw_port = _free_port(), _free_port()
    backend = subprocess.Popen(
        [sys.executable, str(MOCK_SERVER), "--port", str(backend_port), "--latency-ms", "5", "--jitter-ms", "0"]
        + ["--idle-motion", "0.02", "--max-batch", "4"],
        cwd=MOCK_SERVER.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    backend_url = f"ws://127.0.0.1:{backend_port}/v1/realtime/robot/openpi"
    gateway = Gateway({"api_keys": [KEY], "models": {"m": {"backends": [backend_url], "adapter": ADAPTER}}})
    loop = asyncio.new_event_loop()
    ready, stop = threading.Event(), None

    async def run():
        nonlocal stop
        stop = asyncio.Event()
        async with serve(gateway.handle, "127.0.0.1", gw_port, process_request=gateway.process_request, max_size=None):
            ready.set()
            await stop.wait()

    thread = threading.Thread(target=lambda: loop.run_until_complete(run()), daemon=True)
    thread.start()
    ready.wait(10)
    deadline = time.time() + 15
    while time.time() < deadline:  # the mock backend imports openpi-client; give it time
        try:
            with connect(backend_url, open_timeout=1):
                break
        except OSError:
            time.sleep(0.2)
    yield f"http://127.0.0.1:{gw_port}"
    loop.call_soon_threadsafe(stop.set)
    thread.join(5)
    backend.terminate()
    backend.wait(5)


def _run(robot_file: Path, gateway_url: str, hardware: bool, **payload) -> tuple[ActRun, dict]:
    cfg = load_robot_config(robot_file)
    catalog = build_robot_catalog(cfg, hardware)
    config = parse_act_config({"model": "m", "instruction": "wave", "max_steps": 40, **payload}, catalog)
    run = ActRun(config, cfg, gateway=gateway_url, api_key=KEY, hardware=hardware)
    run.set_watched({0})
    run.start()
    for thread in run.threads:
        thread.join(30)
    run.stop()
    return run, run.snapshot()


def test_config_gates_motion_on_hardware():
    cfg = load_robot_config(ROBOTS / "mock.yaml")
    catalog = build_robot_catalog(cfg, hardware=True)
    assert catalog["max_robots"] == 1 and catalog["robot_config"]["model"] == "pi05_mock"
    assert parse_act_config({"instruction": "go"}, catalog).mode == "dry_run"  # the file's start_mode
    with pytest.raises(ValueError, match="moves the real arm"):
        parse_act_config({"instruction": "go", "mode": "live"}, catalog)
    assert parse_act_config({"instruction": "go", "mode": "live", "confirm_motion": True}, catalog).mode == "live"
    with pytest.raises(ValueError, match="instruction"):
        parse_act_config({"instruction": " "}, catalog)
    with pytest.raises(RuntimeError, match="no MuJoCo twin"):
        build_robot_catalog(cfg, hardware=False)


def test_hardware_mode_runs_the_robot_file(gateway_url):
    run, snap = _run(ROBOTS / "mock.yaml", gateway_url, hardware=True, mode="live", confirm_motion=True)
    session = snap["sessions"][0]
    assert session["state"] == "done", session["error"]
    robot = session["robot"]
    assert robot["robot_id"] == "mock-01" and robot["embodiment"] == "mock"
    assert robot["inferences"] >= 2 and robot["commands_sent"] == 40
    assert robot["safety"]["mode"] == "live" and robot["safety"]["commands"] == 40
    assert robot["last_episode"]["ticks"] == 40 and robot["provides"]["space"] == "joint_pos"
    assert snap["global"]["latency_p50_ms"] is not None
    kinds = {kind for kind, _, _ in run.drain_frames()}
    assert kinds == {1, 2}  # thumbnails and, for the watched session, the side-by-side view


def test_dry_run_sends_nothing(gateway_url):
    _, snap = _run(ROBOTS / "mock.yaml", gateway_url, hardware=True, mode="dry_run")
    robot = snap["sessions"][0]["robot"]
    assert robot["commands_sent"] == 0 and robot["safety"]["commands"] == 40


def _twin_or_skip(render: bool = False):
    # piper_sim picks the GL backend (MUJOCO_GL) before mujoco is first imported.
    piper_sim = pytest.importorskip("piper_sim", exc_type=ImportError)
    PiperSimDriver, piper_model_path = piper_sim.PiperSimDriver, piper_sim.piper_model_path

    if not piper_model_path().is_file():
        pytest.skip("MuJoCo Menagerie agilex_piper not found")
    if render and not _can_render():
        pytest.skip("no working offscreen GL backend (try MUJOCO_GL=osmesa)")
    return PiperSimDriver


_RENDER_OK: list[bool] = []


def _can_render() -> bool:
    # A broken EGL can hang inside the renderer with the GIL held, so probe in a subprocess.
    if not _RENDER_OK:
        probe = "import mujoco; mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8).close()"
        try:
            ok = subprocess.run([sys.executable, "-c", probe], timeout=30, capture_output=True).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        _RENDER_OK.append(ok)
    return _RENDER_OK[0]


def test_twin_matches_the_real_driver_spec():
    sim_driver = _twin_or_skip()
    from vla_maas_robot.driver import Command
    from vla_maas_robot.drivers.piper import PiperDriver

    cfg = load_robot_config(ROBOTS / "piper.yaml")
    driver_cfg = {k: v for k, v in cfg.driver.items() if k != "kind"}
    sim, real = sim_driver(**driver_cfg), PiperDriver(**driver_cfg)
    for name in ("embodiment", "dof", "modes", "gripper_range", "has_eef_pose", "has_gripper"):
        assert getattr(sim.spec, name) == getattr(real.spec, name)
    assert np.array_equal(sim.spec.joint_limits, real.spec.joint_limits)
    assert np.array_equal(sim.spec.max_joint_vel, real.spec.max_joint_vel)
    sim.connect()
    state = sim.read_state()
    assert np.allclose(state.joint_pos, cfg.driver["home_joints"]) and state.gripper == pytest.approx(0.07)
    target = np.array([0.1, 0.5, -0.5, 0.0, 0.2, 0.0])
    sim.command(Command("joint_pos", target, gripper=0.02))
    deadline = time.time() + 5
    while time.time() < deadline and not np.allclose(sim.read_state().joint_pos, target, atol=1e-3):
        time.sleep(0.05)
    state = sim.read_state()
    assert np.allclose(state.joint_pos, target, atol=1e-3)
    assert np.linalg.norm(state.eef_pose[3:]) == pytest.approx(1.0)


def test_twin_runs_piper_yaml(gateway_url, tmp_path):
    _twin_or_skip(render=True)
    robot = yaml.safe_load((ROBOTS / "piper.yaml").read_text())
    # Start from a ready pose: at the rest pose (joint 2/3 on their limits) the mock's
    # downward deltas are clamped and the safety filter stops the episode, as on the real arm.
    robot["driver"]["home_joints"] = [0.0, 1.0, -1.0, 0.0, 0.4, 0.0]
    robot_file = tmp_path / "piper.yaml"
    robot_file.write_text(yaml.safe_dump(robot))
    _, snap = _run(robot_file, gateway_url, hardware=False, mode="live", robots=2)
    for session in snap["sessions"]:
        assert session["state"] == "done", session["error"]
        assert session["robot"]["embodiment"] == "piper" and session["robot"]["commands_sent"] == 40
    assert [s["robot"]["robot_id"] for s in snap["sessions"]] == ["piper-01-0", "piper-01-1"]
