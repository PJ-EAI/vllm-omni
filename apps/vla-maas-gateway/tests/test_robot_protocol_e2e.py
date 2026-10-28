# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Robot client -> gateway (/act) -> CPU mock OpenPI backend, all on localhost.

Needs vla-maas-protocol, vla-maas-robot, openpi-client and websockets.
"""

import asyncio
import base64
import contextlib
import io
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import yaml
from gateway import Gateway
from vla_maas_protocol import ProtocolError, RobotHello, packb, unpackb
from vla_maas_robot import robotctl
from vla_maas_robot.client import PolicyClient, act_url
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[3]
MOCK_SERVER = ROOT / "examples/online_serving/vla_maas/mock_policy_server.py"
OPERATOR, ROBOT_KEY = "op-key", "robot-key"
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
def stack():
    backend_port, gw_port = _free_port(), _free_port()
    backend = subprocess.Popen(
        [sys.executable, str(MOCK_SERVER), "--port", str(backend_port), "--latency-ms", "10", "--jitter-ms", "0"]
        + ["--idle-motion", "0.02", "--max-batch", "4"],
        cwd=MOCK_SERVER.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    backend_url = f"ws://127.0.0.1:{backend_port}/v1/realtime/robot/openpi"
    config = {
        "api_keys": [OPERATOR],
        "models": {
            "mock": {"backends": [backend_url], "adapter": ADAPTER},
            "native_only": {"backends": [backend_url]},
        },
        "robots": {"piper-01": {"api_key": ROBOT_KEY, "models": ["mock"]}},
    }
    gateway = Gateway(config)
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
    yield {"gw": f"ws://127.0.0.1:{gw_port}", "http": f"http://127.0.0.1:{gw_port}", "gateway": gateway}
    loop.call_soon_threadsafe(stop.set)
    thread.join(5)
    backend.terminate()
    backend.wait(5)


def _get(stack, path, key=OPERATOR):
    req = urllib.request.Request(stack["http"] + path, headers={"Authorization": f"Api-Key {key}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _hello(robot_id="piper-01", **kw):
    base = dict(
        robot_id=robot_id,
        embodiment="piper",
        dof=6,
        images=["front", "wrist"],
        state=["joint_pos", "joint_vel", "gripper"],
        action_spaces=["joint_pos"],
    )
    return RobotHello(**{**base, **kw})


def _obs(seq=1):
    return {
        "type": "obs",
        "seq": seq,
        "t": 12.5,
        "instruction": "wave",
        "images": {"front": np.zeros((64, 64, 3), np.uint8), "wrist": np.zeros((64, 64, 3), np.uint8)},
        "state": {"joint_pos": np.full(6, 0.5, np.float32), "gripper": np.array([1.0], np.float32)},
    }


def test_access_control(stack):
    def status(model, robot_id, key):
        url = act_url(stack["gw"], model, robot_id) if robot_id else f"{stack['gw']}/v1/robot/{model}/act"
        with pytest.raises(InvalidStatus) as exc:
            with connect(url, additional_headers={"Authorization": f"Api-Key {key}"}):
                pass
        return exc.value.response.status_code

    assert status("mock", "piper-01", "wrong") == 401
    assert status("mock", "other-robot", ROBOT_KEY) == 401  # a robot key is bound to its robot_id
    assert status("native_only", "piper-01", OPERATOR) == 404  # no adapter, no /act
    assert status("mock", None, OPERATOR) == 400
    assert _get(stack, "/v1/robots", ROBOT_KEY)[0] == 401  # robot keys are not operator keys
    code, info = _get(stack, "/v1/models/mock", ROBOT_KEY)
    assert code == 200 and info["requires"]["state"] == ["joint_pos", "gripper"]


def test_handshake_infer_stop_and_listing(stack):
    client = PolicyClient(stack["gw"], "mock", "piper-01", ROBOT_KEY, embodiment="piper")
    server = client.connect(_hello())
    try:
        assert server.requires.images == {"front": [64, 64], "wrist": [64, 64]}
        assert (server.provides.space, server.provides.horizon, server.provides.control_hz) == ("joint_pos", 20, 50.0)
        reply = client.request(_obs())
        assert reply["type"] == "actions" and reply["seq"] == 1 and reply["t_obs"] == 12.5
        joints, grip = reply["actions"]["joint_pos"], reply["actions"]["gripper"]
        assert joints.shape == (20, 6) and grip.shape == (20, 1)
        # The mock returns small delta sweeps on joints 0-2; targets are absolute around the observed 0.5.
        assert np.all(np.abs(joints - 0.5) <= 0.02 + 1e-6)
        assert reply["server_ms"] > 0
        with pytest.raises(ProtocolError) as exc:
            client.request({**_obs(), "state": {"joint_pos": np.zeros(7, np.float32), "gripper": np.ones(1)}})
        assert exc.value.code == "bad_obs"
        _, robots = _get(stack, "/v1/robots")
        assert [r["robot_id"] for r in robots["robots"]] == ["piper-01"]
        assert _get(stack, "/v1/robots/piper-01/stop?reason=test")[0] == 200
        assert client.request(_obs(2)) == {"type": "stop", "reason": "test"}
        assert client.request(_obs(3))["type"] == "actions"  # stop is one-shot
        assert client.request({"type": "episode_result", "success": True}) == {"type": "ok"}
        assert client.request({"type": "reset"}) == {"type": "ok"}
        # A second session for the same robot is refused.
        dup = PolicyClient(stack["gw"], "mock", "piper-01", ROBOT_KEY)
        with pytest.raises(ConnectionClosed) as closed:
            dup.connect(_hello())
        assert closed.value.rcvd.code == 4409
    finally:
        client.close()
    time.sleep(0.1)
    assert _get(stack, "/v1/robots")[1]["robots"] == []
    assert _get(stack, "/metrics")[1]["mock"]["episodes"] == 1


def test_incompatible_robot_is_refused(stack):
    client = PolicyClient(stack["gw"], "mock", "piper-01", ROBOT_KEY)
    with pytest.raises(ProtocolError) as exc:
        client.connect(_hello(images=["front"], action_spaces=["eef_pose"]))
    assert exc.value.code == "incompatible"
    assert any("wrist" in p for p in exc.value.problems)
    assert any("outputs 'joint_pos'" in p for p in exc.value.problems)


def test_native_openpi_relay_unchanged(stack):
    url = f"{stack['gw']}/v1/robot/native_only/openpi"
    with connect(url, additional_headers={"Authorization": f"Api-Key {OPERATOR}"}) as ws:
        metadata = unpackb(ws.recv(timeout=5))
        assert metadata["model"] == "native_only" and metadata["mock"] is True
        ws.send(packb({"observation.images.image": np.zeros((64, 64, 3), np.uint8), "prompt": "x"}))
        assert unpackb(ws.recv(timeout=5)).shape == (50, 32)


def test_robotctl_runs_mock_robot(stack, tmp_path):
    robot = {
        "robot_id": "piper-01",
        "gateway": {"url": stack["http"], "model": "mock", "api_key": ROBOT_KEY},
        "driver": {"kind": "mock", "dof": 6, "embodiment": "piper"},
        "cameras": {"front": {"kind": "mock", "fps": 100}, "wrist": {"kind": "mock", "fps": 100, "seed": 1}},
        "control": {"prefetch_threshold": 5, "blend": "linear", "blend_steps": 2, "jpeg_quality": 80},
        "safety": {"start_mode": "dry_run"},
    }
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(robot))
    assert robotctl.main(["check", "--robot", str(path)]) == 0
    out = tmp_path / "episodes.jsonl"
    args = ["run", "--robot", str(path), "--instruction", "wave", "--max-steps", "60", "--mode", "live", "--yes"]
    assert robotctl.main([*args, "--output", str(out)]) == 0
    line = json.loads(out.read_text())
    assert line["stopped"] is None and line["commands_sent"] > 30 and line["inferences"] >= 2
    assert line["network_p50_ms"] is not None and line["safety"]["worst_streak"] < 5


@contextlib.contextmanager
def _monitor(stack, key=OPERATOR):
    with connect(f"{stack['gw']}/v1/monitor") as ws:
        ws.send(json.dumps({"type": "auth", "key": key}))
        yield ws


def _next(ws, kind, timeout=5.0, where=lambda msg: True):
    deadline = time.time() + timeout
    while True:
        msg = json.loads(ws.recv(timeout=max(0.1, deadline - time.time())))
        if msg["type"] == kind and where(msg):
            return msg


def _jpeg_size(b64):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def test_monitor_requires_operator_key(stack):
    for key in ("wrong", ROBOT_KEY):
        with _monitor(stack, key) as ws, pytest.raises(ConnectionClosed) as closed:
            ws.recv(timeout=5)
        assert closed.value.rcvd.code == 4401


def test_monitor_lists_watches_and_stops_act_session(stack):
    client = PolicyClient(stack["gw"], "mock", "piper-01", ROBOT_KEY, embodiment="piper")
    client.connect(_hello())
    try:
        assert client.request(_obs())["type"] == "actions"
        with _monitor(stack) as ws:
            hello = _next(ws, "hello")
            assert len(hello["models"]["mock"]) > 0  # zero-filled one-second history
            snap = _next(ws, "snapshot", where=lambda m: any(s["robot_id"] == "piper-01" for s in m["sessions"]))
            row = next(s for s in snap["sessions"] if s["robot_id"] == "piper-01")
            assert (row["kind"], row["status"], row["instruction"], row["requests"]) == ("act", "running", "wave", 1)
            assert row["embodiment"] == "piper" and row["can_stop"] and row["peer"].startswith("127.0.0.1")
            assert snap["models"]["mock"]["backends"][0]["active"] >= 1
            ws.send(json.dumps({"type": "watch", "session_id": row["session_id"], "fps": 10}))
            detail = _next(ws, "detail", where=lambda m: "frames" in m)
            assert {role: _jpeg_size(b64) for role, b64 in detail["frames"].items()} == {
                "front": (64, 64),
                "wrist": (64, 64),
            }
            assert detail["state"]["joint_pos"] == [0.5] * 6
            assert len(detail["actions"]["joint_pos"]) == 20 and len(detail["actions"]["joint_pos"][0]) == 6
            assert [e["kind"] for e in detail["events"]][:2] == ["connected", "ready"]
            ws.send(json.dumps({"type": "stop", "session_id": row["session_id"], "reason": "dashboard"}))
            assert _next(ws, "stop_ack")["ok"] is True
            assert client.request(_obs(2)) == {"type": "stop", "reason": "dashboard"}
    finally:
        client.close()
    with _monitor(stack) as ws:
        snap = _next(ws, "snapshot", where=lambda m: not any(s["robot_id"] == "piper-01" for s in m["sessions"]))
        closed = next(s for s in snap["closed"] if s["robot_id"] == "piper-01")
        assert closed["status"] == "closed" and closed["close_reason"] == "robot disconnected"
        assert not closed["can_stop"]


def test_monitor_watches_native_openpi_session(stack):
    url = f"{stack['gw']}/v1/robot/native_only/openpi"
    image = np.full((48, 64, 3), 200, np.uint8)
    with connect(url, additional_headers={"Authorization": f"Api-Key {OPERATOR}"}) as relay, _monitor(stack) as ws:
        unpackb(relay.recv(timeout=5))
        relay.send(packb({"observation.images.image": image, "observation.state": np.ones(8), "prompt": "stack"}))
        relay.recv(timeout=5)
        snap = _next(ws, "snapshot", where=lambda m: any(s["model"] == "native_only" for s in m["sessions"]))
        row = next(s for s in snap["sessions"] if s["model"] == "native_only")
        assert row["kind"] == "openpi" and row["label"].startswith("openpi-") and not row["can_stop"]
        ws.send(json.dumps({"type": "watch", "session_id": row["session_id"]}))
        detail = _next(ws, "detail", where=lambda m: "frames" in m)
        assert _jpeg_size(detail["frames"]["observation.images.image"]) == (64, 48)
        assert detail["summary"]["instruction"] == "stack"
        assert detail["state"]["observation.state"] == [1.0] * 8
        assert len(detail["actions"]["actions"]) == 50
        ws.send(json.dumps({"type": "stop", "session_id": row["session_id"]}))
        assert _next(ws, "stop_ack")["ok"] is False
