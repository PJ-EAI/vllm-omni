# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Swapping a model's checkpoint and model type live: probe, drain, cutover, rollback, logs.

Two CPU mock backends report different checkpoints; robots on the old one keep
their session while new robots land on the new one.
"""

import asyncio
import io
import json
import logging
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import maas_log
import numpy as np
import pytest
import yaml
from gateway import Gateway
from routes import RouteSpec, check_reported, normalize_model_type
from vla_maas_protocol import RobotHello
from vla_maas_robot.client import PolicyClient, act_url
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus
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
    "action": {"space": "joint_pos", "layout": [{"field": "joint_pos", "mode": "delta"}, "gripper"]},
    "control_hz": 50,
    "horizon": 20,
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mock(checkpoint: str, model_type: str) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(MOCK_SERVER), "--port", str(port), "--latency-ms", "5", "--jitter-ms", "0"]
        + ["--checkpoint", checkpoint, "--model-type", model_type],
        cwd=MOCK_SERVER.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc, f"ws://127.0.0.1:{port}/v1/realtime/robot/openpi"


def _wait_up(url: str) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with connect(url, open_timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not start")


@pytest.fixture
def swap(tmp_path):
    procs = [_mock("ckpt-a", "Pi05Pipeline"), _mock("ckpt-b", "InternVLA_A1Pipeline")]
    for _, url in procs:
        _wait_up(url)
    (url_a, url_b), config_path = [url for _, url in procs], tmp_path / "gateway.yaml"

    def write(backends, checkpoint, model_type, **extra):
        route = {"backends": backends, "adapter": ADAPTER, "checkpoint": checkpoint, "model_type": model_type}
        config = {
            "api_keys": [OPERATOR],
            "models": {"piper": route, **extra},
            "robots": {f"r{i}": {"api_key": ROBOT_KEY} for i in range(1, 4)},
        }
        config_path.write_text(yaml.safe_dump(config))

    write([url_a], "ckpt-a", "pi05")
    lines = io.StringIO()
    handler = logging.StreamHandler(lines)
    handler.setFormatter(maas_log.JsonFormatter())
    log = logging.getLogger("vla_maas_gateway")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    gateway = Gateway(yaml.safe_load(config_path.read_text()), config_path=config_path)
    gw_port, loop = _free_port(), asyncio.new_event_loop()
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
    yield {
        "gw": f"ws://127.0.0.1:{gw_port}",
        "http": f"http://127.0.0.1:{gw_port}",
        "gateway": gateway,
        "a": url_a,
        "b": url_b,
        "write": write,
        "logs": lambda: [json.loads(line) for line in lines.getvalue().splitlines()],
    }
    loop.call_soon_threadsafe(stop.set)
    thread.join(5)
    log.removeHandler(handler)
    for proc, _ in procs:
        proc.terminate()
        proc.wait(5)


def _get(stack, path, key=OPERATOR):
    req = urllib.request.Request(stack["http"] + path, headers={"Authorization": f"Api-Key {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _robot(stack, robot_id) -> PolicyClient:
    client = PolicyClient(stack["gw"], "piper", robot_id, ROBOT_KEY)
    hello = RobotHello(
        robot_id=robot_id,
        embodiment="piper",
        dof=6,
        images=["front", "wrist"],
        state=["joint_pos", "gripper"],
        action_spaces=["joint_pos"],
    )
    client.connect(hello)
    return client


def _obs(seq=1):
    return {
        "type": "obs",
        "seq": seq,
        "t": 1.0,
        "episode_id": "ep-1",
        "instruction": "wave",
        "images": {"front": np.zeros((64, 64, 3), np.uint8), "wrist": np.zeros((64, 64, 3), np.uint8)},
        "state": {"joint_pos": np.zeros(6, np.float32), "gripper": np.ones(1, np.float32)},
    }


def _until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_swap_drains_old_sessions_and_routes_new_ones(swap):
    old = _robot(swap, "r1")
    assert old.ready["checkpoint"] == "ckpt-a" and old.ready["revision"].startswith("r1-")
    assert old.request(_obs())["type"] == "actions"

    swap["write"]([swap["b"]], "ckpt-b", "internvla_a1")
    code, body = _get(swap, "/v1/admin/reload")
    assert code == 200, body
    (change,) = body["changes"]
    assert change["from"]["checkpoint"] == "ckpt-a" and change["to"]["checkpoint"] == "ckpt-b"
    assert change["to"]["model_type"] == "internvla_a1" and change["draining_sessions"] == 1
    _, models = _get(swap, "/v1/models")
    assert models["models"][0]["revision"].startswith("r2-") and models["models"][0]["draining"] == 1

    new = _robot(swap, "r2")
    try:
        assert new.ready["checkpoint"] == "ckpt-b" and new.ready["revision"] == change["to"]["revision"]
        assert new.request(_obs())["type"] == "actions"
        assert old.request(_obs(2))["type"] == "actions"  # still served by the old backend
        _, robots = _get(swap, "/v1/robots")
        assert {r["robot_id"]: r["checkpoint"] for r in robots["robots"]} == {"r1": "ckpt-a", "r2": "ckpt-b"}
    finally:
        new.close()
        old.close()
    assert _until(lambda: _get(swap, "/v1/models")[1]["models"][0]["draining"] == 0)
    _, revisions = _get(swap, "/v1/admin/revisions")
    assert revisions["history"][0]["to"]["checkpoint"] == "ckpt-b" and revisions["draining"]["piper"] == []

    logs = swap["logs"]()
    r1 = [line for line in logs if line.get("session") == old.session_id]
    assert {line["event"] for line in r1} >= {"session.open", "session.ready", "act.request", "session.close"}
    assert {(line["checkpoint"], line["revision"]) for line in r1} == {("ckpt-a", old.ready["revision"])}
    assert all(line["robot"] == "r1" and line["backend"] == swap["a"] for line in r1)
    close = next(line for line in r1 if line["event"] == "session.close")
    assert close["requests"] == 2 and close["errors"] == 0
    (rev_line,) = [line for line in logs if line["event"] == "route.revision"]
    assert rev_line["previous_checkpoint"] == "ckpt-a" and rev_line["checkpoint"] == "ckpt-b"
    assert any(line["event"] == "backend.drained" and line["backend"] == swap["a"] for line in logs)


def test_failed_probe_changes_nothing(swap):
    before = _get(swap, "/v1/models")[1]
    # The config says ckpt-c but the backend serves ckpt-b.
    swap["write"]([swap["b"]], "ckpt-c", "internvla_a1")
    code, body = _get(swap, "/v1/admin/reload")
    assert code == 409 and "ckpt-b" in body["problems"]["piper"][0]
    # A backend that is not up.
    swap["write"]([f"ws://127.0.0.1:{_free_port()}/v1/realtime/robot/openpi"], "ckpt-b", "internvla_a1")
    code, body = _get(swap, "/v1/admin/reload")
    assert code == 409 and "probe failed" in body["problems"]["piper"][0]
    # Model type mismatch.
    swap["write"]([swap["b"]], "ckpt-b", "pi05")
    assert _get(swap, "/v1/admin/reload")[0] == 409
    # A broken file.
    swap["write"]([swap["b"]], "ckpt-b", "internvla_a1", ghost={"no_backends": True})
    assert _get(swap, "/v1/admin/reload")[0] == 400
    assert _get(swap, "/v1/models")[1] == before
    client = _robot(swap, "r1")
    try:
        assert client.ready["checkpoint"] == "ckpt-a"
    finally:
        client.close()
    rejected = [line for line in swap["logs"]() if line["event"] == "route.reload_rejected"]
    assert len(rejected) == 4


def test_reload_needs_operator_key_and_is_idempotent(swap):
    assert _get(swap, "/v1/admin/reload", ROBOT_KEY)[0] == 401
    code, body = _get(swap, "/v1/admin/reload")
    assert code == 200 and body["status"] == "unchanged"


def test_cutover_stop_and_paused_route(swap):
    old = _robot(swap, "r1")
    try:
        swap["write"]([swap["b"]], "ckpt-b", None)
        code, body = _get(swap, "/v1/admin/reload?cutover=stop")
        assert code == 200 and body["stopped_sessions"] == 1
        # The backend reports its model type when the label is left out.
        assert body["changes"][0]["to"]["model_type"] == "InternVLA_A1Pipeline"
        reply = old.request(_obs())
        assert reply["type"] == "stop" and "swapped" in reply["reason"]
    finally:
        old.close()
    # backends: [] pauses the route (a swap that frees the GPUs first).
    swap["write"]([], "ckpt-b", None)
    assert _get(swap, "/v1/admin/reload")[0] == 200
    with pytest.raises(InvalidStatus) as exc:
        with connect(act_url(swap["gw"], "piper", "r3"), additional_headers={"Authorization": f"Api-Key {OPERATOR}"}):
            pass
    assert exc.value.response.status_code == 503
    swap["write"]([swap["a"]], "ckpt-a", "pi05")
    assert _get(swap, "/v1/admin/reload")[0] == 200
    client = _robot(swap, "r3")
    try:
        assert client.ready["checkpoint"] == "ckpt-a" and client.ready["revision"].startswith("r4-")
    finally:
        client.close()


def test_reported_identity_checks():
    spec = RouteSpec(backends=("ws://x",), checkpoint="a", model_type="pi05")
    assert check_reported(spec, "ws://x", {"server": {"checkpoint": "a", "model_type": "Pi05Pipeline"}})[1] == []
    assert check_reported(spec, "ws://x", {})[1] == []  # backends that report nothing are taken on the label
    assert len(check_reported(spec, "ws://x", {"server": {"checkpoint": "b"}})[1]) == 1
    assert normalize_model_type("InternVLA_A1Pipeline") == normalize_model_type("internvla-a1") == "internvlaa1"
