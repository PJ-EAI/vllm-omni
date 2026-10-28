# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import base64
import time

import numpy as np
from monitor import Backend, ModelStats, SessionView, render_detail
from openpi_client import msgpack_numpy
from vla_maas_protocol import encode_jpeg


def test_history_is_one_point_per_second_with_gaps_as_zero(monkeypatch):
    clock = [1000.2]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    stats = ModelStats()
    for ms in (10.0, 20.0, 30.0):
        stats.record(ms)
    stats.fail()
    clock[0] = 1003.5  # 1001 and 1002 had no traffic
    stats.record(50.0)
    points = stats.history(since=999)
    assert [p["t"] for p in points] == [1000, 1001, 1002]
    assert (points[0]["requests"], points[0]["errors"], points[0]["p50_ms"]) == (3, 1, 20.0)
    assert points[1] == {"t": 1001, "requests": 0, "errors": 0, "p50_ms": None, "p99_ms": None}
    assert stats.history(since=1002) == []  # 1003 is still open
    assert stats.snapshot()["requests"] == 0 and stats.errors == 1  # requests counts started requests


def test_counting_on_a_session_counts_on_its_model_and_backend():
    stats, backend = ModelStats(), Backend("ws://b")
    view = SessionView("act", "m", stats, backend, "1.2.3.4:5", robot_id="r1")
    view.record(12.0, {"joint_pos": np.zeros((4, 6))})
    view.fail("model_error", "boom")
    view.episode(True)
    assert (view.requests, view.errors, view.episodes) == (1, 1, 1)
    assert (backend.requests, backend.errors, stats.errors, stats.successes) == (1, 1, 1, 1)
    assert view.summary()["status"] == "connecting"
    view.ready = True
    view.observe({"instruction": "wave", "episode_id": "e1"})
    assert view.status() == "running" and view.instruction == "wave"
    view.stop_reason = "x"
    assert view.status() == "stopping"
    assert [e["kind"] for e in view.events] == ["error", "episode_result", "episode"]


def test_render_detail_passes_jpeg_through_and_encodes_raw_frames():
    view = SessionView("act", "m", ModelStats(), Backend("ws://b"), "peer", robot_id="r1")
    jpeg = encode_jpeg(np.zeros((16, 16, 3), np.uint8))
    view.observe(
        {
            "images": {"front": jpeg, "wrist": np.zeros((8, 8, 3), np.uint8)},
            "state": {"joint_pos": np.arange(6, dtype=np.float32)},
            "instruction": "pick",
        }
    )
    view.record(5.0, {"joint_pos": np.zeros((3, 6)), "gripper": np.ones((3, 1))})
    detail = render_detail(view.capture(), frames=True)

    assert base64.b64decode(detail["frames"]["front"]) == jpeg["data"]
    assert base64.b64decode(detail["frames"]["wrist"])[:2] == b"\xff\xd8"
    assert detail["state"]["joint_pos"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert detail["actions"]["gripper"] == [[1.0]] * 3
    assert "frames" not in render_detail(view.capture(), frames=False)


def test_render_detail_decodes_openpi_frames_lazily():
    view = SessionView("openpi", "m", ModelStats(), Backend("ws://b"), "peer")
    chw = np.zeros((3, 10, 12), np.uint8)  # some OpenPI clients send channel-first images
    obs = {"observation.images.image": chw, "observation.state": np.ones(4), "prompt": "go"}
    view.observe(msgpack_numpy.packb(obs))
    view.record(5.0, msgpack_numpy.packb(np.zeros((50, 7))))
    detail = render_detail(view.capture(), frames=True)
    assert set(detail["frames"]) == {"observation.images.image"}
    assert detail["summary"]["instruction"] == "go" and detail["state"]["observation.state"] == [1.0] * 4
    assert np.asarray(detail["actions"]["actions"]).shape == (50, 7)
    view.observe(b"not msgpack")
    assert render_detail(view.capture(), frames=True)["frames"] == {}
