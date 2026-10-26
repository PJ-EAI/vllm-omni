# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cameras, each grabbed on its own thread so a read returns the latest frame.

Config keys are the camera's protocol role (``front``, ``wrist`` ...); see
``vla_maas_protocol.IMAGE_ROLES``. Vendor libraries load lazily:
``uvc`` needs ``opencv-python``, ``realsense`` needs ``pyrealsense2``.
"""

from __future__ import annotations

import threading
import time

import numpy as np


class Camera:
    """``grab`` blocks until the next frame; ``read`` returns the latest one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: tuple[float, np.ndarray] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: BaseException | None = None

    def open(self) -> None:
        pass

    def grab(self) -> np.ndarray:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def start(self, timeout_s: float = 5.0) -> None:
        self.open()
        self._thread = threading.Thread(target=self._run, daemon=True, name=type(self).__name__)
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while self._frame is None and time.monotonic() < deadline:
            if self.error is not None:
                raise self.error
            time.sleep(0.01)
        if self._frame is None:
            raise TimeoutError(f"{type(self).__name__}: no frame within {timeout_s}s")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.grab()
                with self._lock:
                    self._frame = (time.monotonic(), frame)
        except BaseException as exc:  # surfaced on the next read
            self.error = exc

    def read(self) -> tuple[float, np.ndarray]:
        if self.error is not None:
            raise RuntimeError(f"{type(self).__name__} failed") from self.error
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"{type(self).__name__} not started")
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.release()


class MockCamera(Camera):
    """Synthetic frames whose content changes over time, at ``fps``."""

    def __init__(self, width: int = 320, height: int = 240, fps: float = 30.0, seed: int = 0) -> None:
        super().__init__()
        self.size, self.period = (height, width), 1.0 / fps
        self.base = np.random.default_rng(seed).integers(0, 255, (*self.size, 3), dtype=np.uint8)

    def grab(self) -> np.ndarray:
        time.sleep(self.period)
        shift = int(time.monotonic() * 20) % self.size[1]
        return np.roll(self.base, shift, axis=1)


class UVCCamera(Camera):
    def __init__(self, device: int | str = 0, width: int = 640, height: int = 480, fps: float = 30.0) -> None:
        super().__init__()
        self.device, self.size, self.fps = device, (height, width), fps
        self._cap = None

    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.device!r}")

    def grab(self) -> np.ndarray:
        import cv2

        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.device!r} returned no frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class RealSenseCamera(Camera):
    def __init__(self, serial: str | None = None, width: int = 640, height: int = 480, fps: int = 30) -> None:
        super().__init__()
        self.serial, self.size, self.fps = serial, (height, width), int(fps)
        self._pipeline = None

    def open(self) -> None:
        import pyrealsense2 as rs

        config = rs.config()
        if self.serial:
            config.enable_device(str(self.serial))
        config.enable_stream(rs.stream.color, self.size[1], self.size[0], rs.format.rgb8, self.fps)
        self._pipeline = rs.pipeline()
        self._pipeline.start(config)

    def grab(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data()).copy()

    def release(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()


CAMERAS = {"mock": MockCamera, "uvc": UVCCamera, "realsense": RealSenseCamera}


def make_camera(cfg: dict) -> Camera:
    cfg = dict(cfg)
    kind = cfg.pop("kind", None)
    if kind not in CAMERAS:
        raise ValueError(f"unknown camera kind {kind!r}; available: {sorted(CAMERAS)}")
    return CAMERAS[kind](**cfg)
