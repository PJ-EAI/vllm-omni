# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-episode MP4 recording for simulated robots, headless.

Frames come from the simulator's offscreen cameras (EGL on GPU servers), so no
display is needed. Each frame gets a caption bar with the robot, model, step,
last inference latency and, at the end, the episode outcome. Encoding uses
``imageio`` with the ffmpeg binary bundled by ``imageio-ffmpeg``; captions use
Pillow when it is installed and are skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CAPTION_HEIGHT = 18


def _caption_drawer():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    font = ImageFont.load_default()

    def draw(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
        bar = Image.new("RGB", (frame.shape[1], CAPTION_HEIGHT), (20, 20, 20))
        ImageDraw.Draw(bar).text((4, 3), text, fill=color, font=font)
        return np.concatenate((np.asarray(bar), frame), axis=0)

    return draw


def _even(frame: np.ndarray) -> np.ndarray:
    # yuv420p (the pixel format browsers play) needs even width and height.
    pad_h, pad_w = frame.shape[0] % 2, frame.shape[1] % 2
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)))
    return frame


class EpisodeRecorder:
    """Streams one episode to ``<out_dir>/<name>.mp4``; ``finish`` names it by outcome.

    Frames go straight to the encoder, so memory stays flat however long the
    episode and however many robots record at once.
    """

    def __init__(self, out_dir: Path, fps: float, stride: int = 1) -> None:
        import imageio  # noqa: F401  # fail at start-up, not after the first episode

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, stride)
        self.fps = max(1.0, fps / self.stride)
        self._draw = _caption_drawer()
        self._writer = None
        self._tmp: Path | None = None
        self._last: np.ndarray | None = None
        self._header = ""
        self._frames = 0
        self._step = 0

    def start(self, header: str, name: str) -> None:
        import imageio

        self.abort()
        self._header = header
        self._frames = 0
        self._step = 0
        self._last = None
        self._tmp = self.out_dir / f".{name}.partial.mp4"
        self._writer = imageio.get_writer(
            self._tmp, fps=self.fps, codec="libx264", macro_block_size=1, pixelformat="yuv420p", quality=7
        )

    def add(self, frame: np.ndarray, step: int, latency_ms: float | None) -> None:
        """Add the frame observed after ``step`` control steps (0 = after reset)."""
        self._step = step
        self._frames += 1
        if self._writer is None or (self._frames - 1) % self.stride:
            return
        self._last = frame
        if self._draw is not None:
            lat = f"{latency_ms:.0f}ms" if latency_ms is not None else "-"
            frame = self._draw(frame, f"{self._header}  step {self._step}  infer {lat}", (230, 230, 230))
        self._writer.append_data(_even(frame))

    def finish(self, name: str, success: bool) -> Path | None:
        if self._writer is None:
            return None
        if self._draw is not None and self._last is not None:
            # Hold the outcome on screen for ~1 s at the end of the clip.
            status = "SUCCESS" if success else "FAIL"
            color = (90, 220, 90) if success else (240, 90, 90)
            end = _even(self._draw(self._last, f"{self._header}  {status} after {self._step} steps", color))
            for _ in range(int(self.fps)):
                self._writer.append_data(end)
        self._writer.close()
        self._writer = None
        path = self.out_dir / f"{name}.mp4"
        self._tmp.replace(path)
        return path

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._tmp.unlink(missing_ok=True)
