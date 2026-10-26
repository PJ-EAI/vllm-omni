#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tile episode MP4s from ``sim_fleet.py`` into one grid video.

Clips play side by side from the start; a clip that ends holds its last frame
until the longest one finishes. Use it to watch every arm of a run at once.

Usage:
    python video_grid.py videos/ --output videos/grid.mp4 --cols 4 --max-clips 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def find_clips(video_dir: Path, max_clips: int) -> list[Path]:
    clips = sorted(p for p in Path(video_dir).rglob("*.mp4") if not p.name.startswith(".") and p.name != "grid.mp4")
    # One clip per group first, so a capped grid still shows every arm type.
    by_group: dict[Path, list[Path]] = {}
    for clip in clips:
        by_group.setdefault(clip.parent, []).append(clip)
    ordered: list[Path] = []
    while any(by_group.values()) and len(ordered) < max_clips:
        for group in list(by_group):
            if by_group[group] and len(ordered) < max_clips:
                ordered.append(by_group[group].pop(0))
    return ordered


def make_grid(clips: list[Path], output: Path, cols: int, fps: float) -> Path:
    import imageio

    if not clips:
        raise ValueError("no clips to tile")
    readers = [imageio.get_reader(c) for c in clips]
    iters = [iter(r) for r in readers]
    last: list[np.ndarray] = [next(it) for it in iters]
    h = max(f.shape[0] for f in last)
    w = max(f.shape[1] for f in last)
    cols = max(1, min(cols, len(clips)))
    rows = -(-len(clips) // cols)
    # yuv420p needs even dimensions.
    canvas_h, canvas_w = rows * h + (rows * h) % 2, cols * w + (cols * w) % 2
    live = [True] * len(clips)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", macro_block_size=1, pixelformat="yuv420p", quality=7)
    try:
        while True:
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for i, frame in enumerate(last):
                r, c = divmod(i, cols)
                canvas[r * h : r * h + frame.shape[0], c * w : c * w + frame.shape[1]] = frame
            writer.append_data(canvas)
            for i, it in enumerate(iters):
                if live[i]:
                    try:
                        last[i] = next(it)
                    except StopIteration:
                        live[i] = False
            if not any(live):
                break
    finally:
        writer.close()
        for r in readers:
            r.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Tile sim_fleet episode videos into one grid MP4")
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Default: <video_dir>/grid.mp4")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    clips = find_clips(args.video_dir, args.max_clips)
    out = make_grid(clips, args.output or args.video_dir / "grid.mp4", args.cols, args.fps)
    print(f"{len(clips)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
