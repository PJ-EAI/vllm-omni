# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Image values on the wire: raw ``uint8[H, W, 3]`` arrays or ``{"enc": "jpeg"}`` maps.

JPEG needs Pillow (the ``jpeg`` extra); raw arrays need nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_jpeg(image: np.ndarray, quality: int = 90) -> dict:
    """Compress an ``uint8[H, W, 3]`` image for the wire (needs Pillow)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return {"enc": "jpeg", "data": buf.getvalue()}


def decode_image(value: Any) -> np.ndarray:
    """Raw arrays pass through; ``{"enc": "jpeg"}`` maps are decoded (needs Pillow)."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict) and value.get("enc") == "jpeg":
        import io

        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(value["data"])).convert("RGB"))
    raise ValueError(f"unsupported image value: {type(value).__name__}")


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Keep the aspect ratio and pad with black, the OpenPI client convention."""
    h, w = image.shape[:2]
    if (h, w) == (height, width):
        return image
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    try:
        from PIL import Image

        resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
    except ImportError:  # nearest neighbour without Pillow
        rows = (np.arange(nh) * h / nh).astype(int)
        cols = (np.arange(nw) * w / nw).astype(int)
        resized = image[rows][:, cols]
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    top, left = (height - nh) // 2, (width - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out
