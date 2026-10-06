"""
Error Level Analysis (ELA) — pure image processing, no model/training involved.

ELA re-saves an image as JPEG at a known quality level and measures the
pixel-wise difference between the original and the re-saved version.
JPEG compresses in 8x8 blocks, so a region that was already
JPEG-compressed at some point (e.g. spliced in from a different image,
or resaved after editing) reaches a different local error minimum than
its surroundings when it is compressed again. Regions with unusually
HIGH error (bright in the ELA output) after a fresh, uniform
recompression are therefore regions whose compression history is
inconsistent with the rest of the image — a common sign of localized
tampering. A perfectly uniform, single-generation JPEG tends to show a
fairly flat, low ELA response everywhere.

This module never touches a model or a training loop; it only reads
pixels in, and returns pixels (plus a few summary numbers) out, so it
can be tested and called completely independently of the CNN and of
the web app.
"""

import io
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, Image.Image, np.ndarray]

DEFAULT_JPEG_QUALITY = 90


def _to_pil(image: ImageInput) -> Image.Image:
    """Accepts a file path, an already-open PIL image, or an RGB numpy
    array and returns a PIL RGB image, so every function below can take
    whichever form is most convenient for its caller (disk path in a
    CLI/test, an in-memory upload in the web app)."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def compute_ela_diff(image: ImageInput, quality: int = DEFAULT_JPEG_QUALITY) -> np.ndarray:
    """
    Re-saves `image` as JPEG at `quality` and returns the raw, unamplified
    per-pixel absolute difference between the original and the re-saved
    version.

    Returns:
        np.ndarray, shape (H, W, 3), dtype float32, values in [0, 255].
    """
    if not (1 <= quality <= 100):
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")

    original = _to_pil(image)

    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    orig_arr = np.asarray(original, dtype=np.float32)
    resaved_arr = np.asarray(resaved, dtype=np.float32)
    return np.abs(orig_arr - resaved_arr)


def compute_ela_stats(diff: np.ndarray) -> Dict[str, float]:
    """Summary statistics of a raw ELA diff array, for logging/comparison
    across images without needing to look at the rendered heatmap."""
    return {
        "mean": float(diff.mean()),
        "max": float(diff.max()),
        "std": float(diff.std()),
    }


def _normalize_to_uint8(diff_gray: np.ndarray) -> np.ndarray:
    """Scales a single-channel diff map so its brightest pixel maps to 255,
    making the (usually very small) JPEG re-encoding error visible. Falls
    back to an all-zero image rather than dividing by zero for an
    identical-after-recompression input (e.g. a flat/blank image)."""
    max_val = diff_gray.max()
    if max_val <= 0:
        return np.zeros_like(diff_gray, dtype=np.uint8)
    scaled = diff_gray * (255.0 / max_val)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def generate_ela_heatmap(
    image: ImageInput,
    quality: int = DEFAULT_JPEG_QUALITY,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Full ELA pipeline: re-compress, diff, amplify, colorize.

    Returns:
        (heatmap, stats)
        heatmap: PIL.Image (RGB) — colorized, normalized ELA visualization,
            same size as the input image.
        stats: dict with mean/max/std of the *raw* (unamplified) diff, for
            logging/comparison independent of the visualization.
    """
    diff = compute_ela_diff(image, quality=quality)
    stats = compute_ela_stats(diff)

    diff_gray = diff.max(axis=2)  # strongest per-pixel channel error
    diff_uint8 = _normalize_to_uint8(diff_gray)

    colored_bgr = cv2.applyColorMap(diff_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(colored_rgb)

    return heatmap, stats
