"""
Block-wise Discrete Cosine Transform (DCT) analysis — pure image
processing, no model/training involved.

JPEG compression itself works by splitting an image into (usually 8x8)
blocks and DCT-transforming each one, then quantizing the result more
aggressively at high spatial frequencies (the coefficients away from the
top-left "DC" corner). Re-compressing, splicing in content from a
different source, or locally editing a region tends to leave that
region with a different amount of high-frequency energy than its
neighbors — either abnormally suppressed (from repeated recompression
smoothing out detail) or abnormally elevated (from sharp edges
introduced by pasting/blending). Mapping high-frequency energy back out
per-block turns that into a visual and numeric signal a reviewer can
compare across regions of one image, or across images.

This module never touches a model or a training loop; it only reads
pixels in, and returns pixels/numbers out, so it can be tested and
called completely independently of the CNN and of the web app.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, Image.Image, np.ndarray]

DEFAULT_BLOCK_SIZE = 8


def _to_grayscale_float(image: ImageInput) -> np.ndarray:
    """Accepts a file path, PIL image, or numpy array and returns a
    float32 single-channel (grayscale) array. DCT analysis here is done
    on luminance, matching how JPEG itself treats the Y channel."""
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("L")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("L")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("L")
    else:
        raise TypeError(f"Unsupported image input type: {type(image)!r}")
    return np.asarray(pil_image, dtype=np.float32)


def _pad_to_multiple(gray: np.ndarray, block_size: int) -> np.ndarray:
    h, w = gray.shape
    pad_h = (-h) % block_size
    pad_w = (-w) % block_size
    if pad_h or pad_w:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")
    return gray


def compute_block_dct(image: ImageInput, block_size: int = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    """
    Splits the image into non-overlapping `block_size` x `block_size`
    blocks (edge-padding the bottom/right edge if the image doesn't
    divide evenly) and computes a 2D DCT-II of each block independently.

    Returns:
        np.ndarray, shape (n_blocks_h, n_blocks_w, block_size, block_size),
        dtype float32 — the DCT coefficients of each block.
    """
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")

    gray = _pad_to_multiple(_to_grayscale_float(image), block_size)
    h, w = gray.shape
    n_h, n_w = h // block_size, w // block_size

    dct_blocks = np.empty((n_h, n_w, block_size, block_size), dtype=np.float32)
    for i in range(n_h):
        row = gray[i * block_size:(i + 1) * block_size, :]
        for j in range(n_w):
            block = row[:, j * block_size:(j + 1) * block_size]
            dct_blocks[i, j] = cv2.dct(block)

    return dct_blocks


def _high_freq_mask(block_size: int, cutoff: Optional[int] = None) -> np.ndarray:
    """
    Boolean mask over a (block_size, block_size) DCT coefficient grid
    marking the "high-frequency" coefficients: those whose horizontal +
    vertical frequency index (u + v) is >= cutoff. Coefficient (0, 0) is
    the DC term (average block brightness) and is always excluded.

    `cutoff` defaults to `block_size`, which keeps roughly the
    upper-right anti-diagonal quadrant of the coefficient grid — the
    conventional "high frequency" half of an 8x8 JPEG block.
    """
    if cutoff is None:
        cutoff = block_size
    u, v = np.meshgrid(np.arange(block_size), np.arange(block_size), indexing="ij")
    return (u + v) >= cutoff


def compute_high_frequency_energy(
    dct_blocks: np.ndarray,
    cutoff: Optional[int] = None,
) -> np.ndarray:
    """
    Per-block high-frequency energy: sum of squared high-frequency DCT
    coefficients (see `_high_freq_mask`).

    Returns:
        np.ndarray, shape (n_blocks_h, n_blocks_w), dtype float32.
    """
    block_size = dct_blocks.shape[-1]
    mask = _high_freq_mask(block_size, cutoff)
    return np.sum((dct_blocks ** 2) * mask, axis=(2, 3))


def compute_dct_stats(energy: np.ndarray) -> Dict[str, float]:
    """Summary statistics of the per-block high-frequency energy map, for
    logging/comparison across images without needing the rendered heatmap."""
    return {
        "mean_high_freq_energy": float(energy.mean()),
        "variance_high_freq_energy": float(energy.var()),
        "max_high_freq_energy": float(energy.max()),
    }


def generate_dct_heatmap(
    image: ImageInput,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cutoff: Optional[int] = None,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Full DCT analysis pipeline: block-split, DCT, high-frequency energy,
    normalize, colorize, upscale back to the original image's pixel size
    (each block's single energy value fills its original block footprint)
    so it can be shown side by side with the source image.

    Returns:
        (heatmap, stats)
        heatmap: PIL.Image (RGB), same size as the input image.
        stats: dict with mean/variance/max of the per-block high-frequency
            energy, for logging/comparison independent of the visualization.
    """
    original_size = _to_pil_size(image)
    dct_blocks = compute_block_dct(image, block_size=block_size)
    energy = compute_high_frequency_energy(dct_blocks, cutoff=cutoff)
    stats = compute_dct_stats(energy)

    # log1p compresses the typically wide dynamic range of block energies
    # (a few sharp-edge blocks can otherwise wash out every other block).
    log_energy = np.log1p(energy)
    max_val = log_energy.max()
    normalized = (
        np.zeros_like(log_energy, dtype=np.uint8)
        if max_val <= 0
        else np.clip(log_energy * (255.0 / max_val), 0, 255).astype(np.uint8)
    )

    upscaled = cv2.resize(
        normalized,
        (energy.shape[1] * block_size, energy.shape[0] * block_size),
        interpolation=cv2.INTER_NEAREST,
    )
    # Crop off any edge padding added by compute_block_dct so the heatmap
    # lines up 1:1 with the original image dimensions.
    upscaled = upscaled[: original_size[1], : original_size[0]]

    colored_bgr = cv2.applyColorMap(upscaled, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(colored_rgb)

    return heatmap, stats


def _to_pil_size(image: ImageInput) -> Tuple[int, int]:
    """(width, height) of the input, without needing a second full decode
    beyond what grayscale conversion already requires."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return img.size
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        return (w, h)
    if isinstance(image, Image.Image):
        return image.size
    raise TypeError(f"Unsupported image input type: {type(image)!r}")
