"""
Unit tests for src/inference/normalize.py.

Each test targets one specific per-mode rule from
scripts/normalize_dataset.py::normalize_to_rgb that this module mirrors,
plus the full re-encode-as-JPEG step. Pure PIL/NumPy, no model/torch
involved.
"""

import numpy as np
import pytest
from PIL import Image

from src.inference.normalize import normalize_and_reencode_jpeg, normalize_to_rgb


def _rgb_image(width=40, height=30, color=(200, 100, 50)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


class TestNormalizeToRGB:
    def test_plain_rgb_passes_through_unchanged(self):
        img = _rgb_image()
        result = normalize_to_rgb(img)
        assert result.mode == "RGB"
        assert result.size == img.size
        assert np.array_equal(np.asarray(result), np.asarray(img))

    def test_grayscale_converts_to_rgb(self):
        img = Image.new("L", (20, 20), 128)
        result = normalize_to_rgb(img)
        assert result.mode == "RGB"
        arr = np.asarray(result)
        assert (arr == 128).all()

    def test_rgba_composites_onto_white_background(self):
        # Fully transparent red pixel should disappear into white; fully
        # opaque red pixel should stay red.
        img = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
        img.putpixel((1, 0), (255, 0, 0, 255))
        result = normalize_to_rgb(img)
        assert result.mode == "RGB"
        arr = np.asarray(result)
        assert tuple(arr[0, 0]) == (255, 255, 255)  # transparent -> white
        assert tuple(arr[0, 1]) == (255, 0, 0)  # opaque -> unchanged

    def test_palette_with_transparency_composites_onto_white(self):
        img = Image.new("P", (2, 1))
        img.putpixel((0, 0), 0)
        img.putpixel((1, 0), 1)
        img.putpalette([255, 255, 255, 255, 0, 0])  # index 0 = white, index 1 = red
        img.info["transparency"] = 0  # index 0 is the transparent color
        result = normalize_to_rgb(img)
        assert result.mode == "RGB"

    def test_16bit_grayscale_rescales_not_reinterprets(self):
        # A uniform 16-bit value of 65535 (max) should become 255 after
        # >> 8, not near-zero (the classic .convert("RGB") bug this
        # function exists to avoid).
        arr16 = np.full((10, 10), 65535, dtype=np.int32)
        img = Image.fromarray(arr16, mode="I")
        result = normalize_to_rgb(img)
        assert result.mode == "RGB"
        out_arr = np.asarray(result)
        assert out_arr.min() >= 250  # should be ~255, not ~0

    def test_16bit_grayscale_midrange_value(self):
        arr16 = np.full((10, 10), 32768, dtype=np.int32)  # ~half of 65535
        img = Image.fromarray(arr16, mode="I")
        result = normalize_to_rgb(img)
        out_arr = np.asarray(result)
        # 32768 >> 8 == 128
        assert out_arr[0, 0, 0] == 128

    def test_float_mode_min_max_stretched(self):
        arr = np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32)
        img = Image.fromarray(arr, mode="F")
        result = normalize_to_rgb(img)
        assert result.mode == "RGB"
        out_arr = np.asarray(result)
        assert out_arr.min() == 0
        assert out_arr.max() == 255

    def test_float_mode_constant_value_does_not_divide_by_zero(self):
        arr = np.full((5, 5), 3.0, dtype=np.float32)
        img = Image.fromarray(arr, mode="F")
        result = normalize_to_rgb(img)  # should not raise ZeroDivisionError
        assert np.asarray(result).max() == 0

    def test_fix_orientation_false_leaves_pixels_alone(self):
        img = _rgb_image(20, 10)
        exif = img.getexif()
        exif[0x0112] = 6  # "Rotate 90 CW" orientation tag
        img.info["exif"] = exif.tobytes()
        result = normalize_to_rgb(img, fix_orientation=False)
        assert result.size == (20, 10)  # unchanged, no rotation applied

    def test_fix_orientation_true_applies_exif_rotation(self):
        img = _rgb_image(20, 10)
        exif = img.getexif()
        exif[0x0112] = 6  # 90-degree rotation swaps width/height
        img.info["exif"] = exif.tobytes()
        result = normalize_to_rgb(img, fix_orientation=True)
        assert result.size == (10, 20)


class TestNormalizeAndReencodeJPEG:
    def test_returns_rgb_same_size(self):
        img = _rgb_image(50, 40)
        result = normalize_and_reencode_jpeg(img, quality=90)
        assert result.mode == "RGB"
        assert result.size == (50, 40)

    def test_applies_jpeg_compression(self):
        # A noisy image re-saved through JPEG should differ at least
        # slightly from the original (compression is lossy).
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result = normalize_and_reencode_jpeg(img, quality=90)
        assert not np.array_equal(np.asarray(result), arr)

    def test_handles_non_rgb_input_end_to_end(self):
        img = Image.new("RGBA", (30, 30), (10, 20, 30, 128))
        result = normalize_and_reencode_jpeg(img, quality=85)
        assert result.mode == "RGB"
        assert result.size == (30, 30)

    def test_lower_quality_increases_compression_error(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 256, size=(60, 60, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")

        high_q = np.asarray(normalize_and_reencode_jpeg(img, quality=95), dtype=np.int16)
        low_q = np.asarray(normalize_and_reencode_jpeg(img, quality=10), dtype=np.int16)
        orig = arr.astype(np.int16)

        high_q_error = np.abs(high_q - orig).mean()
        low_q_error = np.abs(low_q - orig).mean()
        assert low_q_error > high_q_error
