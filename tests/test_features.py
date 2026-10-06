"""
Unit tests for the pure image-processing feature modules (ELA, DCT).

Deliberately model-free: every test here builds its own tiny synthetic
image with PIL/NumPy, so these tests run instantly and never touch a
checkpoint, torch, or any training code.
"""

import numpy as np
import pytest
from PIL import Image

from src.features.dct_features import (
    compute_block_dct,
    compute_dct_stats,
    compute_high_frequency_energy,
    generate_dct_heatmap,
)
from src.features.ela import compute_ela_diff, compute_ela_stats, generate_ela_heatmap


def _synthetic_image(width=64, height=48, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _flat_image(width=64, height=48, color=(120, 120, 120)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


# ---------------------------------------------------------------------------
# ELA
# ---------------------------------------------------------------------------

class TestELA:
    def test_diff_shape_matches_input(self):
        img = _synthetic_image(64, 48)
        diff = compute_ela_diff(img, quality=90)
        assert diff.shape == (48, 64, 3)

    def test_diff_is_nonnegative(self):
        img = _synthetic_image()
        diff = compute_ela_diff(img, quality=75)
        assert diff.min() >= 0.0

    def test_flat_image_has_near_zero_diff(self):
        # A uniform-color image survives JPEG re-encoding almost losslessly,
        # so its ELA response should be very low compared to noisy content.
        flat = _flat_image()
        noisy = _synthetic_image()
        flat_diff = compute_ela_diff(flat, quality=90)
        noisy_diff = compute_ela_diff(noisy, quality=90)
        assert flat_diff.mean() < noisy_diff.mean()

    def test_invalid_quality_raises(self):
        img = _synthetic_image()
        with pytest.raises(ValueError):
            compute_ela_diff(img, quality=0)
        with pytest.raises(ValueError):
            compute_ela_diff(img, quality=101)

    def test_stats_keys(self):
        diff = compute_ela_diff(_synthetic_image(), quality=90)
        stats = compute_ela_stats(diff)
        assert set(stats.keys()) == {"mean", "max", "std"}
        assert stats["max"] >= stats["mean"] >= 0.0

    def test_generate_heatmap_matches_input_size_and_mode(self):
        img = _synthetic_image(80, 60)
        heatmap, stats = generate_ela_heatmap(img, quality=90)
        assert heatmap.size == (80, 60)
        assert heatmap.mode == "RGB"
        assert set(stats.keys()) == {"mean", "max", "std"}

    def test_accepts_path_str(self, tmp_path):
        img = _synthetic_image(40, 40)
        path = tmp_path / "sample.jpg"
        img.save(path, format="JPEG", quality=95)
        heatmap, stats = generate_ela_heatmap(str(path), quality=90)
        assert heatmap.size == (40, 40)


# ---------------------------------------------------------------------------
# DCT
# ---------------------------------------------------------------------------

class TestDCT:
    def test_block_dct_shape(self):
        img = _synthetic_image(64, 48)  # exact multiple of 8
        blocks = compute_block_dct(img, block_size=8)
        assert blocks.shape == (6, 8, 8, 8)  # (n_blocks_h, n_blocks_w, 8, 8)

    def test_block_dct_pads_uneven_dimensions(self):
        img = _synthetic_image(70, 50)  # not a multiple of 8
        blocks = compute_block_dct(img, block_size=8)
        assert blocks.shape == (7, 9, 8, 8)  # ceil(50/8)=7, ceil(70/8)=9

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError):
            compute_block_dct(_synthetic_image(), block_size=1)

    def test_high_frequency_energy_shape_and_nonnegative(self):
        blocks = compute_block_dct(_synthetic_image(64, 48), block_size=8)
        energy = compute_high_frequency_energy(blocks)
        assert energy.shape == (6, 8)
        assert (energy >= 0).all()

    def test_flat_image_has_near_zero_high_freq_energy(self):
        flat_blocks = compute_block_dct(_flat_image(64, 48), block_size=8)
        noisy_blocks = compute_block_dct(_synthetic_image(64, 48), block_size=8)
        flat_energy = compute_high_frequency_energy(flat_blocks).mean()
        noisy_energy = compute_high_frequency_energy(noisy_blocks).mean()
        assert flat_energy < noisy_energy

    def test_stats_keys(self):
        blocks = compute_block_dct(_synthetic_image(), block_size=8)
        energy = compute_high_frequency_energy(blocks)
        stats = compute_dct_stats(energy)
        assert set(stats.keys()) == {
            "mean_high_freq_energy",
            "variance_high_freq_energy",
            "max_high_freq_energy",
        }

    def test_generate_heatmap_matches_input_size(self):
        img = _synthetic_image(70, 50)
        heatmap, stats = generate_dct_heatmap(img, block_size=8)
        assert heatmap.size == (70, 50)
        assert heatmap.mode == "RGB"
        assert "mean_high_freq_energy" in stats

    def test_different_block_size_still_matches_input_size(self):
        img = _synthetic_image(64, 48)
        heatmap, _ = generate_dct_heatmap(img, block_size=16)
        assert heatmap.size == (64, 48)
