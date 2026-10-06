"""
Unit tests for src/explainability/gradcam.py.

Uses a tiny, randomly-initialized TamperResNet50 (pretrained=False) so
this runs fast and offline - Grad-CAM's correctness as a *technique* is
the pytorch-grad-cam library's responsibility (it's a well-maintained
third-party dependency, per the project brief); what's tested here is
that this project's wiring (target layer, tensors, output image) works
end to end without errors.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.explainability.gradcam import GradCAMError, generate_gradcam_overlay
from src.inference.predictor import load_model, predict_image
from src.models.tamper_resnet50 import TamperResNet50


@pytest.fixture(scope="module")
def fake_checkpoint_path(tmp_path_factory):
    model = TamperResNet50(freeze_stages=[], freeze_bn=False, dropout=0.5, pretrained=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_init_kwargs": {"freeze_stages": [], "freeze_bn": False, "dropout": 0.5},
        "img_size": 224,
        "inference_threshold": 0.5,
    }
    path = tmp_path_factory.mktemp("ckpt") / "fake_model.pth"
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(150, 200, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_generate_gradcam_overlay_returns_matching_size(fake_checkpoint_path, sample_image):
    device = torch.device("cpu")
    model, checkpoint = load_model(fake_checkpoint_path, device)
    result = predict_image(model, sample_image, device, img_size=checkpoint["img_size"])

    overlay = generate_gradcam_overlay(model, result["input_tensor"], result["display_image"])

    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"
    assert overlay.size == result["display_image"].size


def test_model_still_usable_for_prediction_after_gradcam(fake_checkpoint_path, sample_image):
    # Grad-CAM enables gradients on a clone of the input tensor; this checks
    # that doesn't leave the model or a later plain prediction call broken.
    device = torch.device("cpu")
    model, checkpoint = load_model(fake_checkpoint_path, device)
    result = predict_image(model, sample_image, device, img_size=checkpoint["img_size"])
    generate_gradcam_overlay(model, result["input_tensor"], result["display_image"])

    second = predict_image(model, sample_image, device, img_size=checkpoint["img_size"])
    assert second["prob_tampered"] == pytest.approx(result["prob_tampered"], abs=1e-5)


def test_gradcam_error_on_broken_target_layer(fake_checkpoint_path, sample_image):
    device = torch.device("cpu")
    model, checkpoint = load_model(fake_checkpoint_path, device)
    result = predict_image(model, sample_image, device, img_size=checkpoint["img_size"])

    # Wrong-shaped tensor (missing batch dim handling upstream) should fail
    # loudly as GradCAMError, not crash the caller with a raw traceback.
    with pytest.raises(GradCAMError):
        generate_gradcam_overlay(model, torch.randn(3, 224, 224), result["display_image"])
