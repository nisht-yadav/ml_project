"""
Unit tests for src/inference/predictor.py.

Builds a tiny, randomly-initialized TamperResNet50 checkpoint on the fly
(pretrained=False, so no network access or ImageNet download is needed)
instead of depending on any real trained checkpoint - these tests only
exercise the checkpoint-loading / single-image-inference machinery
itself, independent of model quality or the training pipeline.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.predictor import (
    ImageDecodeError,
    ModelLoadError,
    load_image,
    load_model,
    predict_image,
)
from src.models.tamper_resnet50 import TamperResNet50


@pytest.fixture(scope="module")
def fake_checkpoint_path(tmp_path_factory):
    model = TamperResNet50(freeze_stages=[], freeze_bn=False, dropout=0.5, pretrained=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "TamperResNet50",
        "model_init_kwargs": {"freeze_stages": [], "freeze_bn": False, "dropout": 0.5},
        "img_size": 224,
        "inference_threshold": 0.5,
        "class_names": ["authentic", "tampered"],
    }
    path = tmp_path_factory.mktemp("ckpt") / "fake_model.pth"
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestLoadModel:
    def test_loads_successfully(self, fake_checkpoint_path):
        device = torch.device("cpu")
        model, checkpoint = load_model(fake_checkpoint_path, device)
        assert isinstance(model, TamperResNet50)
        assert model.training is False  # eval mode
        assert checkpoint["img_size"] == 224

    def test_missing_file_raises_model_load_error(self):
        with pytest.raises(ModelLoadError):
            load_model("/nonexistent/path/model.pth", torch.device("cpu"))

    def test_corrupt_file_raises_model_load_error(self, tmp_path):
        bad_path = tmp_path / "bad.pth"
        bad_path.write_bytes(b"not a real checkpoint")
        with pytest.raises(ModelLoadError):
            load_model(bad_path, torch.device("cpu"))


class TestLoadImage:
    def test_accepts_pil_image(self, sample_image):
        result = load_image(sample_image)
        assert result.mode == "RGB"

    def test_accepts_bytes(self, sample_image, tmp_path):
        import io
        buf = io.BytesIO()
        sample_image.save(buf, format="PNG")
        result = load_image(buf.getvalue())
        assert result.size == sample_image.size

    def test_corrupted_bytes_raise_image_decode_error(self):
        with pytest.raises(ImageDecodeError):
            load_image(b"definitely not an image")


class TestPredictImage:
    def test_returns_expected_keys(self, fake_checkpoint_path, sample_image):
        device = torch.device("cpu")
        model, checkpoint = load_model(fake_checkpoint_path, device)
        result = predict_image(model, sample_image, device, img_size=checkpoint["img_size"])
        assert result["label"] in ("authentic", "tampered")
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["prob_tampered"] <= 1.0
        assert result["input_tensor"].shape == (1, 3, 224, 224)
        assert result["display_image"].size == (224, 224)

    def test_confidence_is_probability_of_predicted_label(self, fake_checkpoint_path, sample_image):
        device = torch.device("cpu")
        model, checkpoint = load_model(fake_checkpoint_path, device)
        result = predict_image(model, sample_image, device, img_size=checkpoint["img_size"])
        if result["label"] == "tampered":
            assert result["confidence"] == pytest.approx(result["prob_tampered"])
        else:
            assert result["confidence"] == pytest.approx(1.0 - result["prob_tampered"])

    def test_threshold_changes_label_without_changing_probability(self, fake_checkpoint_path, sample_image):
        device = torch.device("cpu")
        model, checkpoint = load_model(fake_checkpoint_path, device)
        low = predict_image(model, sample_image, device, img_size=224, threshold=0.0)
        high = predict_image(model, sample_image, device, img_size=224, threshold=1.0)
        assert low["prob_tampered"] == pytest.approx(high["prob_tampered"])
        assert low["label"] == "tampered"  # threshold 0.0 -> everything is "tampered"
        assert high["label"] == "authentic"  # threshold 1.0 -> everything is "authentic"
