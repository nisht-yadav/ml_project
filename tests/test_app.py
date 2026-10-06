"""
Tests for the Flask web app (app/app.py) and its pipeline (app/pipeline.py).

Uses a small randomly-initialized checkpoint so these run offline and fast;
the real trained checkpoint is exercised by the manual integration run,
not by the unit suite.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.app import create_app
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


@pytest.fixture(scope="module")
def client(fake_checkpoint_path):
    app = create_app(checkpoint_path=str(fake_checkpoint_path))
    app.config["TESTING"] = True
    return app.test_client()


def _png_bytes(width=120, height=90, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, data: bytes, filename: str):
    return client.post(
        "/analyze",
        data={"image": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


class TestIndexAndHealth:
    def test_index_renders_form_and_settings(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert 'action="/analyze"' in html
        assert "ELA quality" in html
        assert "DCT block size" in html

    def test_health_reports_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_error"] is None


class TestAnalyze:
    def test_full_analysis_returns_all_panels(self, client):
        resp = _upload(client, _png_bytes(), "sample.png")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Prediction:" in html
        assert "AUTHENTIC" in html or "TAMPERED" in html
        assert html.count("data:image/png;base64,") >= 4  # original, gradcam, ela, dct
        assert "Unavailable:" not in html

    def test_jpeg_upload_is_accepted(self, client):
        buf = io.BytesIO()
        Image.fromarray(np.full((50, 50, 3), 90, dtype=np.uint8)).save(buf, format="JPEG")
        resp = _upload(client, buf.getvalue(), "sample.jpg")
        assert resp.status_code == 200
        assert "Prediction:" in resp.get_data(as_text=True)

    def test_rgba_png_upload_is_normalized_and_analyzed(self, client):
        img = Image.new("RGBA", (64, 64), (200, 30, 30, 120))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = _upload(client, buf.getvalue(), "alpha.png")
        assert resp.status_code == 200
        assert "Prediction:" in resp.get_data(as_text=True)

    def test_corrupted_upload_shows_clear_error(self, client):
        resp = _upload(client, b"this is not an image at all", "garbage.jpg")
        assert resp.status_code == 422
        assert "be read as an image" in resp.get_data(as_text=True)

    def test_missing_file_shows_error(self, client):
        resp = client.post("/analyze", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Please choose an image" in resp.get_data(as_text=True)

    def test_unsupported_extension_rejected(self, client):
        resp = _upload(client, _png_bytes(), "notes.txt")
        assert resp.status_code == 415
        assert "Unsupported file type" in resp.get_data(as_text=True)

    def test_oversized_upload_rejected(self, fake_checkpoint_path):
        app = create_app(checkpoint_path=str(fake_checkpoint_path))
        app.config["MAX_CONTENT_LENGTH"] = 1000
        resp = _upload(app.test_client(), _png_bytes(400, 400), "big.png")
        assert resp.status_code == 413
        assert "too large" in resp.get_data(as_text=True)


class TestAlternateWeights:
    def test_other_checkpoint_uses_its_own_img_size_and_threshold(self, tmp_path):
        model = TamperResNet50(freeze_stages=["conv1", "layer1"], freeze_bn=True, dropout=0.1, pretrained=False)
        path = tmp_path / "other_weights.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_init_kwargs": {"freeze_stages": ["conv1", "layer1"], "freeze_bn": True, "dropout": 0.1},
                "img_size": 128,
                "inference_threshold": 0.3,
            },
            path,
        )
        app = create_app(checkpoint_path=str(path))
        client = app.test_client()
        assert client.get("/health").get_json()["status"] == "ok"

        resp = _upload(client, _png_bytes(), "sample.png")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "decision threshold = 0.30" in html
        assert "Prediction:" in html


class TestModelLoadFailure:
    def test_bad_checkpoint_degrades_without_crashing(self, tmp_path):
        app = create_app(checkpoint_path=str(tmp_path / "missing.pth"))
        client = app.test_client()

        index = client.get("/")
        assert index.status_code == 200
        assert "Could not load the model checkpoint" in index.get_data(as_text=True)

        health = client.get("/health")
        assert health.status_code == 503
        assert health.get_json()["status"] == "degraded"

        resp = _upload(client, _png_bytes(), "sample.png")
        assert resp.status_code == 503
