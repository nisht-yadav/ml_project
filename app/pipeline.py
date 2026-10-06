"""
End-to-end analysis of one uploaded image, independent of the web framework.

Takes raw upload bytes, returns plain Python values with every image
embedded as a PNG data URI, so the Flask routes in app/app.py stay thin
and this same pipeline can be called from a script or a test.

Order of operations for every upload:
    1. decode the bytes as an image
    2. normalize to the training-dataset format (8-bit RGB, JPEG at a
       fixed quality) - see src/inference/normalize.py
    3. CNN prediction + Grad-CAM (serialized by a lock: the model is
       shared across requests and Grad-CAM runs a backward pass)
    4. ELA and DCT, pure image processing on the normalized image

Each analysis stage is isolated: if one fails, its error is reported
next to the others instead of failing the whole request.
"""

import base64
import io
import threading
from typing import Dict, Optional

from PIL import Image

from src.explainability.gradcam import GradCAMError, generate_gradcam_overlay
from src.features.dct_features import generate_dct_heatmap
from src.features.ela import generate_ela_heatmap
from src.inference.normalize import normalize_and_reencode_jpeg
from src.inference.predictor import ImageDecodeError, load_image, load_model, predict_image
from src.utils.helpers import get_device


class ImageInputError(RuntimeError):
    """The upload could not be decoded or normalized into an image."""


def to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class Analyzer:
    """Holds the loaded checkpoint and runs the full analysis per upload.

    Raises ModelLoadError from the constructor if the checkpoint can't be
    loaded, so the caller decides how to surface that."""

    def __init__(
        self,
        checkpoint_path: str,
        ela_quality: int,
        dct_block_size: int,
        normalize_quality: int,
        fix_orientation: bool,
        threshold_override: Optional[float],
    ):
        self.ela_quality = ela_quality
        self.dct_block_size = dct_block_size
        self.normalize_quality = normalize_quality
        self.fix_orientation = fix_orientation

        self.device = get_device()
        self.model, self.checkpoint = load_model(checkpoint_path, self.device)
        self.img_size = self.checkpoint.get("img_size", 224)
        self.threshold = (
            threshold_override
            if threshold_override is not None
            else self.checkpoint.get("inference_threshold", 0.5)
        )
        self._model_lock = threading.Lock()

    def analyze(self, raw_bytes: bytes) -> Dict:
        try:
            decoded = load_image(raw_bytes)
            normalized = normalize_and_reencode_jpeg(
                decoded, quality=self.normalize_quality, fix_orientation=self.fix_orientation
            )
        except ImageDecodeError as exc:
            raise ImageInputError(f"This file couldn't be read as an image: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - unusual modes/bit depths shouldn't crash the server
            raise ImageInputError(f"This image couldn't be normalized: {exc}") from exc

        result: Dict = {"original": to_data_uri(normalized), "threshold": self.threshold}
        result.update(self.run_model(normalized))
        result.update(self.run_forensics(normalized))
        return result

    def run_model(self, normalized: Image.Image) -> Dict:
        """CNN prediction and Grad-CAM. Uses only the shared model, so it is
        serialized by the model lock and independent of ELA/DCT."""
        out: Dict = {
            "label": None,
            "confidence_pct": None,
            "prob_tampered_pct": None,
            "gradcam": None,
            "gradcam_error": None,
            "prediction_error": None,
        }
        with self._model_lock:
            try:
                prediction = predict_image(
                    self.model, normalized, self.device, img_size=self.img_size, threshold=self.threshold
                )
            except Exception as exc:  # noqa: BLE001
                out["prediction_error"] = f"Prediction failed: {exc}"
                return out

            out["label"] = prediction["label"]
            out["confidence_pct"] = prediction["confidence"] * 100
            out["prob_tampered_pct"] = prediction["prob_tampered"] * 100
            try:
                overlay = generate_gradcam_overlay(
                    self.model, prediction["input_tensor"], prediction["display_image"]
                )
                out["gradcam"] = to_data_uri(overlay)
            except GradCAMError as exc:
                out["gradcam_error"] = str(exc)
            finally:
                self.model.zero_grad(set_to_none=True)
        return out

    def run_forensics(self, normalized: Image.Image) -> Dict:
        """ELA and DCT. Pure image processing: no model, no lock, and their
        output does not depend on the prediction."""
        out: Dict = {"ela": None, "ela_stats": None, "ela_error": None,
                     "dct": None, "dct_stats": None, "dct_error": None}
        try:
            ela_image, ela_stats = generate_ela_heatmap(normalized, quality=self.ela_quality)
            out["ela"] = to_data_uri(ela_image)
            out["ela_stats"] = ela_stats
        except Exception as exc:  # noqa: BLE001
            out["ela_error"] = str(exc)

        try:
            dct_image, dct_stats = generate_dct_heatmap(normalized, block_size=self.dct_block_size)
            out["dct"] = to_data_uri(dct_image)
            out["dct_stats"] = dct_stats
        except Exception as exc:  # noqa: BLE001
            out["dct_error"] = str(exc)
        return out
