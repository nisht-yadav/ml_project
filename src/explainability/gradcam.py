"""
Grad-CAM explainability for TamperResNet50, built on the well-maintained
`pytorch-grad-cam` library (package `grad-cam` in requirements.txt,
imported as `pytorch_grad_cam`) rather than a from-scratch hook
implementation.

Grad-CAM highlights which spatial regions of the input most influenced
the model's output by backpropagating from that output to the last
convolutional feature map and weighting each channel by how much it
mattered. `TamperResNet50` exposes exactly the hook point Grad-CAM
needs via `get_last_conv_layer()` (the final Conv2d in `layer4`), so no
architecture-specific wiring is needed here beyond that one call.

Because the model has a single raw logit output (not a multi-class
softmax), `RawScoresOutputTarget` is used instead of a class-index
target: it backpropagates directly from that one score, which is the
exact same score `predict()` applies sigmoid + threshold to.
"""

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

from src.models.tamper_resnet50 import TamperResNet50


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM computation fails for a given model/input pair."""


def generate_gradcam_overlay(
    model: TamperResNet50,
    input_tensor: torch.Tensor,
    display_image: Image.Image,
) -> Image.Image:
    """
    Args:
        model: a loaded TamperResNet50 (see src/inference/predictor.py::load_model).
        input_tensor: the exact (1, 3, H, W) normalized tensor the model was
            scored on (e.g. `predict_image(...)["input_tensor"]`) — using the
            same tensor the prediction used means the heatmap explains the
            prediction actually shown, not a re-preprocessed approximation.
        display_image: the same image, letterboxed to match `input_tensor`'s
            spatial size but NOT normalized (e.g.
            `predict_image(...)["display_image"]`), so the heatmap overlays
            onto correctly-colored pixels.

    Returns:
        PIL.Image (RGB) — `display_image` with the Grad-CAM heatmap overlaid,
        same size as `display_image`.
    """
    try:
        target_layers = [model.get_last_conv_layer()]

        # Grad-CAM needs gradients: predict()/extract_features() run under
        # torch.no_grad() internally, but this call is independent of those
        # and must run with autograd enabled end to end.
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        with torch.enable_grad(), GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=[RawScoresOutputTarget()])[0]

        rgb_float = np.asarray(display_image.convert("RGB"), dtype=np.float32) / 255.0
        overlay_bgr_or_rgb = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        return Image.fromarray(overlay_bgr_or_rgb)
    except Exception as exc:  # noqa: BLE001 - surfaced as one typed, catchable failure mode
        raise GradCAMError(f"Grad-CAM computation failed: {exc}") from exc
