"""
Generic single-image inference on top of a trained TamperResNet50
checkpoint. This is the "given a checkpoint and an image, load the model
and return a prediction" building block used by the web app (and
reusable from a notebook/script/test without any web-framework dependency).

Deliberately generic: no dataset paths, no training-loop concepts,
just "checkpoint path in, prediction out" for exactly one image at a
time — matching how a real upload arrives in the web app.

Errors are raised as the two typed exceptions below (rather than a bare
Exception) so a caller such as the web app can catch each failure mode
separately and show a specific, non-crashing message instead of a
generic error page.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from PIL import Image, UnidentifiedImageError

from src.data.preprocessing import get_eval_transforms
from src.models.tamper_resnet50 import TamperResNet50

CLASS_NAMES = ["authentic", "tampered"]  # index 0/1, matches src/data/dataset.py CLASS_TO_LABEL


class ModelLoadError(RuntimeError):
    """The checkpoint file is missing, unreadable, or not a valid TamperResNet50 checkpoint."""


class ImageDecodeError(RuntimeError):
    """The given file/bytes could not be decoded as an image (corrupted upload,
    unsupported/unrecognized format, etc.)."""


def load_model(checkpoint_path: Union[str, Path], device: torch.device) -> Tuple[TamperResNet50, dict]:
    """
    Loads a TamperResNet50 checkpoint saved by scripts/train_tamper_resnet50.py.
    The checkpoint's own `model_init_kwargs` is used to reconstruct the exact
    architecture it was trained with, so this works for any checkpoint from
    that pipeline without needing to know its freeze/dropout settings ahead
    of time.

    Returns (model in eval mode, raw checkpoint dict — useful for img_size,
    inference_threshold, class_names, etc).
    """
    checkpoint_path = str(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise ModelLoadError(f"Model checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        init_kwargs = checkpoint["model_init_kwargs"]
        # pretrained=False: skip downloading/loading ImageNet weights since
        # load_state_dict below overwrites them anyway.
        model = TamperResNet50(**init_kwargs, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure here means "unusable checkpoint"
        raise ModelLoadError(f"Failed to load checkpoint '{checkpoint_path}': {exc}") from exc

    return model, checkpoint


def load_image(image: Union[str, Path, Image.Image, bytes]) -> Image.Image:
    """Accepts a file path, raw bytes (e.g. an uploaded file's contents), or
    an already-open PIL image, and returns a decoded RGB PIL image."""
    try:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, bytes):
            import io
            return Image.open(io.BytesIO(image)).convert("RGB")
        return Image.open(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError(f"Could not decode image: {exc}") from exc


def preprocess(image: Image.Image, img_size: int, device: torch.device) -> torch.Tensor:
    """Applies the same deterministic letterbox-resize + normalize pipeline
    used for validation/test during training, so a live prediction matches
    what was measured at training time. Returns a (1, 3, H, W) batch tensor."""
    transform = get_eval_transforms(img_size)
    return transform(image).unsqueeze(0).to(device)


def predict_image(
    model: TamperResNet50,
    image: Union[str, Path, Image.Image, bytes],
    device: torch.device,
    img_size: int = 224,
    threshold: float = 0.5,
) -> Dict:
    """
    Runs one image through `model` end to end.

    Returns a dict:
        label:          "authentic" or "tampered"
        confidence:     sigmoid probability OF THE PREDICTED LABEL, in [0, 1]
                        (i.e. how confident the model is in whatever it
                        predicted — the number most useful to show a user)
        prob_tampered:  sigmoid probability of "tampered" specifically,
                        in [0, 1], regardless of which label won
        logit:          raw pre-sigmoid score
        threshold:      decision threshold that was applied
        input_tensor:   the (1, 3, img_size, img_size) normalized tensor fed
                        to the model (reusable by Grad-CAM without redoing
                        preprocessing)
        display_image:  the letterboxed-but-unnormalized RGB PIL image at
                        img_size x img_size, matching what the model
                        actually "saw" (reusable as the Grad-CAM overlay base)
    """
    pil_image = load_image(image)
    tensor = preprocess(pil_image, img_size, device)

    labels, probs, logits = model.predict(tensor, threshold=threshold)
    label_idx = int(labels.item())
    prob_tampered = float(probs.item())
    confidence = prob_tampered if label_idx == 1 else (1.0 - prob_tampered)

    from src.data.preprocessing import LetterboxResize
    display_image = LetterboxResize(img_size)(pil_image)

    return {
        "label": CLASS_NAMES[label_idx],
        "label_idx": label_idx,
        "confidence": confidence,
        "prob_tampered": prob_tampered,
        "logit": float(logits.item()),
        "threshold": threshold,
        "input_tensor": tensor,
        "display_image": display_image,
    }
