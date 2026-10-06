"""
Single-image inference for a TamperResNet50 checkpoint, demonstrating
all three call modes the checkpoint supports out of the box:

    --mode predict           -> label + confidence + raw logit
    --mode extract_features  -> 2048-d feature vector
    --mode both              -> predict_with_features(): all of the above in one call

This is what makes a checkpoint saved by scripts/train_tamper_resnet50.py
directly reusable as Agent A/B in a later fusion setup: nothing here is
re-architected, it just calls the methods already defined on
TamperResNet50 (src/models/tamper_resnet50.py).

Run from the project root:
    python scripts/predict_tamper_resnet50.py --image path/to/image.jpg --model models/saved/tamper_resnet50_baseline_best.pth
"""

import argparse
import os
import sys

import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import get_eval_transforms
from src.models.tamper_resnet50 import TamperResNet50
from src.utils.helpers import get_device

CLASS_NAMES = ["Authentic", "Tampered"]  # index 0, 1 — matches src/data/dataset.py CLASS_TO_LABEL


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # pretrained=False: no need to download/load ImageNet weights just to
    # immediately overwrite them with the checkpoint's own state_dict below.
    model = TamperResNet50(**checkpoint["model_init_kwargs"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_tensor(image_path: str, img_size: int, device: torch.device) -> torch.Tensor:
    transform = get_eval_transforms(img_size)
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)


def main():
    parser = argparse.ArgumentParser(description="Run a TamperResNet50 checkpoint on a single image.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--model", required=True, help="Path to a TamperResNet50 checkpoint (.pt/.pth).")
    parser.add_argument(
        "--mode", choices=["predict", "extract_features", "both"], default="predict",
        help="predict = label+confidence+logit. extract_features = 2048-d vector. "
             "both = predict_with_features() in one call.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the checkpoint's saved inference threshold (bias INFERENCE only, "
             "no retraining needed - see TamperResNet50's module docstring).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"{args.model} not found.")

    device = get_device()
    model, checkpoint = load_model(args.model, device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("inference_threshold", 0.5)
    tensor = load_tensor(args.image, checkpoint.get("img_size", 224), device)

    print(f"Loaded checkpoint '{args.model}' (epoch {checkpoint.get('epoch')}, "
          f"run '{checkpoint.get('run_name')}'), threshold={threshold}")

    if args.mode == "predict":
        labels, probs, logits = model.predict(tensor, threshold=threshold)
        label = CLASS_NAMES[labels.item()]
        print(f"Prediction: {label}")
        print(f"Confidence (P(tampered)): {probs.item() * 100:.2f}%")
        print(f"Raw logit: {logits.item():.4f}")

    elif args.mode == "extract_features":
        features = model.extract_features(tensor)
        print(f"Feature vector shape: {tuple(features.shape)}")
        print(f"First 10 values: {features.squeeze(0)[:10].tolist()}")

    else:  # both
        labels, probs, logits, features = model.predict_with_features(tensor, threshold=threshold)
        label = CLASS_NAMES[labels.item()]
        print(f"Prediction: {label}")
        print(f"Confidence (P(tampered)): {probs.item() * 100:.2f}%")
        print(f"Raw logit: {logits.item():.4f}")
        print(f"Feature vector shape: {tuple(features.shape)}")


if __name__ == "__main__":
    main()
