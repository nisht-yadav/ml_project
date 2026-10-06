"""
Single-image prediction using the trained baseline CNN.

Run from the project root:
    python scripts/predict.py --image path/to/image.jpg
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
from src.models.baseline_cnn import BaselineCNN
from src.models.deep_cnn import DeepCNN
from src.models.resnet_model import ResNetCNN
from src.utils.helpers import get_device

MODEL_CLASSES = {"baseline": BaselineCNN, "deep": DeepCNN, "resnet50": ResNetCNN}

MODEL_PATH = os.path.join("models", "saved", "best_model.pth")
CLASS_NAMES = ["Authentic", "Tampered"]  # index 0, 1 — must match src/data/dataset.py CLASS_TO_LABEL


def predict_image(image_path: str, model: torch.nn.Module, device: torch.device, img_size: int = 224):
    """
    1. loads the image
    2. applies the SAME preprocessing used for validation/test data
       (no augmentation — a prediction must be deterministic)
    3. runs it through the CNN
    4. converts the raw logit to a probability via sigmoid
    5. classifies as authentic/tampered and prints the confidence
    """
    transform = get_eval_transforms(img_size)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension: (1, 3, H, W)

    model.eval()
    with torch.no_grad():
        logit = model(tensor)
        # The model outputs a raw logit; sigmoid converts it to
        # P(tampered) in [0, 1]. This is done here (not inside the
        # model) because training uses BCEWithLogitsLoss directly on
        # the logits — sigmoid is only needed to turn the output into a
        # human-readable probability at inference time.
        prob_tampered = torch.sigmoid(logit).item()

    if prob_tampered >= 0.5:
        predicted_class = CLASS_NAMES[1]
        confidence = prob_tampered
    else:
        predicted_class = CLASS_NAMES[0]
        confidence = 1 - prob_tampered

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict authentic/tampered for a single image.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the model checkpoint.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"{args.model} not found. Run scripts/train.py first to train "
            f"and save the best model checkpoint."
        )

    device = get_device()
    checkpoint = torch.load(args.model, map_location=device)
    img_size = checkpoint.get("img_size", 224)
    model_name = checkpoint.get("model_name", "baseline")  # older checkpoints predate this field

    model_cls = MODEL_CLASSES[model_name]
    model = model_cls(img_size=img_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    predict_image(args.image, model, device, img_size=img_size)


if __name__ == "__main__":
    main()
