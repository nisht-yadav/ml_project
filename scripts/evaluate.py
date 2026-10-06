"""
Standalone test-set evaluation.

Reuses the exact split that scripts/train.py saved to
data/processed/split.json (rather than re-scanning the dataset folder
and re-splitting), so the test set here is guaranteed to be the same
images the model never trained or validated on.

Run from the project root, after scripts/train.py has produced
models/saved/best_model.pth:
    python scripts/evaluate.py
"""

import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import ImagePathDataset
from src.data.preprocessing import get_eval_transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.deep_cnn import DeepCNN
from src.models.resnet_model import ResNetCNN
from src.utils.helpers import get_device

MODEL_CLASSES = {"baseline": BaselineCNN, "deep": DeepCNN, "resnet50": ResNetCNN}
from src.utils.metrics import compute_confusion, compute_metrics, print_metrics
from src.utils.visualize import plot_confusion_matrix

MODEL_PATH = os.path.join("models", "saved", "best_model.pth")
SPLIT_PATH = os.path.join("data", "processed", "split.json")
BATCH_SIZE = 32
PLOTS_DIR = os.path.join("results", "plots")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images)).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().flatten().astype(int).tolist())
    return y_true, y_pred


def main():
    device = get_device()

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(
            f"{SPLIT_PATH} not found. Run scripts/train.py first — it "
            f"creates the train/val/test split and saves it here."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run scripts/train.py first to train "
            f"and save the best model checkpoint."
        )

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    test_samples = [tuple(sample) for sample in split["test"]]

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    img_size = checkpoint.get("img_size", 224)
    class_names = checkpoint.get("class_names", ["authentic", "tampered"])
    model_name = checkpoint.get("model_name", "baseline")  # older checkpoints predate this field

    model_cls = MODEL_CLASSES[model_name]
    test_ds = ImagePathDataset(test_samples, transform=get_eval_transforms(img_size))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = model_cls(img_size=img_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded {model_name} checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")
    print(f"Evaluating on {len(test_samples)} held-out test images...\n")

    y_true, y_pred = evaluate(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred)
    print("Test set results (tampered = positive class):")
    print_metrics(metrics)

    cm = compute_confusion(y_true, y_pred)
    print("\nConfusion matrix ([authentic, tampered] x [authentic, tampered]):")
    print(cm)
    plot_confusion_matrix(
        cm, class_names=class_names,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"),
    )


if __name__ == "__main__":
    main()
