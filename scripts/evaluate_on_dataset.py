"""
Evaluates a trained TamperResNet50 checkpoint on an ENTIRE dataset
directory (<dataset_dir>/authentic, <dataset_dir>/tampered) - no
train/val/test split, since the point is to test the model against a
dataset it was never trained on (or to report class-by-class results
when part of that dataset overlaps with the training set - see the
per-class breakdown below).

Guarantees the model is frozen (pure inference, no learning):
    - model.eval() disables Dropout and switches BatchNorm to use its
      running statistics instead of batch statistics.
    - Every forward pass runs under torch.no_grad(), so no gradients
      are computed and .backward()/optimizer.step() are never called -
      the checkpoint's weights cannot change no matter how this script
      is invoked.

Reuses the existing pipeline's components: src/data/dataset.py
(build_dataset_index, ImagePathDataset), src/data/preprocessing.py
(get_eval_transforms - the same deterministic transform used for
val/test/single-image prediction), src/utils/metrics.py, and
src/utils/visualize.py.

Run from the project root:
    python scripts/evaluate_on_dataset.py \\
        --model models/saved/tamper_resnet50_full_finetune_resumed_b96_best.pth \\
        --dataset-dir data/raw \\
        --report-dir results/reports/raw_direct
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import ImagePathDataset, LABEL_TO_CLASS, build_dataset_index
from src.data.preprocessing import get_eval_transforms
from src.models.tamper_resnet50 import TamperResNet50
from src.utils.helpers import get_device
from src.utils.metrics import compute_confusion, compute_metrics, print_metrics
from src.utils.visualize import plot_confusion_matrix


@torch.no_grad()
def run_inference(model, loader, device, threshold):
    model.eval()  # frozen: no Dropout, BatchNorm uses running stats only
    y_true, y_pred, probs_all = [], [], []

    for images, labels in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images)).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)

        probs_all.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().flatten().astype(int).tolist())

    return y_true, y_pred, probs_all


def per_class_accuracy(y_true, y_pred):
    """Accuracy computed separately within each true class - useful
    when one class overlaps with the model's training data and the
    other doesn't, so the two numbers shouldn't be read as one
    combined "accuracy" figure."""
    result = {}
    for label, name in LABEL_TO_CLASS.items():
        idx = [i for i, t in enumerate(y_true) if t == label]
        if not idx:
            continue
        correct = sum(1 for i in idx if y_pred[i] == y_true[i])
        result[name] = {"n": len(idx), "correct": correct, "accuracy": correct / len(idx)}
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TamperResNet50 checkpoint on a whole dataset directory.")
    parser.add_argument("--model", required=True, help="Path to a TamperResNet50 checkpoint (.pt/.pth).")
    parser.add_argument("--dataset-dir", required=True, help="Directory with authentic/ and tampered/ subfolders.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Override the checkpoint's saved inference threshold.")
    parser.add_argument("--report-dir", required=True, help="Where to save the confusion matrix plot and results JSON.")
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.model, map_location=device)
    img_size = checkpoint.get("img_size", 224)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("inference_threshold", 0.5)

    model = TamperResNet50(**checkpoint["model_init_kwargs"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint '{args.model}' (epoch {checkpoint.get('epoch')}, "
          f"run '{checkpoint.get('run_name')}'), threshold={threshold}")
    print("Model frozen for evaluation: model.eval() + torch.no_grad() - no weights will change.\n")

    print(f"Scanning dataset at '{args.dataset_dir}' ...")
    samples, _corrupted = build_dataset_index(args.dataset_dir)
    dataset = ImagePathDataset(samples, transform=get_eval_transforms(img_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"\nRunning inference on {len(samples)} images...")
    y_true, y_pred, probs = run_inference(model, loader, device, threshold)

    metrics = compute_metrics(y_true, y_pred)
    print(f"\nOverall results on '{args.dataset_dir}' (tampered = positive class, threshold={threshold}):")
    print_metrics(metrics)

    per_class = per_class_accuracy(y_true, y_pred)
    print("\nPer-class accuracy:")
    for name, stats in per_class.items():
        print(f"  {name:<10s}: {stats['correct']}/{stats['n']} = {stats['accuracy'] * 100:.2f}%")

    cm = compute_confusion(y_true, y_pred)
    print("\nConfusion matrix ([authentic, tampered] x [authentic, tampered]):")
    print(cm)

    os.makedirs(args.report_dir, exist_ok=True)
    plot_confusion_matrix(cm, class_names=["authentic", "tampered"],
                           save_path=os.path.join(args.report_dir, "confusion_matrix.png"))

    results = {
        "model": args.model,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_run_name": checkpoint.get("run_name"),
        "dataset_dir": args.dataset_dir,
        "threshold": threshold,
        "num_images": len(samples),
        "overall_metrics": metrics,
        "per_class_accuracy": per_class,
        "confusion_matrix": {"labels": ["authentic", "tampered"], "matrix": cm.tolist()},
    }
    with open(os.path.join(args.report_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.report_dir}/ (results.json, confusion_matrix.png)")


if __name__ == "__main__":
    main()
