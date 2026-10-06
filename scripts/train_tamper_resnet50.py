"""
Config-driven training pipeline for TamperResNet50 (ResNet50-backbone
binary authentic-vs-tampered classifier).

Every major design choice - stage freezing, differential LRs, BN
freezing, augmentation, class-bias vs. decision-threshold, epoch caps,
etc. - is read from a YAML config file, not hardcoded here. The same
script produces the "plain baseline", "class-biased", and
"feature-extracting" variants described in the project brief just by
pointing --config at a different file (see configs/tamper_resnet50_*.yaml)
- training logic never changes.

training.mixed_precision (config) switches the forward pass + loss to
float16 autocast with gradient scaling instead of plain float32 - see
resolve_amp() below. Falls back to float32 automatically on CPU.

Continuing training on a second dataset (e.g. domain adaptation, or
just picking up where an earlier run left off) is a config value, not
a code change: set model.init_from_checkpoint in the YAML to a
previous run's checkpoint path, point data.dataset_dir at the new
dataset, and give it a new experiment.run_name. The model's weights
are loaded from that checkpoint instead of ImageNet init; everything
else (freeze_stages, lr_groups, split, early stopping, ...) still
comes from this run's own config, so you can also change what's frozen
or the learning rates for the second phase.

Reuses the existing pipeline's components wherever they already do the
job: src/data/dataset.py + dataloader.py + preprocessing.py for data,
src/utils/helpers.py for seeding/device/early-stopping,
src/utils/metrics.py + visualize.py for evaluation/plots. Only the
model (src/models/tamper_resnet50.py, needed for the 2048-d feature
hook + explicit stage freezing + differential LR groups) and this
training script are new.

Run from the project root:
    python scripts/train_tamper_resnet50.py --config configs/tamper_resnet50_baseline.yaml
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import create_dataloaders
from src.models.tamper_resnet50 import TamperResNet50, print_shape_trace
from src.utils.helpers import EarlyStopping, get_device, set_seed
from src.utils.metrics import compute_confusion, compute_metrics, print_metrics
from src.utils.visualize import plot_confusion_matrix, plot_training_history

DEFAULT_THRESHOLD_FOR_MONITORING = 0.5
# Per-epoch train/val accuracy/precision/recall during training always use
# the plain 0.5 boundary, deliberately independent of cfg["inference"]["threshold"].
# That threshold is a post-training inference-time lever (see
# TamperResNet50's module docstring) - applying it mid-training would
# make early-stopping/checkpoint-selection decisions depend on a knob
# meant for post-hoc deployment tuning, not model selection.


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_amp(mixed_precision: bool, device: torch.device) -> bool:
    """
    training.mixed_precision (config) requests float16 autocast for the
    forward pass + loss, with a GradScaler handling the backward pass -
    this is standard mixed-precision training (float32 master weights,
    float16 compute), not a blanket model.half(): running BatchNorm and
    gradient accumulation purely in float16 is numerically unstable,
    which is exactly why GradScaler exists (it rescales the loss to
    stop float16 gradients from underflowing to zero).

    float16 autocast only helps (and is only reliably supported) on
    CUDA, so it's silently disabled on CPU regardless of the config -
    the run still works, just without the speed/memory benefit.
    """
    use_amp = bool(mixed_precision) and device.type == "cuda"
    if mixed_precision and not use_amp:
        print("Note: training.mixed_precision=true but device is CPU - "
              "running in float32 instead (float16 autocast needs CUDA).")
    return use_amp


# ============================================================
# TRAIN / VALIDATION LOOPS
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits.float()) >= DEFAULT_THRESHOLD_FOR_MONITORING).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss, total = 0.0, 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = (torch.sigmoid(logits.float()) >= DEFAULT_THRESHOLD_FOR_MONITORING).float()
        y_pred.extend(preds.cpu().numpy().flatten().astype(int).tolist())
        y_true.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    metrics = compute_metrics(y_true, y_pred)
    return running_loss / total, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]


@torch.no_grad()
def run_test_evaluation(model, loader, device, threshold: float, use_amp: bool):
    """Uses cfg["inference"]["threshold"] - the one place besides
    predict()/predict_with_features() where the inference-time decision
    boundary actually matters, since this is the number reported as
    "how the model performs when deployed"."""
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)

        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().flatten().astype(int).tolist())

    return y_true, y_pred


def archive_experiment(cfg, run_dir, checkpoint, test_metrics, confusion, stopped_at_epoch, image_counts):
    os.makedirs(run_dir, exist_ok=True)

    paths = cfg["paths"]
    shutil.copy2(paths["model_save_path"], os.path.join(run_dir, "best_model.pt"))
    if os.path.exists(paths["last_checkpoint_path"]):
        shutil.copy2(paths["last_checkpoint_path"], os.path.join(run_dir, "last_model.pt"))
    if os.path.exists(paths["split_save_path"]):
        shutil.copy2(paths["split_save_path"], os.path.join(run_dir, "split.json"))
    for plot_name in ("confusion_matrix.png", "training_history.png"):
        src = os.path.join(paths["plots_dir"], plot_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, plot_name))

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    summary = {
        "run_name": cfg["experiment"]["run_name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "image_counts": image_counts,
        "best_checkpoint_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "best_val_acc": checkpoint["val_acc"],
        "stopped_at_epoch": stopped_at_epoch,
        "inference_threshold_used_for_test_metrics": cfg["inference"]["threshold"],
        "test_metrics": test_metrics,
        "confusion_matrix": {
            "labels": ["authentic", "tampered"],
            "matrix": confusion.tolist(),
        },
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment archived to {run_dir}/ "
          f"(best_model.pt, last_model.pt, split.json, plots, config.yaml, summary.json)")


def main():
    parser = argparse.ArgumentParser(description="Train TamperResNet50 from a YAML config.")
    parser.add_argument(
        "--config", default=os.path.join("configs", "tamper_resnet50_baseline.yaml"),
        help="Path to a YAML config (see configs/tamper_resnet50_*.yaml).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg, data_cfg, model_cfg = cfg["experiment"], cfg["data"], cfg["model"]
    optim_cfg, loss_cfg, infer_cfg = cfg["optim"], cfg["loss"], cfg["inference"]
    train_cfg, paths_cfg = cfg["training"], cfg["paths"]

    print(f"Run: {exp_cfg['run_name']}  (config: {args.config})")

    set_seed(exp_cfg["seed"])
    device = get_device()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_dir=data_cfg["dataset_dir"],
        img_size=data_cfg["img_size"],
        batch_size=data_cfg["batch_size"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        seed=exp_cfg["seed"],
        num_workers=data_cfg["num_workers"],
        split_save_path=paths_cfg["split_save_path"],
        augment=data_cfg["augment"],
        hflip=data_cfg["hflip"],
        random_crop=data_cfg["random_crop"],
        color_jitter=data_cfg["color_jitter"],
        jpeg_recompression=data_cfg["jpeg_recompression"],
        jpeg_quality_range=tuple(data_cfg["jpeg_quality_range"]),
    )

    image_counts = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
        "test": len(test_loader.dataset),
        "total": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
    }
    print(f"\nImage counts -> train: {image_counts['train']}, val: {image_counts['val']}, "
          f"test: {image_counts['test']} (total: {image_counts['total']})")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    init_from_checkpoint = model_cfg.get("init_from_checkpoint")

    model = TamperResNet50(
        freeze_stages=model_cfg["freeze_stages"],
        freeze_bn=model_cfg["freeze_bn"],
        dropout=model_cfg["dropout"],
        # ImageNet pretrained weights are skipped (not loaded at all) when
        # continuing from a checkpoint below - they'd just be immediately
        # overwritten by load_state_dict(), so loading them first would
        # only cost a download/init with no effect.
        pretrained=init_from_checkpoint is None,
    ).to(device)

    if init_from_checkpoint:
        prev_checkpoint = torch.load(init_from_checkpoint, map_location=device)
        model.load_state_dict(prev_checkpoint["model_state_dict"])
        print(f"\nInitialized weights from checkpoint: {init_from_checkpoint} "
              f"(run '{prev_checkpoint.get('run_name')}', epoch {prev_checkpoint.get('epoch')}) "
              f"- continuing training on '{data_cfg['dataset_dir']}'.")

    print(f"\nModel: TamperResNet50 ({model.freeze_description()})")
    print_shape_trace(model, img_size=data_cfg["img_size"])

    # ------------------------------------------------------------------
    # Loss: pos_weight biases TRAINING (see TamperResNet50 module docstring)
    # ------------------------------------------------------------------
    if loss_cfg["use_pos_weight"]:
        pos_weight = torch.tensor([loss_cfg["pos_weight"]], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss(pos_weight={loss_cfg['pos_weight']})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss (no pos_weight)")

    # ------------------------------------------------------------------
    # Optimizer: differential LR per stage group, entirely config-driven
    # ------------------------------------------------------------------
    param_groups = model.get_param_groups(optim_cfg["lr_groups"])
    optimizer = Adam(param_groups, weight_decay=optim_cfg["weight_decay"])
    print("Optimizer param groups:")
    for group in param_groups:
        n_params = sum(p.numel() for p in group["params"])
        print(f"  {group['name']:<16s} lr={group['lr']:<10} params={n_params:,}")

    # ------------------------------------------------------------------
    # Mixed precision (float16 autocast + loss scaling) - see resolve_amp()
    # ------------------------------------------------------------------
    use_amp = resolve_amp(train_cfg["mixed_precision"], device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision (float16): {'enabled' if use_amp else 'disabled'}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    early_stopping = EarlyStopping(patience=train_cfg["early_stopping_patience"])
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [],
    }

    os.makedirs(os.path.dirname(paths_cfg["model_save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths_cfg["last_checkpoint_path"]), exist_ok=True)
    os.makedirs(paths_cfg["plots_dir"], exist_ok=True)

    epochs_cap = train_cfg["epochs_cap"]
    print(f"\nStarting training for up to {epochs_cap} epochs "
          f"(early stopping patience = {train_cfg['early_stopping_patience']})...\n")

    def build_checkpoint(epoch, val_loss, val_acc):
        return {
            "model_state_dict": model.state_dict(),
            "model_class": "TamperResNet50",
            "model_init_kwargs": {
                "freeze_stages": model_cfg["freeze_stages"],
                "freeze_bn": model_cfg["freeze_bn"],
                "dropout": model_cfg["dropout"],
            },
            "img_size": data_cfg["img_size"],
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "class_names": class_names,
            "inference_threshold": infer_cfg["threshold"],
            "image_counts": image_counts,
            "run_name": exp_cfg["run_name"],
            "initialized_from": init_from_checkpoint,  # provenance: None for a from-ImageNet run
            "trained_with_amp": use_amp,  # provenance only - weights are stored/loaded as float32 either way
        }

    best_epoch = None
    final_epoch = 0
    for epoch in range(1, epochs_cap + 1):
        final_epoch = epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_epoch(model, val_loader, criterion, device, use_amp)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{epochs_cap}")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"Validation Precision: {val_precision * 100:.2f}% | "
              f"Validation Recall: {val_recall * 100:.2f}% | "
              f"Validation F1: {val_f1 * 100:.2f}%")

        # Checkpoint every epoch ("last"), independent of whether it's the best.
        torch.save(build_checkpoint(epoch, val_loss, val_acc), paths_cfg["last_checkpoint_path"])

        is_best = early_stopping.step(val_loss)
        if is_best:
            torch.save(build_checkpoint(epoch, val_loss, val_acc), paths_cfg["model_save_path"])
            best_epoch = epoch
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        if early_stopping.should_stop:
            print(f"\nEarly stopping: validation loss hasn't improved for "
                  f"{train_cfg['early_stopping_patience']} epochs. Stopping at epoch {epoch}.")
            break

        print()

    print(f"\nTraining finished after {final_epoch} epoch(s). Best epoch: {best_epoch} "
          f"(val_loss={early_stopping.best_loss:.4f}).")

    # ------------------------------------------------------------------
    # Test evaluation (best checkpoint, inference-time threshold applied)
    # ------------------------------------------------------------------
    checkpoint = torch.load(paths_cfg["model_save_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint: epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc'] * 100:.2f}%")

    print(f"\nEvaluating on the held-out TEST set (threshold={infer_cfg['threshold']})...")
    y_true, y_pred = run_test_evaluation(model, test_loader, device, threshold=infer_cfg["threshold"], use_amp=use_amp)

    test_metrics = compute_metrics(y_true, y_pred)
    print("\nTest set results (tampered = positive class):")
    print_metrics(test_metrics)

    cm = compute_confusion(y_true, y_pred)
    print("\nConfusion matrix ([authentic, tampered] x [authentic, tampered]):")
    print(cm)
    plot_confusion_matrix(
        cm, class_names=["authentic", "tampered"],
        save_path=os.path.join(paths_cfg["plots_dir"], "confusion_matrix.png"),
    )
    plot_training_history(history, save_path=os.path.join(paths_cfg["plots_dir"], "training_history.png"))

    print(f"\nDone. Best model checkpoint (epoch {best_epoch}): {paths_cfg['model_save_path']}")
    print(f"Last-epoch checkpoint: {paths_cfg['last_checkpoint_path']}")

    run_dir = os.path.join(paths_cfg["experiments_dir"], exp_cfg["run_name"])
    archive_experiment(cfg, run_dir, checkpoint, test_metrics, cm, stopped_at_epoch=final_epoch, image_counts=image_counts)


if __name__ == "__main__":
    main()
