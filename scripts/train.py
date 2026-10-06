"""
Baseline CNN training pipeline - Step 1 of the Image Tampering Detection
project (no DCT/ELA/Grad-CAM/deployment yet, just: image -> CNN ->
authentic/tampered).

Run from the project root:
    python scripts/train.py
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.optim import Adam

# Make `src` importable regardless of the working directory this is run from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import create_dataloaders
from src.models.baseline_cnn import BaselineCNN
from src.models.baseline_cnn import print_shape_trace as print_shape_trace_baseline
from src.models.deep_cnn import DeepCNN
from src.models.deep_cnn import print_shape_trace as print_shape_trace_deep
from src.models.resnet_model import ResNetCNN
from src.models.resnet_model import print_shape_trace as print_shape_trace_resnet
from src.utils.helpers import EarlyStopping, get_device, set_seed
from src.utils.metrics import compute_confusion, compute_metrics, print_metrics
from src.utils.visualize import plot_confusion_matrix, plot_training_history

# ============================================================
# 2. CONFIGURATION
# ============================================================
DATASET_DIR = os.path.join("data", "subset")  # expects DATASET_DIR/authentic, DATASET_DIR/tampered
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

LEARNING_RATE = 3e-4        # used for BaselineCNN/DeepCNN, and resnet50's head
RESNET_BACKBONE_LR = 1e-5   # resnet50 only, when FREEZE_BACKBONE=False - see note above
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 2

# "baseline" = 3 conv blocks, from scratch      (src/models/baseline_cnn.py)
# "deep"     = 7 conv layers, from scratch      (src/models/deep_cnn.py) -
#              turned out unstable to train (see resnet_model.py docstring)
# "resnet50" = pretrained ResNet50 backbone      (src/models/resnet_model.py)
MODEL_NAME = "resnet50"

# resnet50 only: True = backbone fully frozen, only the ~262K-parameter
# head is trained. False = num_frozen_layers below controls how much of
# the backbone is trainable.
FREEZE_BACKBONE = False

# resnet50 only, when FREEZE_BACKBONE=False: freezes the stem (conv1+bn1)
# plus this many of ResNet50's 4 named stages (layer1..layer4), 0-4.
# 0 = full fine-tuning (every weight trainable - real overfitting risk on
#     a ~3500-image training set, needs RESNET_BACKBONE_LR to stay stable).
# 3 = freezes stem+layer1+layer2+layer3, leaves only layer4 (ResNet50's
#     most task-specific stage) + the head trainable - more aggressive
#     than num_frozen_layers=2 (which left layer3 trainable too).
RESNET_NUM_FROZEN_LAYERS = 3

# data/subset is ~3000 authentic vs. 2064 tampered (~1.45:1) - much milder
# than data/raw's ~3.6:1, so a smaller pos_weight is enough to stop the
# loss from favoring "predict authentic" without over-correcting into
# "predict tampered" instead (which is what happened with pos_weight=3.6
# on the already-imbalanced data/raw).
USE_POS_WEIGHT = True
POS_WEIGHT = 1.4

MODEL_SAVE_PATH = os.path.join("models", "saved", "best_model.pth")
SPLIT_SAVE_PATH = os.path.join("data", "processed", "split.json")
PLOTS_DIR = os.path.join("results", "plots")

# A short, distinctive name for this run. At the end of training, the
# checkpoint/split/plots/config/results are copied into
# experiments/<RUN_NAME>/ as a permanent, self-contained record - so later
# runs (which overwrite MODEL_SAVE_PATH/PLOTS_DIR above) never destroy the
# evidence for an earlier one.
RUN_NAME = "resnet50_partial_freeze3_subset_posweight1.4_lr3e-4"
EXPERIMENTS_DIR = "experiments"


# ============================================================
# 10. TRAINING FUNCTION (one epoch over train_loader)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)                 # raw scores, shape (B, 1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # sigmoid turns the logit into P(tampered); threshold at 0.5 for
        # a hard prediction. (Applied here only for the accuracy metric -
        # BCEWithLogitsLoss itself works directly on the raw logits.)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ============================================================
# 11. VALIDATION FUNCTION
# ============================================================
@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """
    Runs one pass over `loader` and returns loss/accuracy plus
    precision/recall/F1 (tampered = positive class). Precision/recall
    are tracked every epoch - not just at final test time - because with
    an imbalanced dataset, accuracy alone can look good (~78-80%) while
    the model has actually just learned to predict "authentic" and is
    barely catching real tampered images (low recall). Watching recall
    per epoch shows that failure mode as it happens instead of only at
    the end.
    """
    model.eval()
    running_loss, total = 0.0, 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        y_pred.extend(preds.cpu().numpy().flatten().astype(int).tolist())
        y_true.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    metrics = compute_metrics(y_true, y_pred)
    return running_loss / total, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]


@torch.no_grad()
def run_test_evaluation(model, loader, device):
    """Collects predictions over the full test set for sklearn metrics."""
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images)).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().flatten().astype(int).tolist())

    return y_true, y_pred


def archive_experiment(checkpoint, test_metrics, confusion, stopped_at_epoch):
    """
    Copies this run's checkpoint/split/plots into experiments/<RUN_NAME>/
    and writes a summary.json recording the exact config and results, so
    the run remains a permanent, self-contained record even after
    MODEL_SAVE_PATH/PLOTS_DIR are overwritten by a later run.
    """
    run_dir = os.path.join(EXPERIMENTS_DIR, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    shutil.copy2(MODEL_SAVE_PATH, os.path.join(run_dir, "best_model.pth"))
    if os.path.exists(SPLIT_SAVE_PATH):
        shutil.copy2(SPLIT_SAVE_PATH, os.path.join(run_dir, "split.json"))
    for plot_name in ("confusion_matrix.png", "training_history.png"):
        src = os.path.join(PLOTS_DIR, plot_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, plot_name))

    summary = {
        "run_name": RUN_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_name": MODEL_NAME,
            "freeze_backbone": FREEZE_BACKBONE if MODEL_NAME == "resnet50" else None,
            "num_frozen_layers": RESNET_NUM_FROZEN_LAYERS if MODEL_NAME == "resnet50" and not FREEZE_BACKBONE else None,
            "dataset_dir": DATASET_DIR,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "seed": SEED,
            "learning_rate_head": LEARNING_RATE,
            "learning_rate_backbone": RESNET_BACKBONE_LR if MODEL_NAME == "resnet50" and not FREEZE_BACKBONE else None,
            "epochs_cap": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "use_pos_weight": USE_POS_WEIGHT,
            "pos_weight": POS_WEIGHT if USE_POS_WEIGHT else None,
        },
        "best_checkpoint_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "best_val_acc": checkpoint["val_acc"],
        "stopped_at_epoch": stopped_at_epoch,
        "test_metrics": test_metrics,
        "confusion_matrix": {
            "labels": ["authentic", "tampered"],
            "matrix": confusion.tolist(),
        },
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment archived to {run_dir}/ "
          f"(best_model.pth, split.json, plots, summary.json)")


def main():
    # ============================================================
    # 3. RANDOM SEED
    # ============================================================
    set_seed(SEED)

    device = get_device()

    # ============================================================
    # 4-7. DATASET LOADING, SPLITTING, TRANSFORMS, DATALOADERS
    #      (all handled inside create_dataloaders — see
    #       src/data/dataset.py, dataloader.py, preprocessing.py)
    # ============================================================
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        split_save_path=SPLIT_SAVE_PATH,
    )

    # ============================================================
    # 8. CNN MODEL
    # ============================================================
    if MODEL_NAME == "resnet50":
        model = ResNetCNN(
            img_size=IMG_SIZE,
            freeze_backbone=FREEZE_BACKBONE,
            num_frozen_layers=RESNET_NUM_FROZEN_LAYERS,
        ).to(device)
        shape_trace_fn = print_shape_trace_resnet
    elif MODEL_NAME == "deep":
        model = DeepCNN(img_size=IMG_SIZE).to(device)
        shape_trace_fn = print_shape_trace_deep
    else:
        model = BaselineCNN(img_size=IMG_SIZE).to(device)
        shape_trace_fn = print_shape_trace_baseline

    print(f"\nModel: {MODEL_NAME}")
    print("Model architecture:")
    print(model)
    print()
    shape_trace_fn(model, img_size=IMG_SIZE)

    # ============================================================
    # 9. LOSS AND OPTIMIZER
    # ============================================================
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one step.
    # It's used (instead of applying sigmoid ourselves + plain BCELoss)
    # because it's numerically more stable, and it's the standard choice
    # for a single-output-neuron binary classifier.
    # pos_weight must be a tensor on the same device as the logits it's
    # multiplied against.
    if USE_POS_WEIGHT:
        pos_weight = torch.tensor([POS_WEIGHT], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    if MODEL_NAME == "resnet50" and not FREEZE_BACKBONE:
        # Differential learning rates: whatever part of the backbone is
        # trainable (all of it, or just layer3+layer4 when num_frozen_layers
        # > 0) gets a much smaller step size than the freshly-initialized
        # head, since its weights already encode useful features and large
        # updates would destroy them before the head has learned anything
        # dataset-specific. Filtered to requires_grad=True so frozen
        # sub-layers (when num_frozen_layers > 0) aren't given optimizer
        # state they'll never use.
        backbone_trainable = [p for p in model.features.parameters() if p.requires_grad]
        head_trainable = list(model.embedding_head.parameters()) + list(model.output_head.parameters())
        optimizer = Adam([
            {"params": backbone_trainable, "lr": RESNET_BACKBONE_LR},
            {"params": head_trainable, "lr": LEARNING_RATE},
        ])
        print(f"Optimizer: backbone lr={RESNET_BACKBONE_LR}, head lr={LEARNING_RATE} ({model.freeze_description()})")
    else:
        # Only optimize parameters that actually require gradients - when the
        # ResNet50 backbone is frozen, its ~25M parameters have requires_grad=False
        # and would otherwise waste optimizer memory (Adam keeps per-parameter
        # momentum buffers) for weights that never update anyway.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(trainable_params, lr=LEARNING_RATE)

    # ============================================================
    # 12. TRAINING LOOP
    # ============================================================
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [],
    }

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\nStarting training for up to {EPOCHS} epochs "
          f"(early stopping patience = {EARLY_STOPPING_PATIENCE})...\n")

    final_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        final_epoch = epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"Validation Precision: {val_precision * 100:.2f}% | "
              f"Validation Recall: {val_recall * 100:.2f}% | "
              f"Validation F1: {val_f1 * 100:.2f}%")

        # ============================================================
        # 13. SAVE BEST MODEL (checkpoint on best validation loss, not
        #     the final epoch's weights - the model may have started
        #     overfitting again after its best epoch by the time
        #     training stops)
        # ============================================================
        is_best = early_stopping.step(val_loss)
        if is_best:
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "img_size": IMG_SIZE,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": class_names,
            }, MODEL_SAVE_PATH)
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        if early_stopping.should_stop:
            print(f"\nEarly stopping: validation loss hasn't improved for "
                  f"{EARLY_STOPPING_PATIENCE} epochs. Stopping at epoch {epoch}.")
            break

        print()

    # ============================================================
    # 14. TEST EVALUATION (using the BEST checkpoint, not the final-epoch
    #     weights — the model may have overfit further after its best
    #     validation epoch, so reloading the checkpoint guarantees the
    #     reported test metrics reflect the model's best generalization
    #     point, not wherever training happened to stop)
    # ============================================================
    print(f"\nTraining finished after {final_epoch} epoch(s).")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint: epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc'] * 100:.2f}%")

    # The test set was never touched during training or model selection
    # (not used for gradient updates, not used for early stopping) -
    # this is the first and only time it's used, giving an unbiased
    # estimate of real-world performance.
    print("\nEvaluating on the held-out TEST set (never seen during training)...")
    y_true, y_pred = run_test_evaluation(model, test_loader, device)

    test_metrics = compute_metrics(y_true, y_pred)
    print("\nTest set results (tampered = positive class):")
    print_metrics(test_metrics)

    # ============================================================
    # 15. CONFUSION MATRIX
    # ============================================================
    cm = compute_confusion(y_true, y_pred)
    print("\nConfusion matrix ([authentic, tampered] x [authentic, tampered]):")
    print(cm)
    plot_confusion_matrix(
        cm, class_names=["authentic", "tampered"],
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"),
    )

    # ============================================================
    # 16. TRAINING PLOTS
    # ============================================================
    plot_training_history(history, save_path=os.path.join(PLOTS_DIR, "training_history.png"))

    print(f"\nDone. Best model checkpoint: {MODEL_SAVE_PATH}")

    # ============================================================
    # 17. ARCHIVE THIS RUN (permanent record under experiments/RUN_NAME/)
    # ============================================================
    archive_experiment(checkpoint, test_metrics, cm, stopped_at_epoch=final_epoch)


if __name__ == "__main__":
    main()
