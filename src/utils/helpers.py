"""
General-purpose helpers shared across the project:
- reproducibility (seeding)
- device selection (CPU vs CUDA)
- early stopping

Kept independent of any specific model/dataset so they can be reused
by later stages of the project (frequency-domain branch, Grad-CAM, etc).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Make results reproducible.

    Setting the same seed for Python's `random`, NumPy, and PyTorch means
    that things like weight initialization, data shuffling, and the
    train/val/test split all happen the same way every time the script
    is run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Pick CUDA if available, otherwise fall back to CPU, and announce
    which one is being used (the task explicitly asks for this to be
    printed clearly).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


class EarlyStopping:
    """
    Stops training when the validation loss has not improved for
    `patience` consecutive epochs.

    We watch validation loss (not training loss) because training loss
    keeps improving even after the model has started overfitting - the
    validation set is our proxy for "unseen data" during training.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Call once per epoch with the latest validation loss.
        Returns True if this epoch is a new best (i.e. the caller should
        checkpoint the model), False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
