"""
Dataset loading for the baseline CNN.

Two things live here, deliberately kept separate:

1. `build_dataset_index(...)` - walks the `authentic/` and `tampered/`
   folders on disk and returns a plain list of (path, label) tuples,
   verifying that every file is actually a readable image.

2. `ImagePathDataset` - a small torch Dataset that takes such a list
   (plus a torchvision transform) and yields (tensor, label) pairs.

Splitting the "which files exist and are valid" step from the
"turn a file list into tensors" step means the *same* file list can be
handed to three different `ImagePathDataset` instances (train/val/test),
each with its own transform - which is exactly what avoids data leakage
between splits (see src/data/dataloader.py).
"""

import os
from typing import List, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Class -> numeric label mapping used everywhere in this project.
# tampered = 1 (positive class), authentic = 0 (negative class).
CLASS_TO_LABEL = {"authentic": 0, "tampered": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

Sample = Tuple[str, int]


def _scan_class_folder(folder: str, label: int) -> Tuple[List[Sample], List[str]]:
    """
    List every valid image in `folder` and assign it `label`.
    Files that aren't images, or that PIL can't open, are skipped and
    reported back instead of crashing the whole loading process.
    """
    samples: List[Sample] = []
    corrupted: List[str] = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Expected class folder not found: {folder}")

    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(fname)[1].lower() not in IMG_EXTENSIONS:
            continue

        try:
            # Image.verify() checks the file is a valid, uncorrupted image
            # without decoding the full pixel data (cheap sanity check).
            with Image.open(path) as img:
                img.verify()
            samples.append((path, label))
        except (UnidentifiedImageError, OSError):
            corrupted.append(path)

    return samples, corrupted


def build_dataset_index(dataset_dir: str) -> Tuple[List[Sample], List[str]]:
    """
    Scan `dataset_dir/authentic` and `dataset_dir/tampered`.

    Returns:
        samples:   list of (image_path, label) for every valid image
        corrupted: list of paths that could not be read (reported, not raised)
    """
    all_samples: List[Sample] = []
    all_corrupted: List[str] = []

    for class_name, label in CLASS_TO_LABEL.items():
        folder = os.path.join(dataset_dir, class_name)
        samples, corrupted = _scan_class_folder(folder, label)
        all_samples.extend(samples)
        all_corrupted.extend(corrupted)

        print(f"  {class_name:<10s} (label={label}): {len(samples)} usable images"
              f"{f', {len(corrupted)} unreadable' if corrupted else ''}")

    if all_corrupted:
        print(f"  WARNING: {len(all_corrupted)} unreadable file(s) were skipped:")
        for p in all_corrupted[:10]:
            print(f"    - {p}")
        if len(all_corrupted) > 10:
            print(f"    ... and {len(all_corrupted) - 10} more")

    if not all_samples:
        raise RuntimeError(
            f"No usable images found under '{dataset_dir}'. "
            f"Expected '{dataset_dir}/authentic' and '{dataset_dir}/tampered' "
            f"to contain image files."
        )

    return all_samples, all_corrupted


class ImagePathDataset(Dataset):
    """
    Minimal Dataset: given a fixed list of (path, label) samples and a
    torchvision transform, loads and transforms images on demand.

    Kept generic (no knowledge of "train" vs "val" vs "test") so the
    same class works for every split - only the `transform` passed in
    differs between them.
    """

    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # Force RGB: some tampered/authentic sets contain grayscale or
        # CMYK/PNG-with-alpha images. The CNN expects a fixed 3-channel input.
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Shape (1,) float tensor: matches the (batch, 1) logits produced by
        # the model's single output neuron, so it plugs straight into
        # BCEWithLogitsLoss without any reshaping in the training loop.
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return image, label_tensor
