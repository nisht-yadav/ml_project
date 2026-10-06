"""
Turns the raw dataset folder into train/val/test DataLoaders.

Split strategy (70% / 15% / 15%):
    1. Scan disk once -> list of (path, label) samples (src/data/dataset.py).
    2. Stratified split on that list, with a fixed random seed, so each
       split keeps approximately the same authentic/tampered ratio as
       the full dataset.
    3. Build one ImagePathDataset per split, each with its OWN transform
       (augmentation for train, plain resize/normalize for val & test).

Why the test set must stay completely unseen during training:
    The test set exists to estimate how the model performs on images it
    has never encountered - a stand-in for "new data in the real world".
    If test images leak into training (directly, or indirectly through
    hyperparameter/early-stopping decisions), the reported accuracy
    becomes optimistic and no longer predicts real-world performance.
    That is also why early stopping and checkpoint selection use the
    VALIDATION set, never the test set.
"""

import json
import os
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import ImagePathDataset, LABEL_TO_CLASS, build_dataset_index
from src.data.preprocessing import get_eval_transforms, get_train_transforms

Sample = Tuple[str, int]


def _stratified_split(
    samples: List[Sample], val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    labels = [label for _, label in samples]

    # First peel off train vs (val+test), then split (val+test) in half.
    temp_ratio = val_ratio + test_ratio
    train_samples, temp_samples, _, temp_labels = train_test_split(
        samples, labels,
        test_size=temp_ratio,
        random_state=seed,
        stratify=labels,
    )

    relative_test_ratio = test_ratio / temp_ratio
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_labels,
    )

    return train_samples, val_samples, test_samples


def _print_split_summary(name: str, samples: List[Sample]) -> None:
    counts = {}
    for _, label in samples:
        class_name = LABEL_TO_CLASS[label]
        counts[class_name] = counts.get(class_name, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  {name:<6s}: {len(samples):5d} images  ({breakdown})")


def _save_split(split_path: str, train, val, test) -> None:
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)
    print(f"  Split saved to {split_path} (so evaluate.py reuses the exact same test set)")


def create_dataloaders(
    dataset_dir: str,
    img_size: int = 128,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    split_save_path: str = "data/processed/split.json",
    augment: bool = True,
    hflip: bool = True,
    random_crop: bool = True,
    color_jitter: bool = True,
    jpeg_recompression: bool = False,
    jpeg_quality_range=(30, 90),
):
    """
    Returns: train_loader, val_loader, test_loader, class_names

    augment/hflip/random_crop/color_jitter/jpeg_recompression/
    jpeg_quality_range: forwarded to get_train_transforms (see
    src/data/preprocessing.py) so the training augmentation pipeline is
    config-driven rather than hardcoded here. val/test always use
    get_eval_transforms (deterministic), regardless of these flags.
    """
    print(f"Scanning dataset at '{dataset_dir}' ...")
    samples, _corrupted = build_dataset_index(dataset_dir)

    min_class_count = min(
        sum(1 for _, l in samples if l == label) for label in LABEL_TO_CLASS
    )
    if min_class_count < 3:
        raise ValueError(
            "Not enough images in one of the classes to create a stratified "
            "70/15/15 split (need at least 3 per class). Add more images."
        )

    train_samples, val_samples, test_samples = _stratified_split(
        samples, val_ratio, test_ratio, seed
    )

    print("Dataset split (70% train / 15% val / 15% test, fixed seed "
          f"{seed}):")
    _print_split_summary("train", train_samples)
    _print_split_summary("val", val_samples)
    _print_split_summary("test", test_samples)

    if split_save_path:
        _save_split(split_save_path, train_samples, val_samples, test_samples)

    # Train gets augmentation; val/test get the plain deterministic transform.
    train_ds = ImagePathDataset(train_samples, transform=get_train_transforms(
        img_size,
        augment=augment,
        hflip=hflip,
        random_crop=random_crop,
        color_jitter=color_jitter,
        jpeg_recompression=jpeg_recompression,
        jpeg_quality_range=jpeg_quality_range,
    ))
    val_ds = ImagePathDataset(val_samples, transform=get_eval_transforms(img_size))
    test_ds = ImagePathDataset(test_samples, transform=get_eval_transforms(img_size))

    pin_memory = True  # harmless on CPU-only runs, speeds up CUDA transfers

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    class_names = [LABEL_TO_CLASS[i] for i in range(len(LABEL_TO_CLASS))]
    return train_loader, val_loader, test_loader, class_names
