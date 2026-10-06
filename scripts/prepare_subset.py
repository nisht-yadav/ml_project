"""
Builds data/subset/ from data/raw/: a class-rebalanced training set used
by several experiments (see experiments/*/summary.json's "dataset_dir").

data/raw is ~7437 authentic vs. 2064 tampered (~3.6:1) - this script
keeps ALL tampered images and randomly samples 3000 of the authentic
ones (seed=42, so the exact same 3000 files are picked every time this
is run), giving a milder ~1.45:1 ratio without discarding any tampered
images (the minority/harder class).

This script exists so data/subset/ can be safely deleted to save disk
space and regenerated on demand - it is fully reproducible from
data/raw/ alone, nothing about it depends on files outside this repo.

Run from the project root:
    python scripts/prepare_subset.py
"""

import os
import random
import shutil

RAW_DIR = os.path.join("data", "raw")
SUBSET_DIR = os.path.join("data", "subset")
NUM_AUTHENTIC_SAMPLED = 3000
SEED = 42


def main():
    random.seed(SEED)

    src_authentic = os.path.join(RAW_DIR, "authentic")
    src_tampered = os.path.join(RAW_DIR, "tampered")
    dst_authentic = os.path.join(SUBSET_DIR, "authentic")
    dst_tampered = os.path.join(SUBSET_DIR, "tampered")

    if not os.path.isdir(src_authentic) or not os.path.isdir(src_tampered):
        raise FileNotFoundError(
            f"Expected {src_authentic} and {src_tampered} to exist. "
            f"Place the downloaded dataset under {RAW_DIR}/authentic and {RAW_DIR}/tampered first."
        )

    os.makedirs(dst_authentic, exist_ok=True)
    os.makedirs(dst_tampered, exist_ok=True)

    authentic_files = sorted(f for f in os.listdir(src_authentic) if f.lower().endswith(".jpg"))
    tampered_files = sorted(f for f in os.listdir(src_tampered) if f.lower().endswith(".jpg"))

    print(f"data/raw: {len(authentic_files)} authentic, {len(tampered_files)} tampered")

    if len(authentic_files) < NUM_AUTHENTIC_SAMPLED:
        raise ValueError(
            f"Only {len(authentic_files)} authentic images available, "
            f"need at least {NUM_AUTHENTIC_SAMPLED}."
        )

    sampled_authentic = random.sample(authentic_files, NUM_AUTHENTIC_SAMPLED)

    for fname in sampled_authentic:
        shutil.copy2(os.path.join(src_authentic, fname), os.path.join(dst_authentic, fname))
    for fname in tampered_files:
        shutil.copy2(os.path.join(src_tampered, fname), os.path.join(dst_tampered, fname))

    print(f"data/subset: copied {len(sampled_authentic)} authentic (sampled, seed={SEED}) "
          f"+ {len(tampered_files)} tampered (all)")


if __name__ == "__main__":
    main()
