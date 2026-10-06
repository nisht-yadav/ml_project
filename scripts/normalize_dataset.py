"""
Normalizes a raw image-forgery dataset into a single, uniform format
ready for the rest of this pipeline (src/data/dataset.py expects
<dataset_dir>/authentic and <dataset_dir>/tampered, each holding plain
8-bit RGB images).

What this does, per image:
    1. Load it regardless of source format (JPEG/PNG/TIFF/BMP/...).
    2. Flatten to standard 8-bit RGB:
        - RGBA/LA/palette-with-transparency -> composited onto a white
          background (dropping the alpha channel).
        - 16-bit / 32-bit-int grayscale (mode "I"/"I;16*", common in
          scanner/camera TIFFs) -> bit-depth reduced to 8-bit by
          shifting the value range down (>> 8), NOT by casting, which
          is a classic Pillow gotcha: img.convert("RGB") on a 16-bit
          image reinterprets the raw values as if they were already
          8-bit, producing a near-black image instead of a properly
          rescaled one.
        - 32-bit float (mode "F") -> per-image min-max stretched to
          0-255 (there's no fixed reference range for this mode).
        - everything else (L, P, CMYK, 1, YCbCr, ...) -> plain
          .convert("RGB").
    3. Save as JPEG at a single fixed quality (default 90). Dimensions
       are never touched - no resizing happens here.
    4. Write to <output_dir>/<authentic|tampered>/<same relative path,
       extension replaced with .jpg> - never touches the source files.

Optimized for large datasets (originally written for ~320K images):
    - A multiprocessing.Pool of worker processes does the actual
      decode/convert/encode work in parallel (CPU-bound), one process
      per core by default.
    - Already-converted outputs are skipped on a re-run (checked by
      destination path existing) unless --overwrite is passed, so an
      interrupted run can simply be restarted.
    - Per-image failures (corrupt/unreadable files) are caught, logged
      to <output_dir>/normalize_errors.log, and skipped instead of
      aborting the whole job - at this scale, a handful of bad files is
      expected and shouldn't lose hours of otherwise-good work.

Run from the project root:
    python scripts/normalize_dataset.py \\
        --authentic-dir /path/to/auth \\
        --tampered-dir /path/to/tam \\
        --output-dir data/normalized
"""

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import IMG_EXTENSIONS  # single source of truth for "what's an image"

JPEG_QUALITY_DEFAULT = 90

# Modes Pillow represents with more than 8 bits per channel (16-bit and
# 32-bit-int grayscale, as produced by many scanners/cameras writing
# TIFF). These need an explicit range remap, not a plain .convert("RGB").
HIGH_BITDEPTH_INT_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "I;16S"}


def normalize_to_rgb(img: Image.Image, fix_orientation: bool) -> Image.Image:
    """Returns a new 8-bit RGB Image for any input mode/bit depth,
    without resizing. See module docstring for the per-mode rules."""
    if fix_orientation:
        # Applies (and strips) any EXIF orientation tag so the pixel
        # data matches how the image is meant to be viewed. Opt-in
        # (default off) because it can swap width/height for 90/270
        # degree rotations, which some callers may want to avoid
        # touching at all alongside "no resizing".
        img = ImageOps.exif_transpose(img)

    mode = img.mode

    if mode in HIGH_BITDEPTH_INT_MODES:
        arr = np.asarray(img, dtype=np.int32)
        arr = np.clip(arr, 0, 65535)
        arr8 = (arr >> 8).astype(np.uint8)
        return Image.fromarray(arr8, mode="L").convert("RGB")

    if mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr8 = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
        else:
            arr8 = np.zeros(arr.shape, dtype=np.uint8)
        return Image.fromarray(arr8, mode="L").convert("RGB")

    has_alpha = mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])  # alpha channel as the paste mask
        return background

    if mode == "RGB":
        return img

    # L, P (no transparency), CMYK, 1, YCbCr, etc.
    return img.convert("RGB")


def _relative_jpg_path(src_root: Path, src_path: Path) -> Path:
    rel = src_path.relative_to(src_root)
    return rel.with_suffix(".jpg")


def _collect_tasks(src_root: Path, dst_root: Path) -> List[Tuple[Path, Path]]:
    """Walks src_root recursively (mirrors any subfolder structure into
    dst_root), returns [(src_path, dst_path), ...] for every file with a
    recognized image extension."""
    tasks = []
    for dirpath, _dirnames, filenames in os.walk(src_root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in IMG_EXTENSIONS:
                continue
            src_path = Path(dirpath) / fname
            dst_path = dst_root / _relative_jpg_path(src_root, src_path)
            tasks.append((src_path, dst_path))
    return tasks


def _process_one(args) -> Tuple[str, str, str]:
    """Runs in a worker process. Returns (status, src_path, message)
    where status is one of 'ok', 'skipped', 'error'."""
    src_path, dst_path, quality, overwrite, fix_orientation = args

    if not overwrite and dst_path.exists():
        return ("skipped", str(src_path), "")

    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            rgb = normalize_to_rgb(img, fix_orientation=fix_orientation)
            # Write to a temp path then rename: an interrupted process
            # (killed mid-write) never leaves a half-written .jpg behind
            # that a later --resume run would mistake for "already done".
            tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
            rgb.save(tmp_path, format="JPEG", quality=quality)
            os.replace(tmp_path, dst_path)
        return ("ok", str(src_path), "")
    except Exception as e:
        return ("error", str(src_path), f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}")


def normalize_split(
    src_dir: str,
    dst_dir: str,
    quality: int,
    workers: int,
    overwrite: bool,
    fix_orientation: bool,
    error_log_path: str,
) -> Tuple[int, int, int]:
    src_root = Path(src_dir)
    dst_root = Path(dst_dir)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    print(f"Scanning '{src_root}' ...")
    tasks = _collect_tasks(src_root, dst_root)
    print(f"  found {len(tasks):,} image file(s)")

    work_items = [(s, d, quality, overwrite, fix_orientation) for s, d in tasks]

    ok = skipped = errors = 0
    start = time.time()

    with Pool(processes=workers) as pool, open(error_log_path, "a") as error_log:
        for i, (status, src_path, message) in enumerate(
            pool.imap_unordered(_process_one, work_items, chunksize=64), start=1
        ):
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                error_log.write(f"{src_path}\t{message}\n")

            if i % 5000 == 0 or i == len(work_items):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                print(f"  {i:,}/{len(work_items):,} processed "
                      f"(ok={ok:,}, skipped={skipped:,}, errors={errors:,}) "
                      f"- {rate:.1f} img/s")

    return ok, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Normalize an image-forgery dataset to uniform 8-bit RGB JPEG (no resizing)."
    )
    parser.add_argument("--authentic-dir", required=True, help="Source folder of authentic images.")
    parser.add_argument("--tampered-dir", required=True, help="Source folder of tampered images.")
    parser.add_argument("--output-dir", required=True, help="Destination root; gets <output-dir>/authentic and /tampered.")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY_DEFAULT, help="JPEG quality (default: 90).")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel worker processes (default: all cores).")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess files even if the .jpg output already exists.")
    parser.add_argument(
        "--fix-orientation", action="store_true",
        help="Apply EXIF orientation before saving (off by default - can swap width/height on 90/270deg rotations).",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    error_log_path = str(output_root / "normalize_errors.log")

    print(f"Workers: {args.workers} | JPEG quality: {args.quality} | "
          f"overwrite: {args.overwrite} | fix_orientation: {args.fix_orientation}")

    grand_ok = grand_skipped = grand_errors = 0
    for label, src_dir in (("authentic", args.authentic_dir), ("tampered", args.tampered_dir)):
        print(f"\n=== {label} ===")
        dst_dir = str(output_root / label)
        ok, skipped, errors = normalize_split(
            src_dir=src_dir,
            dst_dir=dst_dir,
            quality=args.quality,
            workers=args.workers,
            overwrite=args.overwrite,
            fix_orientation=args.fix_orientation,
            error_log_path=error_log_path,
        )
        grand_ok += ok
        grand_skipped += skipped
        grand_errors += errors

    print(f"\nDone. Total: ok={grand_ok:,}, skipped={grand_skipped:,} (already existed), errors={grand_errors:,}")
    if grand_errors:
        print(f"See {error_log_path} for the list of files that failed.")


if __name__ == "__main__":
    main()
