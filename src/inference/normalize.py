"""
Uploaded-image normalization for the web app's inference path.

The trained checkpoint was fit on a dataset that had already been run
through `scripts/normalize_dataset.py`: every source image (whatever mix
of JPEG/PNG/TIFF/BMP, 8-bit/16-bit/float, RGB/RGBA/grayscale/palette it
started as) was converted to plain 8-bit RGB and re-saved as a JPEG at a
single fixed quality, *before* any resizing or augmentation. A live
upload has had none of that done to it - it could be a 16-bit TIFF, a
PNG with alpha, or already a JPEG at some arbitrary quality - so without
this step the model would be scoring pixel data in a different
representation than anything it was ever trained/validated on.

`normalize_to_rgb` below reproduces
`scripts/normalize_dataset.py::normalize_to_rgb` exactly, rule for rule.
It's kept as an independent copy rather than an import from that script
so the inference path never depends on the dataset-prep/training scripts
(and vice versa) - see this project's convention of keeping the
inference/app layer decoupled from the training pipeline. If the
dataset-prep rules in `scripts/normalize_dataset.py` ever change, mirror
the change here too.
"""

import io

import numpy as np
from PIL import Image, ImageOps

# Matches scripts/normalize_dataset.py::JPEG_QUALITY_DEFAULT - the quality
# the training dataset's images were actually re-saved at.
DATASET_JPEG_QUALITY_DEFAULT = 90

# Modes Pillow represents with more than 8 bits per channel (16-bit and
# 32-bit-int grayscale, as produced by many scanners/cameras writing
# TIFF). These need an explicit range remap, not a plain .convert("RGB").
HIGH_BITDEPTH_INT_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "I;16S"}


def normalize_to_rgb(img: Image.Image, fix_orientation: bool = False) -> Image.Image:
    """
    Returns a new 8-bit RGB Image for any input mode/bit depth, without
    resizing:
        - RGBA/LA/palette-with-transparency -> composited onto a white
          background (dropping the alpha channel).
        - 16-bit / 32-bit-int grayscale (mode "I"/"I;16*") -> bit-depth
          reduced to 8-bit by shifting the value range down (>> 8), not
          by casting (which reinterprets the raw values as if already
          8-bit, producing a near-black image).
        - 32-bit float (mode "F") -> per-image min-max stretched to 0-255.
        - everything else (L, P, CMYK, 1, YCbCr, ...) -> plain .convert("RGB").

    `fix_orientation` defaults to False to match
    scripts/normalize_dataset.py's default (off - applying EXIF
    orientation can swap width/height on 90/270 degree rotations).
    """
    if fix_orientation:
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


def normalize_and_reencode_jpeg(
    image: Image.Image,
    quality: int = DATASET_JPEG_QUALITY_DEFAULT,
    fix_orientation: bool = False,
) -> Image.Image:
    """
    Full normalization pipeline for one uploaded image: convert to 8-bit
    RGB (`normalize_to_rgb`), then round-trip it through an actual JPEG
    encode/decode at `quality` - not just a mode conversion. This
    matters: `scripts/normalize_dataset.py` didn't just standardize the
    color mode, it also re-saved every image as a JPEG at one fixed
    quality, which is itself a lossy transformation the model was
    trained to see. Skipping the re-encode would leave (for example) a
    losslessly-decoded PNG upload with no JPEG compression artifacts at
    all - a distribution the model never saw during training.

    Returns a new PIL.Image (RGB), same pixel dimensions as the input.
    """
    rgb = normalize_to_rgb(image, fix_orientation=fix_orientation)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
