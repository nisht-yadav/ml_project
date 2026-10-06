"""
Image preprocessing / transforms for the baseline CNN.

Two separate transform pipelines are provided on purpose:

- `get_train_transforms`: letterbox-resize + config-driven augmentation
  (flip, random crop, mild color jitter, optional JPEG-recompression
  augmentation) + normalize. Augmentation is applied ONLY here.
- `get_eval_transforms`:  letterbox-resize + normalize, nothing else.

Why augmentation is train-only: the point of augmentation is to make the
model robust to variations it hasn't literally seen (flipped/cropped/
recompressed versions of training images). If we also randomly perturbed
validation or test images, we would be measuring how well the model
handles random noise, not how well it generalizes - and the same image
could score differently across evaluation runs, making early stopping /
test metrics unreliable.

`get_train_transforms` exposes every augmentation as its own on/off
flag (all default True except `jpeg_recompression`, default False) so
a config file can toggle them individually for controlled experiments,
without touching this code. `jpeg_recompression` is intentionally kept
independent of the `augment` master switch and the other flags: it
interacts with ELA-based explainability (ELA specifically measures
JPEG recompression error), so a run may want it off even while other
augmentation stays on, or vice versa.
"""

import io
import random

from PIL import Image, ImageOps
from torchvision import transforms

IMG_SIZE_DEFAULT = 224

# ImageNet mean/std. Required for the pretrained ResNet50 branch
# (src/models/resnet_model.py) to see inputs distributed the way it was
# trained on; harmless for the from-scratch BaselineCNN/DeepCNN too, since
# they learn their own weights around whatever normalization is used.
# One shared preprocessing pipeline for every model, as required.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Pillow >= 9.1 renamed Image.BILINEAR to Image.Resampling.BILINEAR.
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


class LetterboxResize:
    """
    Resizes so the LONGEST side becomes `size`, preserving aspect ratio,
    then pads the shorter side (centered) with `fill` so the final image
    is exactly `size x size`.

    A plain `transforms.Resize((size, size))` stretches/squashes the
    image to fit a square, distorting aspect ratio - which can distort
    exactly the kind of geometric artifacts (splice boundaries, cloned
    regions) a tampering detector should be learning from. Letterboxing
    avoids that distortion at the cost of some black padding.
    """

    def __init__(self, size: int = 224, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), _BILINEAR)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        fill_color = (self.fill,) * len(img.getbands())
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=fill_color)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, fill={self.fill})"


class RandomJPEGRecompression:
    """
    With probability `p`, re-encodes the image through JPEG at a random
    quality level and decodes it back - simulating the recompression
    artifacts a real tampered/re-saved image would carry, as an
    augmentation. Kept as a separate opt-in (see module docstring) since
    it directly changes the JPEG-artifact signal that ELA-based
    explainability reads.
    """

    def __init__(self, quality_range=(30, 90), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(quality_range={self.quality_range}, p={self.p})"


def get_train_transforms(
    img_size: int = IMG_SIZE_DEFAULT,
    augment: bool = True,
    hflip: bool = True,
    random_crop: bool = True,
    color_jitter: bool = True,
    jpeg_recompression: bool = False,
    jpeg_quality_range=(30, 90),
    crop_pad: int = 16,
) -> transforms.Compose:
    """
    augment: master switch for the flip/crop/rotation/color-jitter group
        below. False = deterministic resize + normalize only (same as
        get_eval_transforms), for controlled experiments that need to
        isolate some other variable from augmentation. Individual flags
        below only take effect when augment=True.
    hflip: random horizontal flip (p=0.5).
    random_crop: reflect-pads the letterboxed image by `crop_pad` pixels
        on each side, then randomly crops back to img_size x img_size -
        a mild "random crop" that doesn't reintroduce the aspect-ratio
        distortion LetterboxResize was chosen to avoid.
    color_jitter: mild brightness/contrast/saturation/hue jitter.
    jpeg_recompression: INDEPENDENT of `augment` - see module docstring.
        Applied after the geometric augmentations, before ToTensor.
    """
    ops = [LetterboxResize(img_size)]

    if augment:
        if random_crop:
            ops.append(transforms.Pad(crop_pad, padding_mode="reflect"))
            ops.append(transforms.RandomCrop(img_size))
        if hflip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if color_jitter:
            ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02))

    if jpeg_recompression:
        ops.append(RandomJPEGRecompression(quality_range=jpeg_quality_range, p=0.5))

    ops.append(transforms.ToTensor())                       # HWC [0,255] -> CHW [0,1]
    ops.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))
    return transforms.Compose(ops)


def get_eval_transforms(img_size: int = IMG_SIZE_DEFAULT) -> transforms.Compose:
    """Used for validation, test, AND single-image prediction - it must
    match exactly what the model saw at training/validation time, minus
    the randomness."""
    return transforms.Compose([
        LetterboxResize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
