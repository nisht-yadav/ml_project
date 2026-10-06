"""
Web app configuration, loaded from environment variables (optionally via a
`.env` file in the project root, see `.env.example`).

The model checkpoint path, ELA JPEG quality, and DCT block size are
deliberately kept OUT of the web page itself (per the project brief:
"Model checkpoint path ... ELA quality setting, and DCT block size should
be configurable (env var or config file), not hardcoded") - they're read
once here, at process startup, from the deployment environment.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # no-op if the file doesn't exist

APP_HOST: str = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT: int = int(os.environ.get("APP_PORT", "8501"))
MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "20"))


def _resolve_path(value: str) -> str:
    """Relative paths are resolved against the project root so the app
    behaves the same whether it's launched from the repo root or elsewhere."""
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


MODEL_CHECKPOINT_PATH: str = _resolve_path(
    os.environ.get(
        "MODEL_CHECKPOINT_PATH",
        "models/saved/tamper_resnet50_full_finetune_resumed_b96_best.pth",
    )
)

ELA_QUALITY: int = int(os.environ.get("ELA_QUALITY", "90"))
DCT_BLOCK_SIZE: int = int(os.environ.get("DCT_BLOCK_SIZE", "8"))

# Every image the checkpoint was trained on went through
# scripts/normalize_dataset.py first: converted to plain 8-bit RGB, then
# re-saved as JPEG at this quality (see src/inference/normalize.py). The
# app applies the identical normalization to every upload before it
# touches the model, ELA, or DCT, so live predictions see the same kind
# of pixel data training/validation did. Default matches that script's
# own default (90).
NORMALIZE_JPEG_QUALITY: int = int(os.environ.get("NORMALIZE_JPEG_QUALITY", "90"))

# Matches scripts/normalize_dataset.py's --fix-orientation flag, default
# off for the same reason: applying EXIF orientation can swap width/height
# on 90/270 degree rotated photos, which the training data was not
# corrected for either.
FIX_IMAGE_ORIENTATION: bool = os.environ.get("FIX_IMAGE_ORIENTATION", "false").strip().lower() in (
    "1", "true", "yes",
)

# Optional: override the threshold baked into the checkpoint at prediction
# time (see TamperResNet50's module docstring — this never requires
# retraining). Left unset by default, in which case the app uses whatever
# `inference_threshold` the checkpoint itself was saved with.
_threshold_env = os.environ.get("INFERENCE_THRESHOLD")
INFERENCE_THRESHOLD_OVERRIDE = float(_threshold_env) if _threshold_env else None
