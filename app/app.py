"""
Flask web interface for the image tampering detector.

Upload one image and get, on one page: the normalized image, the
prediction (label + confidence), a Grad-CAM overlay explaining the CNN's
decision, an ELA heatmap, and a DCT frequency-energy heatmap.

Configuration (checkpoint path, ELA quality, DCT block size, normalize
quality, host/port, upload limit) comes from app/config.py - environment
variables or .env - never from user input.

Run from the project root:
    python app/app.py
then open http://127.0.0.1:8501
"""

import argparse
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    APP_HOST,
    APP_PORT,
    DCT_BLOCK_SIZE,
    ELA_QUALITY,
    FIX_IMAGE_ORIENTATION,
    INFERENCE_THRESHOLD_OVERRIDE,
    MAX_UPLOAD_MB,
    MODEL_CHECKPOINT_PATH,
    NORMALIZE_JPEG_QUALITY,
)
from app.pipeline import Analyzer, ImageInputError  # noqa: E402
from src.inference.predictor import ModelLoadError  # noqa: E402

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _settings(checkpoint_path: str) -> dict:
    return {
        "checkpoint": checkpoint_path,
        "ela_quality": ELA_QUALITY,
        "dct_block_size": DCT_BLOCK_SIZE,
        "normalize_quality": NORMALIZE_JPEG_QUALITY,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


def create_app(checkpoint_path: str = MODEL_CHECKPOINT_PATH) -> Flask:
    checkpoint_path = str(checkpoint_path)
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    analyzer = None
    model_error = None
    try:
        analyzer = Analyzer(
            checkpoint_path=checkpoint_path,
            ela_quality=ELA_QUALITY,
            dct_block_size=DCT_BLOCK_SIZE,
            normalize_quality=NORMALIZE_JPEG_QUALITY,
            fix_orientation=FIX_IMAGE_ORIENTATION,
            threshold_override=INFERENCE_THRESHOLD_OVERRIDE,
        )
    except ModelLoadError as exc:
        model_error = (
            f"Could not load the model checkpoint. {exc} "
            "Point the app at your own weights with MODEL_CHECKPOINT_PATH in .env, "
            "or start it with --checkpoint <file>."
        )

    settings = _settings(checkpoint_path)
    settings["threshold"] = (
        analyzer.threshold if analyzer is not None else INFERENCE_THRESHOLD_OVERRIDE
    )

    def render(result=None, error=None, status=200):
        return render_template(
            "index.html", result=result, error=error, settings=settings, model_error=model_error
        ), status

    @app.get("/")
    def index():
        return render()

    @app.post("/analyze")
    def analyze():
        if analyzer is None:
            return render(error=model_error, status=503)

        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            return render(error="Please choose an image to upload.", status=400)

        extension = Path(upload.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return render(
                error=f"Unsupported file type '{extension or 'none'}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
                status=415,
            )

        try:
            result = analyzer.analyze(upload.read())
        except ImageInputError as exc:
            return render(error=str(exc), status=422)

        return render(result=result)

    @app.get("/health")
    def health():
        return jsonify(
            status="ok" if analyzer is not None else "degraded",
            model_loaded=analyzer is not None,
            model_error=model_error,
            checkpoint=checkpoint_path,
        ), (200 if analyzer is not None else 503)

    @app.errorhandler(413)
    def too_large(_exc):
        return render(error=f"File is too large. The limit is {MAX_UPLOAD_MB} MB.", status=413)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Image tampering detection web app.")
    parser.add_argument("--checkpoint", default=MODEL_CHECKPOINT_PATH,
                        help="Path to a TamperResNet50 checkpoint (overrides MODEL_CHECKPOINT_PATH).")
    parser.add_argument("--host", default=APP_HOST, help="Interface to bind (default: APP_HOST).")
    parser.add_argument("--port", type=int, default=APP_PORT, help="Port to serve on (default: APP_PORT).")
    args = parser.parse_args()
    create_app(checkpoint_path=args.checkpoint).run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
