# Image Tampering Detection
## Deep Learning · Frequency Domain Analysis · Explainable AI

---

## Project Overview
A CNN-based classifier that predicts whether an image has been tampered with.  
It combines raw image features with **ELA** (Error Level Analysis) and **DCT** frequency coefficients, and uses **Grad-CAM** to visually explain which regions of the image influenced the prediction.

---

## Quick Start

### 1. Clone & enter the project
```bash
git clone https://github.com/nisht-yadav/ml_project
cd ml_project
```

### 2. Create and activate the virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**WSL / Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

**CPU only:**
```bash
pip install -r requirements.txt
```

**GPU (CUDA 12.1) — recommended for training:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Copy and configure environment
```bash
cp .env.example .env
# Edit .env as needed
```

---

## Dataset Setup

See **[Dataset Installation Guide](#dataset-installation)** below.

---

## Project Structure

```
ml_project/
├── .venv/                      ← Python virtual environment (not committed)
├── data/
│   ├── raw/
│   │   ├── authentic/          ← Place original real images here
│   │   └── tampered/           ← Place tampered/forged images here
│   └── processed/              ← Auto-generated train/val/test splits
├── models/
│   ├── saved/                  ← Final trained model weights (.pt)
│   └── checkpoints/            ← Epoch checkpoints during training
├── notebooks/                  ← Jupyter notebooks for exploration
├── results/
│   ├── plots/                  ← Training curves, confusion matrices
│   ├── predictions/            ← Sample predictions with heatmaps
│   └── reports/                ← Evaluation metric reports
├── scripts/
│   ├── prepare_dataset.py      ← Preprocess & split the raw dataset
│   ├── train.py                ← Launch model training
│   ├── evaluate.py             ← Run evaluation on the test set
│   └── predict.py              ← Run inference on a single image
├── src/
│   ├── data/
│   │   ├── dataset.py          ← PyTorch Dataset class
│   │   ├── dataloader.py       ← DataLoader builder
│   │   └── preprocessing.py   ← Resize, normalise helpers
│   ├── features/
│   │   ├── ela.py              ← Error Level Analysis
│   │   └── dct_features.py     ← DCT coefficient extraction
│   ├── models/
│   │   ├── detector.py         ← Dual-branch CNN (image + freq)
│   │   ├── baseline_cnn.py     ← Simple CNN (no freq features)
│   │   ├── resnet_model.py     ← ResNet50 image branch (128-d embedding)
│   │   └── tamper_resnet50.py  ← Config-driven ResNet50 tamper classifier
│   ├── explainability/
│   │   └── gradcam.py          ← Grad-CAM heatmap generation
│   ├── inference/
│   │   └── predictor.py        ← Load a checkpoint, run one image through it
│   └── utils/
│       ├── metrics.py          ← Accuracy, F1, ROC-AUC, confusion matrix
│       ├── visualize.py        ← Plotting utilities
│       ├── logger.py           ← Logging setup
│       └── helpers.py          ← General helpers (seed, config loader)
├── app/
│   ├── app.py                  ← Flask web server and routes
│   ├── pipeline.py             ← End-to-end analysis for one upload
│   └── config.py               ← App config (checkpoint path, ELA/DCT settings) from env vars
├── tests/                      ← Unit tests
├── logs/                       ← Training logs & TensorBoard events
├── config.yaml                 ← Central configuration file
├── requirements.txt            ← Python dependencies
├── .env.example                ← Environment variable template
└── main.py                     ← CLI entry point
```

---

## Dataset Installation

### Option A — CASIA 2.0 (Recommended)

CASIA 2.0 is the standard benchmark for image forensics.

1. **Register & download** from the official source:  
   [https://forensics.idealtest.org/](https://forensics.idealtest.org/)  
   *(Free academic registration required)*

2. **Extract** the downloaded archive.

3. **Copy images** into the project:
   ```
   data/raw/authentic/   ← All images from the "Au" folder
   data/raw/tampered/    ← All images from the "Tp" folder
   ```

4. **Run preprocessing** (once data is in place):
   ```bash
   python scripts/prepare_dataset.py
   ```

### Option B — COVERAGE Dataset

Download from: [https://github.com/wenbihan/coverage](https://github.com/wenbihan/coverage)

Place:
- Authentic images → `data/raw/authentic/`
- Tampered images  → `data/raw/tampered/`

### Option C — NC2016 (NIST Nimble Challenge 2016)

Download from: [https://www.nist.gov/itl/iad/mig/nimble-challenge-2017-evaluation](https://www.nist.gov/itl/iad/mig/nimble-challenge-2017-evaluation)

---

## Running the Project

```bash
# 1. Train the model
python scripts/train.py

# 2. Evaluate on the test set
python scripts/evaluate.py

# 3. Predict on a single image
python scripts/predict.py --image path/to/image.jpg

# 4. Launch the web interface
python app/app.py
```

---

## Web App — Tampering Detection Demo

`app/app.py` is a Flask app: upload one image and see, on one page,
the normalized image, the predicted label + confidence, a **Grad-CAM**
overlay explaining the CNN's decision, an **ELA** heatmap, and a **DCT**
frequency-energy heatmap. Its analysis logic lives in `app/pipeline.py`,
which is independent of Flask.

Every upload is first **normalized** to match exactly what the training
dataset went through (`scripts/normalize_dataset.py`, mirrored in
`src/inference/normalize.py`): converted to plain 8-bit RGB (compositing
away alpha, rescaling 16-bit/float TIFFs, etc.) and re-saved as JPEG at
a fixed quality — *before* the model, Grad-CAM, ELA, or DCT ever see it.
Without this, a PNG or 16-bit TIFF upload would hand the model pixel
data in a representation it never saw during training.

### Running it

```bash
python app/app.py
```

Then open **http://localhost:8501** (host and port are set by `APP_HOST` / `APP_PORT`).
The checkpoint is loaded once at startup; if it can't be loaded, the page shows the
error and `/health` reports `degraded` instead of the server crashing.

### Using your own model weights

Model weights are not stored in this repository (`*.pth` is git-ignored), so
after cloning you need a checkpoint. Any checkpoint saved by
`scripts/train_tamper_resnet50.py` works. The architecture, input size, and
decision threshold are read from the checkpoint itself, so you don't change code.

1. Put the file anywhere, e.g. `models/saved/my_model.pth`.
2. Point the app at it, either way:
   ```bash
   python app/app.py --checkpoint models/saved/my_model.pth    # one-off
   ```
   or in `.env`:
   ```
   MODEL_CHECKPOINT_PATH=models/saved/my_model.pth
   ```

If the checkpoint is missing or invalid, the page and `/health` say so and
explain how to fix it, rather than crashing.

### Configuring it

The model checkpoint path, ELA JPEG quality, and DCT block size are set
via environment variables (or a `.env` file in the project root — copy
`.env.example` to `.env` and edit it), never hardcoded in the app code
or editable from the web UI itself:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CHECKPOINT_PATH` | `models/saved/tamper_resnet50_full_finetune_resumed_b96_best.pth` | Which trained checkpoint the app loads at startup. |
| `ELA_QUALITY` | `90` | JPEG quality used to re-compress images for Error Level Analysis (1-100). |
| `DCT_BLOCK_SIZE` | `8` | Block size (pixels) used for block-wise DCT frequency analysis. |
| `NORMALIZE_JPEG_QUALITY` | `90` | JPEG quality every upload is re-saved at during normalization — should match what the checkpoint was trained on. |
| `FIX_IMAGE_ORIENTATION` | `false` | Whether to apply EXIF orientation correction during normalization (matches the training data's own default). |
| `INFERENCE_THRESHOLD` | *(unset → use checkpoint's own)* | Optional override of the tampered/authentic decision boundary, no retraining needed. |

If the checkpoint fails to load, or an uploaded file isn't a readable
image, the app shows an error message instead of crashing.

### How the pieces fit together

The app is a thin layer over independently testable, independently
callable modules — none of them import Flask or know about each
other:

- `src/inference/normalize.py` — normalizes an upload to match the
  training dataset's format exactly (8-bit RGB, re-saved as JPEG at a
  fixed quality), mirroring `scripts/normalize_dataset.py`.
- `src/inference/predictor.py` — loads a checkpoint and runs one image
  through the model (`load_model`, `predict_image`).
- `src/explainability/gradcam.py` — Grad-CAM overlay for a loaded model +
  image, built on the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library.
- `src/features/ela.py` — Error Level Analysis. Pure image processing,
  no model involved.
- `src/features/dct_features.py` — block-wise DCT frequency analysis.
  Pure image processing, no model involved.

Each has its own tests under `tests/` (`test_normalize.py`,
`test_predictor.py`, `test_gradcam.py`, `test_features.py`) and can be
imported and called on its own, e.g. from a notebook or a batch script,
independent of the web app.

---

## Understanding the Explanations (Grad-CAM, ELA, DCT)

The web app shows three different signals next to each other on
purpose — they answer different questions and can disagree with each
other, which is itself useful information:

**Grad-CAM** highlights the pixels that most influenced *this specific
model's* decision, by tracing gradients back from its output to its
last convolutional layer. It explains the CNN, not the image — if the
model gets the label wrong, Grad-CAM will still confidently point at
whatever it was looking at, not at "the real tampered region."

**ELA (Error Level Analysis)** re-saves the image as JPEG at a fixed
quality and measures how much each pixel changed. JPEG compresses in
8x8 blocks and reaches a different "error minimum" depending on how
many times, and at what quality, a region has already been compressed.
A region that was spliced in from a different image, or locally edited
and re-saved, often carries a different compression history than the
rest of the photo — so it lights up brighter or dimmer than its
surroundings under a fresh, uniform recompression. A single-generation,
uniformly-compressed photo tends to show a flat, low response
everywhere. ELA is plain image processing: no model, no training, just
pixel math.

**DCT (frequency analysis)** splits the image into fixed-size blocks
and measures how much high-frequency detail (fine edges/texture, as
opposed to broad flat color) each block contains. Blocks with unusually
high or unusually low high-frequency energy relative to their
neighbors can indicate re-compression (which tends to smooth away high
frequencies) or a spliced-in region with sharp edges introduced by
blending. Also plain image processing: no model, no training.

Neither ELA nor DCT is a tampering *detector* on its own — they're
forensic signals a human reviewer (or a future model) can read
alongside Grad-CAM. Because they don't depend on the CNN at all, they
can flag something suspicious even when the model's own explanation
looks unremarkable, and they can't be fooled by whatever pattern the
model specifically learned to key off of.

---

## Tech Stack

| Component          | Library                         |
|--------------------|---------------------------------|
| Deep Learning      | PyTorch + timm                  |
| Image Processing   | OpenCV, Pillow                  |
| Frequency Features | SciPy (DCT), OpenCV (ELA)       |
| Explainability     | grad-cam                        |
| Augmentation       | Albumentations                  |
| Web Interface      | Flask                           |
| Experiment Logs    | TensorBoard                     |
| Scientific Compute | NumPy, SciPy, scikit-learn      |

---

## References

1. J. Dong, W. Wang, T. Tan — *CASIA Image Tampering Detection Evaluation Database*, IEEE 2013  
2. R. Selvaraju et al. — *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017  
3. [OpenCV Docs](https://docs.opencv.org/) | [PyTorch Docs](https://pytorch.org/docs/) | [timm](https://huggingface.co/docs/timm)
