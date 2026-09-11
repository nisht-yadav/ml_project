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
│   │   └── baseline_cnn.py     ← Simple CNN (no freq features)
│   ├── explainability/
│   │   └── gradcam.py          ← Grad-CAM heatmap generation
│   └── utils/
│       ├── metrics.py          ← Accuracy, F1, ROC-AUC, confusion matrix
│       ├── visualize.py        ← Plotting utilities
│       ├── logger.py           ← Logging setup
│       └── helpers.py          ← General helpers (seed, config loader)
├── app/
│   └── app.py                  ← Streamlit web interface
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
streamlit run app/app.py
```

---

## Tech Stack

| Component          | Library                         |
|--------------------|---------------------------------|
| Deep Learning      | PyTorch + timm                  |
| Image Processing   | OpenCV, Pillow                  |
| Frequency Features | SciPy (DCT), OpenCV (ELA)       |
| Explainability     | grad-cam                        |
| Augmentation       | Albumentations                  |
| Web Interface      | Streamlit                       |
| Experiment Logs    | TensorBoard                     |
| Scientific Compute | NumPy, SciPy, scikit-learn      |

---

## References

1. J. Dong, W. Wang, T. Tan — *CASIA Image Tampering Detection Evaluation Database*, IEEE 2013  
2. R. Selvaraju et al. — *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017  
3. [OpenCV Docs](https://docs.opencv.org/) | [PyTorch Docs](https://pytorch.org/docs/) | [timm](https://huggingface.co/docs/timm)
