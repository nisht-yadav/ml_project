```bash
#!/usr/bin/env bash
# =============================================================================
# setup_wsl.sh — WSL / Linux / macOS environment setup
# Image Tampering Detection Project
#
# Usage:
#   chmod +x setup_wsl.sh
#   ./setup_wsl.sh
# =============================================================================

set -euo pipefail

VENV_DIR=".venv"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

echo "============================================================"
echo "  Image Tampering Detection — Environment Setup"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Check Python
# -----------------------------------------------------------------------------

echo ""
echo "[1/6] Checking Python version..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found."
    echo ""
    echo "On Ubuntu/WSL, install it with:"
    echo "  sudo apt update"
    echo "  sudo apt install python3 python3-venv python3-pip -y"
    exit 1
fi

PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

echo "Found Python ${PY_MAJOR}.${PY_MINOR}"

if (( PY_MAJOR < PYTHON_MIN_MAJOR ||
      (PY_MAJOR == PYTHON_MIN_MAJOR && PY_MINOR < PYTHON_MIN_MINOR) )); then
    echo "ERROR: Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR} or newer is required."
    echo "Found Python ${PY_MAJOR}.${PY_MINOR}."
    exit 1
fi

echo "[OK] Python version is supported."

# -----------------------------------------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------------------------------------

echo ""
echo "[2/6] Creating virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at ./$VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at ./$VENV_DIR"
fi

# -----------------------------------------------------------------------------
# 3. Activate virtual environment and upgrade packaging tools
# -----------------------------------------------------------------------------

echo ""
echo "[3/6] Activating virtual environment..."

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Using:"
echo "  Python: $(python --version)"
echo "  Pip:    $(python -m pip --version)"

echo ""
echo "Upgrading pip, setuptools and wheel..."

python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# 4. Install PyTorch
# -----------------------------------------------------------------------------

echo ""
echo "[4/6] Installing PyTorch..."

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected."

    echo ""
    echo "Installing CUDA-enabled PyTorch."
    echo "The PyTorch installer will determine the compatible packages."

    python -m pip install torch torchvision torchaudio

else
    echo "No NVIDIA GPU detected."
    echo "Installing PyTorch using the standard package index."

    python -m pip install torch torchvision torchaudio
fi

# -----------------------------------------------------------------------------
# 5. Install remaining project dependencies
# -----------------------------------------------------------------------------

echo ""
echo "[5/6] Installing project dependencies..."

if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found."
    echo "Run this script from the project root directory."
    exit 1
fi

python -m pip install -r requirements.txt

# -----------------------------------------------------------------------------
# 6. Register Jupyter kernel
# -----------------------------------------------------------------------------

echo ""
echo "[6/6] Registering Jupyter kernel..."

python -m pip install ipykernel

python -m ipykernel install \
    --user \
    --name tamper_detect \
    --display-name "Python (tamper_detect)"

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Verifying PyTorch installation"
echo "============================================================"

python -c "
import torch

print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU device      :', torch.cuda.get_device_name(0))
    print('CUDA version    :', torch.version.cuda)
else:
    print('GPU device      : N/A')
    print('Running on CPU')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment later:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify the environment:"
echo "  python verify_env.py"
echo ""
echo "Jupyter kernel:"
echo '  "Python (tamper_detect)"'
echo ""
```
