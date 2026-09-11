```powershell
# =============================================================================
# setup_windows.ps1 — Windows PowerShell environment setup
# Image Tampering Detection Project
#
# Usage:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # one-time
#   .\setup_windows.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"
$PythonMinMajor = 3
$PythonMinMinor = 10

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Image Tampering Detection — Windows Environment Setup"    -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 1. Check Python
# -----------------------------------------------------------------------------

Write-Host "`n[1/6] Checking Python version..." -ForegroundColor Yellow

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue

if (-not $pythonCommand) {
    Write-Host ""
    Write-Host "ERROR: Python was not found." -ForegroundColor Red
    Write-Host "Install Python 3.10 or newer from:" -ForegroundColor Red
    Write-Host "https://www.python.org/downloads/" -ForegroundColor White
    exit 1
}

$pythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Unable to determine Python version."
    exit 1
}

$versionParts = $pythonVersion.Split(".")
$pythonMajor = [int]$versionParts[0]
$pythonMinor = [int]$versionParts[1]

Write-Host "Found Python $pythonVersion" -ForegroundColor Green

if (
    ($pythonMajor -lt $PythonMinMajor) -or
    (($pythonMajor -eq $PythonMinMajor) -and ($pythonMinor -lt $PythonMinMinor))
) {
    Write-Host ""
    Write-Host "ERROR: Python $PythonMinMajor.$PythonMinMinor or newer is required." -ForegroundColor Red
    Write-Host "Found Python $pythonVersion." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Python version is supported." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------------------------------------

Write-Host "`n[2/6] Creating virtual environment..." -ForegroundColor Yellow

if (Test-Path $VenvDir) {
    Write-Host "Virtual environment already exists at .\$VenvDir" -ForegroundColor Green
}
else {
    python -m venv $VenvDir

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }

    Write-Host "Virtual environment created at .\$VenvDir" -ForegroundColor Green
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment Python executable was not found."
    exit 1
}

# -----------------------------------------------------------------------------
# 3. Upgrade pip and packaging tools
# -----------------------------------------------------------------------------

Write-Host "`n[3/6] Upgrading pip, setuptools and wheel..." -ForegroundColor Yellow

& $VenvPython -m pip install --upgrade pip setuptools wheel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip/setuptools/wheel."
    exit 1
}

Write-Host "[OK] Packaging tools updated." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 4. Install PyTorch
# -----------------------------------------------------------------------------

Write-Host "`n[4/6] Installing PyTorch..." -ForegroundColor Yellow

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {

    Write-Host "NVIDIA GPU detected." -ForegroundColor Green

    try {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)

        if ($gpuName) {
            Write-Host "GPU: $gpuName" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "Could not determine GPU name." -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "Installing PyTorch from the standard Python package index." -ForegroundColor White
    Write-Host "PyTorch will use CUDA if the installed package and NVIDIA driver support it." -ForegroundColor White

    & $VenvPython -m pip install torch torchvision torchaudio

}
else {

    Write-Host "No NVIDIA GPU detected." -ForegroundColor DarkYellow
    Write-Host "Installing PyTorch." -ForegroundColor White

    & $VenvPython -m pip install torch torchvision torchaudio
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyTorch installation failed."
    exit 1
}

Write-Host "[OK] PyTorch installed." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 5. Install remaining project dependencies
# -----------------------------------------------------------------------------

Write-Host "`n[5/6] Installing project dependencies..." -ForegroundColor Yellow

$RequirementsFile = "requirements.txt"

if (-not (Test-Path $RequirementsFile)) {
    Write-Host ""
    Write-Host "ERROR: requirements.txt was not found." -ForegroundColor Red
    Write-Host "Run this script from the project root directory." -ForegroundColor Red
    exit 1
}

& $VenvPython -m pip install -r $RequirementsFile

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install project dependencies."
    exit 1
}

Write-Host "[OK] Project dependencies installed." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 6. Register Jupyter kernel
# -----------------------------------------------------------------------------

Write-Host "`n[6/6] Registering Jupyter kernel..." -ForegroundColor Yellow

& $VenvPython -m pip install ipykernel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install ipykernel."
    exit 1
}

& $VenvPython -m ipykernel install --user `
    --name tamper_detect `
    --display-name "Python (tamper_detect)"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to register Jupyter kernel."
    exit 1
}

Write-Host "[OK] Jupyter kernel registered." -ForegroundColor Green

# -----------------------------------------------------------------------------
# Final verification
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Verifying PyTorch installation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& $VenvPython -c @"
import torch

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU device      :", torch.cuda.get_device_name(0))
    print("CUDA version    :", torch.version.cuda)
else:
    print("GPU device      : N/A")
    print("Running on CPU")
"@

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyTorch verification failed."
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "To verify the complete environment:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\python.exe verify_env.py" -ForegroundColor Green

Write-Host ""
Write-Host 'Jupyter kernel: "Python (tamper_detect)"' -ForegroundColor White

Write-Host ""
```
