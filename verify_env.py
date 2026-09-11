"""
verify_env.py  —  Environment verification script
Run: python verify_env.py
"""
import sys
import importlib

print(f"Python: {sys.version}")
print()

# (package_import_name, display_name)
CORE = [
    ("torch",          "torch"),
    ("torchvision",    "torchvision"),
    ("cv2",            "opencv-python"),
    ("numpy",          "numpy"),
    ("scipy",          "scipy"),
    ("sklearn",        "scikit-learn"),
    ("pandas",         "pandas"),
    ("matplotlib",     "matplotlib"),
    ("seaborn",        "seaborn"),
    ("tqdm",           "tqdm"),
    ("timm",           "timm"),
    ("albumentations", "albumentations"),
    ("yaml",           "PyYAML"),
    ("PIL",            "Pillow"),
    ("dotenv",         "python-dotenv"),
    ("click",          "click"),
    ("requests",       "requests"),
]

# Checked separately (heavy / slow)
HEAVY = [
    ("tensorboard",    "tensorboard"),
    ("streamlit",      "streamlit"),
]

print("=== Core packages ===")
failed = []
for mod, name in CORE:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  [OK]   {name:<25} {ver}")
    except Exception as e:
        print(f"  [FAIL] {name:<25} {e}")
        failed.append(name)

print()
print("=== Heavy packages (imported separately) ===")
for mod, name in HEAVY:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  [OK]   {name:<25} {ver}")
    except Exception as e:
        print(f"  [FAIL] {name:<25} {e}")
        failed.append(name)

print()
# PyTorch GPU check
try:
    import torch
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "N/A"
    print(f"  CUDA available : {cuda}")
    print(f"  GPU device     : {device_name}")
    print(f"  PyTorch build  : {torch.version.cuda or 'CPU-only'}")
except Exception as e:
    print(f"  GPU check failed: {e}")

print()
if failed:
    print(f"[WARN] Failed packages: {failed}")
else:
    print("[SUCCESS] All packages verified!")
