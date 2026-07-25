#!/usr/bin/env python3
"""Verify the Python-side prerequisites for an Ubuntu deployment.

This check intentionally uses only the Python standard library, so it can be
run immediately after creating a virtual environment. It does not download
model weights or open camera/microphone devices.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from typing import Dict


REQUIRED_MODULES: Dict[str, str] = {
    "PySide6": "PySide6",
    "OpenCV": "cv2",
    "NumPy": "numpy",
    "SciPy": "scipy",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "sounddevice": "sounddevice",
    "MediaPipe": "mediapipe",
    "Ultralytics": "ultralytics",
    "PyQtGraph": "pyqtgraph",
    "PyOpenGL": "OpenGL",
    "SenseVoice": "sherpa_onnx",
    "PyTorch": "torch",
}


def _check_modules(modules: Dict[str, str]) -> list[str]:
    missing = []
    for label, module_name in modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(f"{label} ({module_name})")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-funasr",
        action="store_true",
        help="also require the optional FunASR backend",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    missing = _check_modules(REQUIRED_MODULES)
    if args.with_funasr:
        missing.extend(_check_modules({"FunASR": "funasr"}))

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {bool(torch.cuda.is_available())}")
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        print(f"PyTorch check failed: {exc}")

    if missing:
        print("Missing modules:")
        for item in missing:
            print(f"  - {item}")
        print("Install the dependencies listed in requirements.txt and retry.")
        return 1

    print("Python dependencies: OK")
    print("Model weights are downloaded lazily on first detector initialization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
