"""Runtime environment: Colab vs. local workstation, device, mixed precision, batch size.

The same code path runs on a Colab T4 and on a local NVIDIA A5000. Only two things differ and both are
auto-detected here: where outputs go (Google Drive vs. ./runs) and how many images fit in one PGD batch.
"""
from __future__ import annotations

import contextlib
import os

import torch

COLAB_OUTPUT_ROOT = "/content/drive/MyDrive/CRPII/exp"


def in_colab() -> bool:
    """True inside Google Colab. `FACECLOAK_ENV=colab|local` forces the answer (useful for testing)."""
    forced = os.environ.get("FACECLOAK_ENV")
    if forced:
        return forced.strip().lower() == "colab"
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast(device: str, fp16: bool):
    """fp16 autocast on CUDA only. MPS/CPU run in fp32 (fp16 there is slow or produces NaNs)."""
    if device == "cuda" and fp16:
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def auto_batch_size(device: str) -> int:
    """PGD batch size at 512x512 with fp16: T4 (15 GB) -> 4, A5000 (24 GB) -> 8. OOM still halves it at runtime."""
    if device != "cuda":
        return 1
    mem = gpu_memory_gb()
    if mem >= 20:
        return 8
    if mem >= 12:
        return 4
    return 2


def describe() -> str:
    dev = get_device()
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "-"
    return (f"env={'colab' if in_colab() else 'local'} | device={dev} | gpu={gpu} "
            f"({gpu_memory_gb():.0f} GB) | torch={torch.__version__}")


def mount_drive_if_needed(path: str) -> None:
    """Mount Google Drive when running in Colab and `path` lives on it. No-op elsewhere."""
    if not str(path).startswith("/content/drive") or os.path.ismount("/content/drive"):
        return
    try:
        from google.colab import drive
    except ImportError:
        return
    drive.mount("/content/drive")
