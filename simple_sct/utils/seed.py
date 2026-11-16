# simple_sct/utils/seed.py
from __future__ import annotations

"""
Reproducible seeding and lightweight CUDA introspection.

Usage
-----
from simple_sct.utils.seed import set_seed, cuda_summary
set_seed(1337, deterministic=False)
logger.info(cuda_summary())
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 1337, deterministic: bool = False) -> None:
    """
    Set Python, NumPy, and PyTorch seeds. Configure cuDNN for either speed (default)
    or deterministic behavior.

    Args:
      seed: base RNG seed
      deterministic:
        - False (default): cuDNN benchmark ON (fastest, non-deterministic kernels possible)
        - True: cuDNN deterministic ON (slower, reproducible convs). Also sets
                CUBLAS_WORKSPACE_CONFIG for determinism on newer CUDA/cuBLAS.

    Notes
    -----
    - For full determinism on NVIDIA:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  (or ":16:8")
      must be set **before** any CUDA context is created. We set it here if requested.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / cuBLAS settings
    if deterministic:
        # Must be set before the first CUDA call to take effect on some stacks
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # (Optional) Stable matmul; safe on Ampere+
    try:
        torch.set_float32_matmul_precision("high")  # "high" or "medium" (PyTorch >= 2.0)
    except Exception:
        pass


def cuda_summary(device: Optional[torch.device] = None) -> str:
    """
    Return a one-line CUDA summary for logging.

    Example:
      "[CUDA] True | device=0 (NVIDIA RTX A6000) | cap=8.6 | bf16=True | tf32=True"
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return "[CUDA] False | device=CPU"

    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    cap = ".".join(map(str, torch.cuda.get_device_capability(idx)))
    bf16 = torch.cuda.is_bf16_supported()
    tf32 = torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends.cuda, "matmul") else False
    return f"[CUDA] True | device={idx} ({name}) | cap={cap} | bf16={bf16} | tf32={tf32}"
