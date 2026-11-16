# simple_sct/utils/__init__.py
"""
Utilities for logging, TensorBoard, reproducibility, and quick CUDA introspection.
"""

from .logging import (  # noqa: F401
    get_logger,
    prepare_dirs,
    create_tb_writer,
    log_scalars,
    log_triplet_panels,
    log_xai_maps,
)
from .seed import (  # noqa: F401
    set_seed,
    cuda_summary,
)

__all__ = [
    "get_logger",
    "prepare_dirs",
    "create_tb_writer",
    "log_scalars",
    "log_triplet_panels",
    "log_xai_maps",
    "set_seed",
    "cuda_summary",
]
