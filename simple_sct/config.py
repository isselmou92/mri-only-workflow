# simple_sct/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import yaml
import torch


# -----------------------------
# Dataclass definitions
# -----------------------------

@dataclass
class DataCfg:
    # Directories (paths resolved relative to YAML if not absolute)
    mri_dir: str = "path/to/MRI"
    ct_dir: str = "path/to/CT"

    # External held-out test cohort. The CV split never touches these volumes;
    # they are only evaluated by scripts/ensemble_test.py with the winning loss.
    test_mri_dir: Optional[str] = None
    test_ct_dir: Optional[str] = None

    # Mouse IDs that must never enter the training/CV dataset, e.g. the
    # external test cohort that physically lives in the same folder as the
    # training mice. Matched case-insensitively against the leading
    # underscore-separated token of each MRI filename stem.
    exclude_mouse_ids: List[str] = field(default_factory=list)

    # Optional cap on the number of acquisitions kept per mouse. When set, the
    # dataset keeps the first N timepoints per mouse (sorted by parsed
    # trailing integer so T1<T2<...<T10). None disables the cap.
    max_acquisitions_per_mouse: Optional[int] = None

    # Declared cap for the external test cohort. Not consumed by the dataset
    # directly; ensemble_test.py copies this value into
    # ``max_acquisitions_per_mouse`` when constructing the test loader.
    test_max_acquisitions_per_mouse: Optional[int] = None

    # 3-fold cross-validation index. 0, 1, or 2. Each fold holds out a
    # disjoint subset of mouse IDs as the validation set.
    fold: int = 0

    # Contract & scaling
    enforce_shape: Tuple[int, int, int] = (128, 128, 112)  # (D,H,W)
    reorient_to_canonical: bool = True
    scale_input: str = "zscore"  # "zscore" | "minmax" | "none"
    scale_target: str = "hu"     # "hu" | "minmax"
    hu_min: float = -1024.0
    hu_max: float = 3071.0

    # Loader workers
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True


@dataclass
class TrainXaiOccCfg:
    patch: Tuple[int, int, int] = (9, 9, 9)
    stride: Tuple[int, int, int] = (8, 8, 8)


@dataclass
class TrainXaiSlicesCfg:
    axial: List[int] = field(default_factory=lambda: [64])
    sagittal: List[int] = field(default_factory=lambda: [64])
    coronal: List[int] = field(default_factory=lambda: [56])


@dataclass
class TrainCfg:
    epochs: int = 200
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    amp: bool = True
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    ema: bool = False
    device_preference: str = "cuda"  # "cuda" | "cpu" | "auto"
    continue_from_checkpoint: Optional[str] = None
    crop_prob: float = 0.5
    flip_prob: float = 0.5

    # XAI during training
    xai_enabled: bool = True
    xai_log_every_steps: int = 200
    xai_methods: List[str] = field(default_factory=lambda: ["saliency"])
    xai_occlusion: TrainXaiOccCfg = field(default_factory=TrainXaiOccCfg)
    xai_log_norm: bool = True
    xai_log_axes: List[str] = field(default_factory=lambda: ["axial", "sagittal", "coronal"])
    xai_slice_indices: TrainXaiSlicesCfg = field(default_factory=TrainXaiSlicesCfg)


@dataclass
class ModelCfg:
    in_channels: int = 1
    out_channels: int = 1
    base_filters: int = 16
    depth: int = 3
    final_act: str = "identity"  # "identity" | "tanh"


@dataclass
class LossCfg:
    l1_weight: float = 1.0
    ssim_weight: float = 0.0
    ssim_kernel: int = 7
    ssim_sigma: float = 1.5


@dataclass
class MetricsCfg:
    compute_psnr: bool = True
    compute_ssim: bool = True


@dataclass
class LoggingSlicesCfg:
    axial: List[int] = field(default_factory=lambda: [64])
    sagittal: List[int] = field(default_factory=lambda: [64])
    coronal: List[int] = field(default_factory=lambda: [56])


@dataclass
class LoggingCfg:
    log_interval: int = 10
    tensorboard: bool = True
    checkpoint_every: int = 20
    save_best_on: str = "val_mae"
    out_dir: str = "out"
    log_dir: str = "runs"
    ckpt_dir: str = "checkpoints"
    tb_images_every_steps: int = 100
    tb_render_axes: List[str] = field(default_factory=lambda: ["axial", "sagittal", "coronal"])
    tb_slice_indices: LoggingSlicesCfg = field(default_factory=LoggingSlicesCfg)

    # Held-out CV val-fold predicted images, logged every K epochs (0 disables).
    # Logs MRI / pred / CT triplet panels for the first N samples of the val
    # loader under tag prefix "val/volumes/sample{i}".
    tb_log_val_images_every_epochs: int = 1
    tb_val_max_samples: int = 1


@dataclass
class CFG:
    experiment_name: str = "simple_sct"
    seed: int = 1337
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    # runtime (filled by loader)
    _config_path: Optional[Path] = None
    _config_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_config_path", None)
        d.pop("_config_dir", None)
        return d


# -----------------------------
# Utilities
# -----------------------------

def _update_dataclass_from_dict(dc_obj, src: Dict[str, Any]):
    """Recursively update dataclass instance fields from a dict; ignore unknown keys."""
    for k, v in (src or {}).items():
        if not hasattr(dc_obj, k):
            continue
        cur = getattr(dc_obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _update_dataclass_from_dict(cur, v)
        else:
            setattr(dc_obj, k, v)


def _resolve_paths(cfg: CFG):
    """Resolve all relative paths against YAML directory (Windows-safe)."""
    cdir = cfg._config_dir or Path(".")

    # Data dirs (test_*_dir are optional; only resolve if set)
    for attr in ["mri_dir", "ct_dir", "test_mri_dir", "test_ct_dir"]:
        val = getattr(cfg.data, attr, None)
        if not val:
            continue
        p = Path(val)
        if not p.is_absolute():
            setattr(cfg.data, attr, str((cdir / p).resolve()))

    # Logging dirs
    for attr in ["out_dir", "log_dir", "ckpt_dir"]:
        p = Path(getattr(cfg.logging, attr))
        if not p.is_absolute():
            setattr(cfg.logging, attr, str((cdir / p).resolve()))


def choose_device(pref: str = "cuda") -> torch.device:
    """CUDA-first device selection.

      - "cuda": pick CUDA if available, else CPU
      - "cpu":  always CPU
      - "auto": CUDA if available, else MPS (Apple), else CPU
    """
    pref = (pref or "cuda").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_cfg(cfg: CFG):
    # enforce_shape is (D,H,W) with positive ints
    D, H, W = tuple(cfg.data.enforce_shape)
    assert all(isinstance(x, int) and x > 0 for x in (D, H, W)), "enforce_shape must be positive ints (D,H,W)"

    # scaling modes
    assert cfg.data.scale_input in ("zscore", "minmax", "none"), "scale_input must be zscore|minmax|none"
    assert cfg.data.scale_target in ("hu", "minmax"), "scale_target must be hu|minmax"

    # cross-validation fold (3-fold)
    assert isinstance(cfg.data.fold, int) and 0 <= cfg.data.fold < 3, (
        f"data.fold must be 0, 1, or 2 (3-fold CV); got {cfg.data.fold!r}"
    )

    # axes sanity
    valid_axes = {"axial", "sagittal", "coronal"}
    assert set(cfg.train.xai_log_axes).issubset(valid_axes), "xai_log_axes must ⊆ {axial,sagittal,coronal}"
    assert set(cfg.logging.tb_render_axes).issubset(valid_axes), "tb_render_axes must ⊆ {axial,sagittal,coronal}"

    # loss must have at least one non-zero weight
    assert cfg.loss.l1_weight > 0 or cfg.loss.ssim_weight > 0, (
        "At least one of loss.l1_weight, loss.ssim_weight must be > 0"
    )


# -----------------------------
# Loader
# -----------------------------

def load_config(path: str | Path) -> CFG:
    path = Path(path).expanduser().resolve()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = CFG()
    _update_dataclass_from_dict(cfg, raw)

    # runtime
    cfg._config_path = path
    cfg._config_dir = path.parent

    _resolve_paths(cfg)
    validate_cfg(cfg)
    return cfg
