# simple_sct/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import yaml
import torch


# -----------------------------
# Dataclass definitions (dir-based pairing)
# -----------------------------

@dataclass
class PairingCfg:
    mode: str = "stem"  # "stem" (match by filename stem) | "sorted" (sort both lists)
    case_insensitive: bool = True
    mri_strip_suffixes: List[str] = field(default_factory=lambda: ["_MR", "_MRI"])
    ct_strip_suffixes: List[str] = field(default_factory=lambda: ["_CT", "_sCT", "_CTHU"])


@dataclass
class DataCfg:
    # Directories (Windows-friendly; paths resolved relative to YAML if not absolute)
    mri_dir: str = "path/to/MRI"
    ct_dir: str = "path/to/CT"

    # Glob patterns (recursive by default)
    mri_glob: str = "**/*.nii*"
    ct_glob: str = "**/*.nii*"

    # Pairing policy
    pairing: PairingCfg = field(default_factory=PairingCfg)

    # Contract & scaling
    enforce_shape: Tuple[int, int, int] = (128, 128, 112)  # (D,H,W)
    reorient_to_canonical: bool = True  # Reorient to RAS+ for consistent anatomy
    scale_input: str = "zscore"  # "zscore" | "minmax" | "none"
    scale_target: str = "hu"     # "hu" | "minmax"
    hu_min: float = -1024.0
    hu_max: float = 3071.0

    # Loader workers
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True  # Faster CPU->GPU transfer


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

    # XAI during training
    xai_enabled: bool = True
    xai_log_every_steps: int = 200
    xai_methods: List[str] = field(default_factory=lambda: ["saliency"])  # ["saliency","occlusion"]
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
    ssim_weight: float = 0.2
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
    tb_num_examples: int = 1


@dataclass
class EvalCfg:
    out_csv: str = "eval_metrics.csv"


@dataclass
class CFG:
    experiment_name: str = "simple_sct_explainable_13_01_2026"
    seed: int = 1337
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

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

    # Data dirs
    for attr in ["mri_dir", "ct_dir"]:
        p = Path(getattr(cfg.data, attr))
        if not p.is_absolute():
            setattr(cfg.data, attr, str((cdir / p).resolve()))

    # Logging dirs
    for attr in ["out_dir", "log_dir", "ckpt_dir"]:
        p = Path(getattr(cfg.logging, attr))
        if not p.is_absolute():
            setattr(cfg.logging, attr, str((cdir / p).resolve()))

    # Eval CSV
    out_csv = Path(cfg.eval.out_csv)
    if not out_csv.is_absolute():
        cfg.eval.out_csv = str((cdir / out_csv).resolve())


def choose_device(pref: str = "cuda") -> torch.device:
    """
    CUDA-first device selection (mirrors typical Windows/PyCharm CUDA preference):
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

    # pairing
    assert cfg.data.pairing.mode in ("stem", "sorted"), "pairing.mode must be 'stem' or 'sorted'"
    assert isinstance(cfg.data.pairing.case_insensitive, bool)
    assert isinstance(cfg.data.pairing.mri_strip_suffixes, list)
    assert isinstance(cfg.data.pairing.ct_strip_suffixes, list)

    # axes sanity
    valid_axes = {"axial", "sagittal", "coronal"}
    assert set(cfg.train.xai_log_axes).issubset(valid_axes), "xai_log_axes must ⊆ {axial,sagittal,coronal}"
    assert set(cfg.logging.tb_render_axes).issubset(valid_axes), "tb_render_axes must ⊆ {axial,sagittal,coronal}"


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
