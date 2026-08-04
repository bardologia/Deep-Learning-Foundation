from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset               import AugmentationConfig, InputConfig
from configuration.normalization.general import NormalizationConfig
from configuration.sar.geometry_config   import GeometryConfig
from configuration.training.backbone     import default_curriculum
from configuration.training.general.loss import LossCurriculumConfig
from configuration.training.general.run  import RunPathsConfig, TrainingQueueConfig, standard_seeds


@dataclass
class PatchGridConfig:
    minimum               : tuple[int, int] = (0, 0)
    maximum               : tuple[int, int] = (96, 48)
    step                  : int             = 0
    stride_ratio          : float           = 0.5
    constant_pixel_budget : bool            = True


@dataclass
class PatchSweepConfig:
    backbone_name   : str  = "unet"
    backbone_head   : str  = "conv"
    model_overrides : dict = field(default_factory=lambda: {"features": [64, 128, 256]})

    dataset_base_path : Path = Path("/ste/rnd/User/vice_vi/Dataset")
    dataset_filter    : list = field(default_factory=list)

    patch : PatchGridConfig = field(default_factory=PatchGridConfig)

    paths         : RunPathsConfig       = field(default_factory=lambda: RunPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/patch_sweep")))
    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    geometry      : GeometryConfig       = field(default_factory=GeometryConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=default_curriculum)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    gpus_file       : str        = ""
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0
