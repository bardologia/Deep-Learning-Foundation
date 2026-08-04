from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark.general import (
    ComparisonReportConfig,
    InferenceQueueConfig,
)
from configuration.cross_validation.jepa                import JepaCvConfig
from configuration.cross_validation.profile_autoencoder import AeCvConfig
from configuration.dataset                              import AugmentationConfig
from configuration.normalization.general                import NormalizationConfig
from configuration.sar.geometry_config                  import GeometryConfig
from configuration.training.backbone                    import default_curriculum
from configuration.training.general.loss                import LossCurriculumConfig
from configuration.training.general.run                 import RunPathsConfig, TrainingQueueConfig, standard_seeds


@dataclass
class FoldConfig:
    n_folds       : int = 10
    azimuth_start : int = 1000
    azimuth_end   : int = 16000
    guard         : int = 64


@dataclass
class CrossValidationConfig:
    training_type   : str  = "backbone"

    backbone_name   : str  = "resunet"
    backbone_head   : str  = "conv"
    model_overrides : dict = field(default_factory=dict)

    paths      : RunPathsConfig         = field(default_factory=lambda: RunPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/cross_validation")))
    folds      : FoldConfig             = field(default_factory=FoldConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    geometry      : GeometryConfig       = field(default_factory=GeometryConfig)
    curriculum    : LossCurriculumConfig = field(default_factory=default_curriculum)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)
    jepa          : JepaCvConfig         = field(default_factory=JepaCvConfig)
    autoencoder   : AeCvConfig           = field(default_factory=AeCvConfig)

    inference_splits : list[str] = field(default_factory=lambda: ["val", "test"])

    gpus            : list[int]  = field(default_factory=lambda: [2, 3])
    gpus_file       : str        = ""
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0

    def runs_inference(self) -> bool:
        return self.training_type != "profile_autoencoder"
