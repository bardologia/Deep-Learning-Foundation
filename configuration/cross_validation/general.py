
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark.general import (
    ComparisonReportConfig,
    InferenceQueueConfig,
)
from configuration.cross_validation.dual                import DualCvConfig
from configuration.cross_validation.image_autoencoder   import ImageAeCvConfig
from configuration.cross_validation.profile_autoencoder import AeCvConfig
from configuration.dataset                              import AugmentationConfig, InputConfig
from configuration.normalization.general                import NormalizationConfig
from configuration.training.backbone                    import LossScaleProbeConfig, default_curriculum
from configuration.training.general.loss                import LossCurriculumConfig
from configuration.training.general.run                 import RunPathsConfig, TrainingQueueConfig, standard_seeds
from configuration.training.general.runtime             import OverfitCheckConfig
from configuration.training.jepa                        import JepaCouplingConfig


@dataclass
class FoldConfig:

    n_folds   : int = 10
    row_start : int = 1000
    row_end   : int = 1512
    guard     : int = 16


@dataclass
class CrossValidationConfig:

    training_type   : str  = "backbone"
    stage           : str  = "full"

    backbone_name   : str  = "resunet"
    backbone_head   : str  = "set_pred"
    model_overrides : dict = field(default_factory=dict)

    paths      : RunPathsConfig         = field(default_factory=lambda: RunPathsConfig(log_base_dir=Path("runs/cross_validation")))
    folds      : FoldConfig             = field(default_factory=FoldConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=lambda: InferenceQueueConfig(save_cubes=True))
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    input         : InputConfig          = field(default_factory=InputConfig.full_stack)
    curriculum    : LossCurriculumConfig = field(default_factory=default_curriculum)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)
    probe         : LossScaleProbeConfig = field(default_factory=LossScaleProbeConfig)
    overfit_check : OverfitCheckConfig   = field(default_factory=OverfitCheckConfig)

    prior_bias_init : bool = False

    dual              : DualCvConfig       = field(default_factory=DualCvConfig)
    jepa              : JepaCouplingConfig = field(default_factory=JepaCouplingConfig)
    autoencoder       : AeCvConfig         = field(default_factory=AeCvConfig)
    image_autoencoder : ImageAeCvConfig    = field(default_factory=ImageAeCvConfig)

    inference_splits : list[str] = field(default_factory=lambda: ["val", "test"])

    gpus            : list[int]  = field(default_factory=lambda: [0])
    gpus_file       : str        = ""
    run_tag         : str | None = None
    resume          : bool       = True
    resume_from     : str        = ""
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0

    def runs_inference(self) -> bool:
        return bool(self.inference_splits)
