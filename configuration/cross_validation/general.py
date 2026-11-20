
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.benchmark.general import (
    ComparisonReportConfig,
    InferenceQueueConfig,
)
from configuration.dataset                  import AugmentationConfig
from configuration.normalization.general    import NormalizationConfig
from configuration.training.general.loss    import LossConfig
from configuration.training.general.run     import RunPathsConfig, TrainingQueueConfig, standard_seeds


@dataclass
class FoldConfig:

    n_folds   : int = 10
    row_start : int = 0
    row_end   : int = 256
    guard     : int = 16


@dataclass
class CrossValidationConfig:

    stage           : str  = "full"

    backbone_name   : str  = "resunet"
    backbone_head   : str  = "conv"
    model_overrides : dict = field(default_factory=dict)

    paths      : RunPathsConfig         = field(default_factory=lambda: RunPathsConfig(log_base_dir=Path("runs/cross_validation")))
    folds      : FoldConfig             = field(default_factory=FoldConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=lambda: InferenceQueueConfig(save_maps=True))
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    loss          : LossConfig           = field(default_factory=LossConfig)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)

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
