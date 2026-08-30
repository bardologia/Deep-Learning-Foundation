"""Configuration of the spatial cross-validation experiment over row folds."""

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
    """Definition of the contiguous row folds used for cross-validation.

    Attributes:
        n_folds   : Number of folds the row range is split into.
        row_start : First row taking part in the split.
        row_end   : Last row taking part in the split.
        guard     : Rows dropped either side of a fold boundary to prevent spatial leakage between train and validation.
    """

    n_folds   : int = 10
    row_start : int = 1000
    row_end   : int = 1512
    guard     : int = 16


@dataclass
class CrossValidationConfig:
    """Top-level configuration of a spatial cross-validation run.

    Attributes:
        training_type     : Pipeline family being cross-validated.
        stage             : Which staged part of the cross-validation to execute.
        backbone_name     : Backbone family trained in each fold.
        backbone_head     : Output head attached to the backbone.
        model_overrides   : Architecture config fields overridden per run.
        paths             : Dataset and run path configuration.
        folds             : Azimuth fold definition.
        training          : Training queue settings shared by all folds.
        inference         : Inference queue settings.
        comparison        : Fold comparison report settings.
        input             : Input representation and channel selection.
        geometry          : Acquisition geometry.
        curriculum        : Loss curriculum applied in every fold.
        normalization     : Input and target normalisation settings.
        augmentation      : Training-time augmentation settings.
        probe             : Loss-scale probe settings.
        overfit_check     : Pre-training overfit sanity gate settings.
        dual              : Extra settings used when cross-validating dual-trunk models.
        jepa              : JEPA coupling settings.
        autoencoder       : Extra settings used when cross-validating profile autoencoders.
        image_autoencoder : Extra settings used when cross-validating image autoencoders.
        inference_splits  : Splits inference runs on after each fold.
        gpus              : GPU indices available to the run queue.
        gpus_file         : Path to a file that can resize the GPU pool while running.
        run_tag           : Optional tag appended to generated run names.
        resume            : Whether completed folds are skipped on a repeated launch.
        seed              : Base random seed.
        seeds             : Seeds trained per fold.
        poll_interval_s   : Scheduler polling interval in seconds.
    """

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

    dual              : DualCvConfig       = field(default_factory=DualCvConfig)
    jepa              : JepaCouplingConfig = field(default_factory=JepaCouplingConfig)
    autoencoder       : AeCvConfig         = field(default_factory=AeCvConfig)
    image_autoencoder : ImageAeCvConfig    = field(default_factory=ImageAeCvConfig)

    inference_splits : list[str] = field(default_factory=lambda: ["val", "test"])

    gpus            : list[int]  = field(default_factory=lambda: [0])
    gpus_file       : str        = ""
    run_tag         : str | None = None
    resume          : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0

    def runs_inference(self) -> bool:
        """Returns whether any inference split was requested."""
        return bool(self.inference_splits)
