from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset                      import AugmentationConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import default_curriculum
from configuration.training.general.loss        import LossCurriculumConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig
from configuration.training.image_autoencoder   import ImageAeLossConfig
from configuration.training.profile_autoencoder import ProfileAeLossConfig
from configuration.tuning.jepa                  import JepaTuneConfig


def _default_ae_loss():
    return ProfileAeLossConfig()


def _default_image_ae_loss():
    return ImageAeLossConfig()


@dataclass
class TuningConfig:
    n_trials : int = 100
    n_epochs : int = 30

    base_seed : int = 42

    early_stop_patience : int = 8

    pruner_n_startup_trials : int = 8
    pruner_n_warmup_steps   : int = 8

    emit_trial_docs  : bool = False
    emit_study_plots : bool = False


@dataclass
class TuningEntryConfig:
    training_type : str = "backbone"

    paths  : RunPathsConfig = field(default_factory=lambda: RunPathsConfig(log_base_dir=Path("/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/tuning")))
    tuning : TuningConfig   = field(default_factory=TuningConfig)

    gpus        : list[int]  = field(default_factory=lambda: [0, 1, 2, 3])
    gpus_file   : str        = ""
    heads       : list[str]  = field(default_factory=lambda: ["conv"])
    skip_models : list[str]  = field(default_factory=list)
    run_tag     : str | None = None

    training        : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    curriculum      : LossCurriculumConfig = field(default_factory=default_curriculum)
    normalization   : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation    : AugmentationConfig   = field(default_factory=AugmentationConfig)
    ae_loss         : object               = field(default_factory=_default_ae_loss)
    image_ae_loss   : object               = field(default_factory=_default_image_ae_loss)
    jepa            : JepaTuneConfig       = field(default_factory=JepaTuneConfig)
    pixel_subsample : float                = 1.0
    keep_empty_frac : float                = 0.05
