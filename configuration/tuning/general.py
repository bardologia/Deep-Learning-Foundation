
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset                      import AugmentationConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.general.loss        import LossConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig


@dataclass
class TuningConfig:

    n_trials : int = 100
    n_epochs : int = 30

    base_seed : int = 42

    early_stop_patience : int = 8

    pruner_n_startup_trials : int = 8
    pruner_n_warmup_steps   : int = 8

    search_lr    : bool = True
    search_arch  : bool = True
    search_batch : bool = False

    batch_sizes : list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])

    plateau_patience  : int   = 0
    plateau_min_delta : float = 0.0
    time_budget_h     : float = 0.0

    emit_trial_docs        : bool = False
    emit_study_plots       : bool = False
    keep_trial_checkpoints : bool = False


@dataclass
class TuningEntryConfig:

    stage : str = "full"

    paths  : RunPathsConfig = field(default_factory=lambda: RunPathsConfig(log_base_dir=Path("runs/tuning")))
    tuning : TuningConfig   = field(default_factory=TuningConfig)

    gpus        : list[int]  = field(default_factory=lambda: [0])
    n_workers   : int        = 0
    gpus_file   : str        = ""
    skip_models : list[str]  = field(default_factory=list)
    run_tag     : str | None = None
    resume_from : str        = ""

    training      : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    loss          : LossConfig           = field(default_factory=LossConfig)
    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)
