"""Configuration for the Optuna hyper-parameter tuning entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.dataset                      import AugmentationConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.backbone            import default_curriculum
from configuration.training.general.loss        import LossCurriculumConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig
from configuration.training.image_autoencoder   import ImageAeLossConfig
from configuration.training.jepa                import JepaCouplingConfig
from configuration.training.profile_autoencoder import ProfileAeLossConfig


def _default_ae_loss():
    """Returns the default profile-autoencoder loss config."""
    return ProfileAeLossConfig()


def _default_image_ae_loss():
    """Returns the default image-autoencoder loss config."""
    return ImageAeLossConfig()


@dataclass
class TuningConfig:
    """Search budget, search space, seeding, pruning and stop rules of one Optuna study.

    Attributes:
        n_trials                : Number of trials the study runs.
        n_epochs                : Epochs each trial trains for.
        base_seed               : Seed offset from which per-trial seeds are derived.
        early_stop_patience     : Validation rounds without improvement inside a trial.
        pruner_n_startup_trials : Trials completed before pruning starts.
        pruner_n_warmup_steps   : Steps a trial is protected from pruning.
        search_lr               : Whether the study samples the learning rates, weight decays and dropout of the model configuration.
        search_arch             : Whether the study samples the architecture fields of the model configuration.
        search_batch            : Whether the study samples the training batch size from ``batch_sizes``; the trial applies the linear learning-rate scaling the training entries apply, so a tuned rate stays quoted at the reference batch.
        batch_sizes             : Batch sizes the study may pick when ``search_batch`` is on.
        plateau_patience        : Finished trials without the best value improving before the study stops; zero runs the trial target out.
        plateau_min_delta       : Smallest improvement of the best value, relative to the value at the last improvement, that counts as progress for the plateau rule.
        time_budget_h           : Hours after which a worker starts no further trial; zero sets no budget.
        emit_trial_docs         : Whether a markdown report is written per trial.
        emit_study_plots        : Whether study-level plots are rendered at the end.
    """

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

    emit_trial_docs  : bool = False
    emit_study_plots : bool = False


@dataclass
class TuningEntryConfig:
    """Command-line surface of the hyper-parameter tuning entry point.

    Attributes:
        training_type   : Which trainer the study optimises, e.g. backbone or jepa.
        stage           : Which part of the run to execute: 'prepare' creates the studies, 'compare' writes the reports of finished studies, 'full' does both around the trials.
        paths           : Dataset locations plus the root the study writes trials under.
        tuning          : Search budget and pruning settings of the study.
        gpus            : CUDA device indices trials are fanned out over.
        n_workers       : Trial workers each study is fanned out over; zero means one per device in the GPU pool.
        gpus_file       : File listing device indices, re-read to resize the pool live.
        heads           : Output heads included in the search.
        skip_models     : Architecture registry keys excluded from the search.
        run_tag         : Suffix appended to the study folder name.
        resume_from     : Path at or inside an existing tuning run to continue; its recorded configuration is loaded, the command-line overrides are applied on top, and the studies in its database are topped up to the trial target.
        training        : Training schedule shared by all trials.
        curriculum      : Loss curriculum stages applied inside each trial.
        normalization   : Input normalization strategy and statistics.
        augmentation    : Patch augmentation settings.
        ae_loss         : Profile-autoencoder loss config used by autoencoder studies.
        image_ae_loss   : Image-autoencoder loss config used by image studies.
        jepa            : Autoencoder coupling used by JEPA studies.
        pixel_subsample : Fraction of scene pixels kept as profile-training samples.
        keep_empty_frac : Fraction of empty profiles kept in that sample.
    """

    training_type : str = "backbone"
    stage         : str = "full"

    paths  : RunPathsConfig = field(default_factory=lambda: RunPathsConfig(log_base_dir=Path("runs/tuning")))
    tuning : TuningConfig   = field(default_factory=TuningConfig)

    gpus        : list[int]  = field(default_factory=lambda: [0])
    n_workers   : int        = 0
    gpus_file   : str        = ""
    heads       : list[str]  = field(default_factory=lambda: ["conv"])
    skip_models : list[str]  = field(default_factory=list)
    run_tag     : str | None = None
    resume_from : str        = ""

    training        : TrainingQueueConfig  = field(default_factory=TrainingQueueConfig)
    curriculum      : LossCurriculumConfig = field(default_factory=default_curriculum)
    normalization   : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation    : AugmentationConfig   = field(default_factory=AugmentationConfig)
    ae_loss         : object               = field(default_factory=_default_ae_loss)
    image_ae_loss   : object               = field(default_factory=_default_image_ae_loss)
    jepa            : JepaCouplingConfig   = field(default_factory=JepaCouplingConfig)
    pixel_subsample : float                = 1.0
    keep_empty_frac : float                = 0.05
