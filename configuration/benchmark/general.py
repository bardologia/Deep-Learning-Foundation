
from __future__ import annotations

from dataclasses import dataclass, field

from configuration.dataset                      import AugmentationConfig
from configuration.normalization.general        import NormalizationConfig
from configuration.training.general.loss        import LossConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig, standard_seeds


@dataclass
class InferenceQueueConfig:

    split           : str        = "test"
    checkpoint_name : str        = "best_model.pt"
    batch_size      : int | None = None
    num_workers     : int        = 4
    seed            : int        = 0

    stitch_window : str  = "hann"
    save_maps     : bool = True
    save_plots    : bool = False

    cmap         : str = "viridis"
    cmap_error   : str = "magma"
    fig_dpi      : int = 150
    save_dpi     : int = 300
    figure_style : str = "report"


@dataclass
class ComparisonReportConfig:

    embed_images: bool = False


@dataclass
class BenchmarkConfig:

    stage : str = "full"

    paths      : RunPathsConfig         = field(default_factory=RunPathsConfig)
    training   : TrainingQueueConfig    = field(default_factory=TrainingQueueConfig)
    inference  : InferenceQueueConfig   = field(default_factory=InferenceQueueConfig)
    comparison : ComparisonReportConfig = field(default_factory=ComparisonReportConfig)

    normalization : NormalizationConfig  = field(default_factory=NormalizationConfig)
    augmentation  : AugmentationConfig   = field(default_factory=AugmentationConfig)
    loss          : LossConfig           = field(default_factory=LossConfig)

    gpus            : list[int]  = field(default_factory=lambda: [0])
    gpus_file       : str        = ""
    skip_models     : list[str]  = field(default_factory=list)
    run_tag         : str | None = None
    resume          : bool       = True
    resume_from     : str        = ""
    infer_after     : bool       = True
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=standard_seeds)
    poll_interval_s : float      = 5.0

    sweep_loss_kinds : list[str] = field(default_factory=lambda: ["mse"])

    def runs_inference(self) -> bool:
        return self.infer_after
