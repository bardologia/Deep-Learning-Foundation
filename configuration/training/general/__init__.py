"""Family-agnostic training configuration blocks.

Re-exports the optimizer, scheduler, runtime, loss curriculum, pretraining and
run-path configs shared by the backbone, dual, autoencoder and JEPA pipelines.
"""

from configuration.training.general.optimization import (
    OptimizerConfig,
    SchedulerConfig,
    WarmupConfig,
    EarlyStoppingConfig,
    GradientClipperConfig,
)
from configuration.training.general.runtime import (
    IOConfig,
    OverfitCheckConfig,
    TrainingLoopConfig,
    MemoryConfig,
    ResourceConfig,
)
from configuration.training.general.loss import (
    CurriculumInheritance,
    LossConfig,
    LossCurriculumConfig,
    ParamMatching,
)
from configuration.training.general.trainer     import SharedSubConfigInheritance
from configuration.training.general.pretraining import PretrainConfig
from configuration.training.general.run import (
    RunPathsConfig,
    TrainingPathsConfig,
    TrainingQueueConfig,
)
