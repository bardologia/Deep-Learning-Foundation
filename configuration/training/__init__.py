"""Training configuration dataclasses for every model family.

Re-exports the shared optimization, runtime, loss and pretraining configs
alongside the backbone, JEPA, autoencoder and dual entry configs.
"""

from configuration.training.general import (
    OptimizerConfig,
    SchedulerConfig,
    WarmupConfig,
    EarlyStoppingConfig,
    GradientClipperConfig,
    IOConfig,
    OverfitCheckConfig,
    TrainingLoopConfig,
    MemoryConfig,
    CurriculumInheritance,
    LossConfig,
    LossCurriculumConfig,
    ParamMatching,
    SharedSubConfigInheritance,
    PretrainConfig,
)
from configuration.training.backbone import (
    default_curriculum,
    PairTrialsConfig,
    LogEnergyTrialsConfig,
    PatchTrialsConfig,
    SecondaryTrialsConfig,
    LossScaleProbeConfig,
    BackboneTrainerConfig,
    BackboneEntryConfig,
)
from configuration.training.jepa import (
    EmbeddingLossConfig,
    JepaCurriculumConfig,
    JepaTrainerConfig,
    JepaDefaults,
    JepaEntryConfig,
)
from configuration.training.image_autoencoder import (
    ImageAeLossConfig,
    ImageAeTrainerConfig,
    ImageAeEntryConfig,
)
from configuration.training.profile_autoencoder import (
    ProfileAeLossConfig,
    ProfileAeTrainerConfig,
    ProfileAeEntryConfig,
)
from configuration.training.dual import (
    DualEntryConfig,
    DualRoutingTrialsConfig,
    DualRatioTrialsConfig,
    dual_curriculum,
)
