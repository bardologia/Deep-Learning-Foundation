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
    ResourceConfig,
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
    PatchTrialsConfig,
    PhysicsTrialsConfig,
    SecondaryTrialsConfig,
    LossScaleProbeConfig,
    BackboneTrainerConfig,
    BackboneEntryConfig,
)
from configuration.training.jepa import (
    EmbeddingLossConfig,
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
from configuration.training.unrolled import (
    UnrolledEntryConfig,
    UnrolledTrainerConfig,
    UnrolledTrainingConfig,
)
from configuration.training.dual import (
    DualEntryConfig,
    DualRoutingTrialsConfig,
    DualRatioTrialsConfig,
    dual_curriculum,
)
