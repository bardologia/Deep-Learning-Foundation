"""Tests for the training configuration dataclasses and curriculum inheritance."""

from __future__ import annotations

import dataclasses

import pytest

from configuration.training.general.optimization import (
    OptimizerConfig,
    SchedulerConfig,
    WarmupConfig,
    EarlyStoppingConfig,
    GradientClipperConfig,
)
from configuration.resources import ResourceConfig
from configuration.training.general.runtime import (
    IOConfig,
    TrainingLoopConfig,
    MemoryConfig,
)
from configuration.training.general.loss import (
    CurriculumInheritance,
    LossConfig,
    LossCurriculumConfig,
)
from configuration.training.general.trainer import SharedSubConfigInheritance

from configuration.training.backbone import (
    default_curriculum,
    PatchTrialsConfig,
    SecondaryTrialsConfig,
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
from configuration.inference.general import InferenceConfig

from tests.configuration._helpers import make_gaussian


SIMPLE_DEFAULT_CONFIGS = [
    OptimizerConfig,
    SchedulerConfig,
    WarmupConfig,
    EarlyStoppingConfig,
    GradientClipperConfig,
    IOConfig,
    TrainingLoopConfig,
    MemoryConfig,
    ResourceConfig,
    LossConfig,
    LossCurriculumConfig,
    EmbeddingLossConfig,
    ImageAeLossConfig,
    ProfileAeLossConfig,
    PatchTrialsConfig,
    SecondaryTrialsConfig,
    BackboneEntryConfig,
    JepaEntryConfig,
    ImageAeEntryConfig,
    ProfileAeEntryConfig,
]

SIMPLE_IDS = [c.__name__ for c in SIMPLE_DEFAULT_CONFIGS]

TRAINER_CONFIGS = [
    BackboneTrainerConfig,
    JepaTrainerConfig,
    ImageAeTrainerConfig,
    ProfileAeTrainerConfig,
]


@pytest.mark.parametrize("config_cls", SIMPLE_DEFAULT_CONFIGS, ids=SIMPLE_IDS)
def test_default_config_instantiates(config_cls):
    """Verifies every simple configuration dataclass instantiates with its defaults."""
    instance = config_cls()
    assert dataclasses.is_dataclass(instance)


@pytest.mark.parametrize("config_cls", TRAINER_CONFIGS, ids=[c.__name__ for c in TRAINER_CONFIGS])
def test_trainer_config_requires_gaussian(config_cls):
    """Verifies trainer configurations require an explicit Gaussian configuration."""
    with pytest.raises(TypeError):
        config_cls()

    instance = config_cls(gaussian=make_gaussian())
    assert dataclasses.is_dataclass(instance)


def test_io_config_carries_no_runtime_writer():
    """Verifies IOConfig holds only the log directory and no runtime writer handle."""
    cfg = IOConfig()

    assert [f.name for f in dataclasses.fields(cfg)] == ["logdir"]
    assert not hasattr(cfg, "writer")
    assert dataclasses.asdict(cfg) == {"logdir": cfg.logdir}


def test_optimizer_config_values_sane():
    """Verifies the optimizer defaults carry a positive epsilon and valid Adam betas."""
    cfg = OptimizerConfig()
    assert cfg.eps > 0
    assert len(cfg.betas) == 2
    assert all(0 < b < 1 for b in cfg.betas)


def test_scheduler_config_positive_epochs():
    """Verifies the scheduler defaults carry positive epochs and a non-negative floor rate."""
    cfg = SchedulerConfig()
    assert cfg.epochs > 0
    assert cfg.eta_min >= 0


def test_warmup_config_factor_in_range():
    """Verifies the warmup defaults carry a fraction and a start factor within the unit interval."""
    cfg = WarmupConfig()
    assert 0.0 <= cfg.warmup_fraction <= 1.0
    assert 0.0 <= cfg.warmup_start_factor <= 1.0


def test_early_stopping_patience_positive():
    """Verifies the early-stopping patience defaults to a positive number of epochs."""
    cfg = EarlyStoppingConfig()
    assert cfg.patience > 0


def test_loss_config_weight_used_directly():
    """Verifies loss weights are stored exactly as passed."""
    cfg = LossConfig(weight_mse_curve=2.0, weight_param_l1=3.0)
    assert cfg.weight_mse_curve == pytest.approx(2.0)
    assert cfg.weight_param_l1  == pytest.approx(3.0)


def test_loss_config_param_weights_tuple():
    """Verifies the per-parameter loss weights form a three-element tuple."""
    cfg = LossConfig()
    assert isinstance(cfg.param_weights, tuple)
    assert len(cfg.param_weights) == 3


def test_loss_curriculum_holds_two_loss_configs():
    """Verifies the curriculum owns two distinct loss configurations."""
    cfg = LossCurriculumConfig()
    assert isinstance(cfg.warmup, LossConfig)
    assert isinstance(cfg.complete, LossConfig)
    assert cfg.warmup is not cfg.complete


def test_loss_curriculum_initial_stage_follows_enabled():
    """Verifies the initial stage and the active stage list follow the curriculum enabled flag."""
    cfg = LossCurriculumConfig()

    cfg.enabled = True
    assert cfg.initial_stage is cfg.warmup
    assert cfg.active_stages() == [cfg.warmup, cfg.complete]

    cfg.enabled = False
    assert cfg.initial_stage is cfg.complete
    assert cfg.active_stages() == [cfg.complete]


def test_curriculum_inheritance_copies_explicit_complete_edits_into_warmup():
    """Verifies complete-stage edits made through overrides are inherited by the warmup stage."""
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.use_l1_curve": True, "curriculum.complete.weight_l1_curve": 0.3}

    cfg.complete.use_l1_curve    = True
    cfg.complete.weight_l1_curve = 0.3

    inherited = CurriculumInheritance(cfg, default_curriculum(), overrides).apply()

    assert set(inherited) == {"use_l1_curve", "weight_l1_curve"}
    assert cfg.warmup.use_l1_curve    is True
    assert cfg.warmup.weight_l1_curve == pytest.approx(0.3)


def test_curriculum_inheritance_respects_explicit_warmup_overrides():
    """Verifies an explicit warmup override blocks inheritance from the complete stage."""
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.use_cosine_curve": False, "curriculum.warmup.use_cosine_curve": True}

    cfg.complete.use_cosine_curve = False

    inherited = CurriculumInheritance(cfg, default_curriculum(), overrides).apply()

    assert inherited == {}
    assert cfg.warmup.use_cosine_curve is True


def test_curriculum_inheritance_skips_fields_with_divergent_stage_defaults():
    """Verifies fields whose stage defaults differ are never inherited."""
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.presence_balance": False}

    cfg.complete.presence_balance = False

    inherited = CurriculumInheritance(cfg, default_curriculum(), overrides).apply()

    assert inherited == {}
    assert cfg.warmup.presence_balance is False


def test_curriculum_inheritance_disabled_by_inherit_flag():
    """Verifies clearing the inherit flag disables curriculum inheritance entirely."""
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.use_l1_curve": True}

    cfg.inherit               = False
    cfg.complete.use_l1_curve = True

    assert CurriculumInheritance(cfg, default_curriculum(), overrides).apply() == {}
    assert cfg.warmup.use_l1_curve is False


def test_backbone_entry_default_subconfigs():
    """Verifies the backbone entry defaults to resunet with its curriculum and inference subconfigs."""
    cfg = BackboneEntryConfig()
    assert cfg.backbone_name == "resunet"
    assert isinstance(cfg.curriculum, LossCurriculumConfig)
    assert isinstance(cfg.inference, InferenceConfig)
    assert isinstance(cfg.warmup_losses, dict)
    assert isinstance(cfg.complete_losses, dict)


def test_backbone_entry_default_complete_losses_nonempty():
    """Verifies every default complete-stage loss specification enables the parameter L1 term."""
    cfg = BackboneEntryConfig()
    assert len(cfg.complete_losses) > 0
    for spec in cfg.complete_losses.values():
        assert spec["use_param_l1"] is True


def test_backbone_trainer_subconfig_factories():
    """Verifies the backbone trainer builds its optimizer, scheduler and clipper subconfigs."""
    cfg = BackboneTrainerConfig(gaussian=make_gaussian())
    assert isinstance(cfg.optimizer, OptimizerConfig)
    assert isinstance(cfg.scheduler, SchedulerConfig)
    assert isinstance(cfg.gradient_clipper, GradientClipperConfig)


def test_jepa_defaults_inference_is_inference_config():
    """Verifies the JEPA inference defaults are an InferenceConfig that saves cubes."""
    inf = JepaDefaults.inference()
    assert isinstance(inf, InferenceConfig)
    assert inf.save_cubes is True


def test_jepa_trainer_param_loss_default():
    """Verifies the JEPA trainer enables the parameter L1 term and owns an embedding loss config."""
    cfg = JepaTrainerConfig(gaussian=make_gaussian())
    assert cfg.param_loss.use_param_l1 is True
    assert isinstance(cfg.embedding_loss, EmbeddingLossConfig)


def test_shared_subconfig_inheritance_copies_fields():
    """Verifies inherit_shared_from rebinds the shared subconfigs to the donor objects."""
    class Base:
        pass

    base = Base()
    for name in ("geometry", "early_stopping", "warmup", "scheduler", "io",
                 "optimizer", "training", "resources", "gradient_clipper", "memory"):
        setattr(base, name, object())

    target = JepaTrainerConfig(gaussian=make_gaussian())
    target.inherit_shared_from(base)
    assert target.optimizer is base.optimizer
    assert target.scheduler is base.scheduler
    assert target.memory    is base.memory


def test_shared_subconfig_class_is_mixin():
    """Verifies the trainer configurations mix in the shared-subconfig inheritance behaviour."""
    assert issubclass(JepaTrainerConfig, SharedSubConfigInheritance)
    assert issubclass(ImageAeTrainerConfig, SharedSubConfigInheritance)
    assert issubclass(ProfileAeTrainerConfig, SharedSubConfigInheritance)


def test_image_ae_entry_training_overrides():
    """Verifies the image autoencoder entry overrides batch size and model name."""
    cfg = ImageAeEntryConfig()
    assert cfg.training.batch_size == 512
    assert cfg.ae_model_name       == "conv2d_ae"


def test_profile_ae_entry_training_overrides():
    """Verifies the profile autoencoder entry overrides the loader and model defaults."""
    cfg = ProfileAeEntryConfig()
    assert cfg.training.batch_size      == 8192
    assert cfg.training.num_workers     == 14
    assert cfg.training.prefetch_factor == 4
    assert cfg.ae_model_name            == "mlp_ae"


def test_profile_ae_entry_lr_defaults_mirror_the_architecture_config():
    """Verifies the profile autoencoder entry mirrors the architecture learning rates and weight decays."""
    from configuration.architectures.profile_autoencoder import ProfileAutoencoderBaseConfig

    cfg = ProfileAeEntryConfig()
    assert cfg.encoder_lr == ProfileAutoencoderBaseConfig.encoder_lr
    assert cfg.decoder_lr == ProfileAutoencoderBaseConfig.decoder_lr
    assert cfg.encoder_wd == ProfileAutoencoderBaseConfig.encoder_wd
    assert cfg.decoder_wd == ProfileAutoencoderBaseConfig.decoder_wd


def test_every_seeded_entry_defaults_to_the_standard_seeds():
    """Verifies every seeded entry configuration defaults to the standard seeds."""
    from configuration.benchmark.general        import BenchmarkConfig
    from configuration.cross_validation.general import CrossValidationConfig
    from configuration.training.dual            import DualEntryConfig
    from configuration.training.general.run     import standard_seeds

    assert standard_seeds() == [0, 1]

    seeded = [
        BackboneEntryConfig,
        JepaEntryConfig,
        ProfileAeEntryConfig,
        ImageAeEntryConfig,
        DualEntryConfig,
        CrossValidationConfig,
        BenchmarkConfig,
    ]
    for config_cls in seeded:
        assert config_cls().seeds == standard_seeds(), config_cls.__name__
