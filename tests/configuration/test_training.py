
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

from configuration.training.backbone import (
    default_curriculum,
    PatchTrialsConfig,
    SecondaryTrialsConfig,
    BackboneTrainerConfig,
    BackboneEntryConfig,
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
    PatchTrialsConfig,
    SecondaryTrialsConfig,
    BackboneEntryConfig,
]

SIMPLE_IDS = [c.__name__ for c in SIMPLE_DEFAULT_CONFIGS]

@pytest.mark.parametrize("config_cls", SIMPLE_DEFAULT_CONFIGS, ids=SIMPLE_IDS)
def test_default_config_instantiates(config_cls):
    instance = config_cls()
    assert dataclasses.is_dataclass(instance)


def test_trainer_config_requires_gaussian():
    with pytest.raises(TypeError):
        BackboneTrainerConfig()

    instance = BackboneTrainerConfig(gaussian=make_gaussian())
    assert dataclasses.is_dataclass(instance)


def test_io_config_carries_no_runtime_writer():
    cfg = IOConfig()

    assert [f.name for f in dataclasses.fields(cfg)] == ["logdir"]
    assert not hasattr(cfg, "writer")
    assert dataclasses.asdict(cfg) == {"logdir": cfg.logdir}


def test_optimizer_config_values_sane():
    cfg = OptimizerConfig()
    assert cfg.eps > 0
    assert len(cfg.betas) == 2
    assert all(0 < b < 1 for b in cfg.betas)


def test_scheduler_config_positive_epochs():
    cfg = SchedulerConfig()
    assert cfg.epochs > 0
    assert cfg.eta_min >= 0


def test_warmup_config_factor_in_range():
    cfg = WarmupConfig()
    assert 0.0 <= cfg.warmup_fraction <= 1.0
    assert 0.0 <= cfg.warmup_start_factor <= 1.0


def test_early_stopping_patience_positive():
    cfg = EarlyStoppingConfig()
    assert cfg.patience > 0


def test_loss_config_weight_used_directly():
    cfg = LossConfig(weight_mse_curve=2.0, weight_param_l1=3.0)
    assert cfg.weight_mse_curve == pytest.approx(2.0)
    assert cfg.weight_param_l1  == pytest.approx(3.0)


def test_loss_config_param_weights_tuple():
    cfg = LossConfig()
    assert isinstance(cfg.param_weights, tuple)
    assert len(cfg.param_weights) == 3


def test_loss_curriculum_holds_two_loss_configs():
    cfg = LossCurriculumConfig()
    assert isinstance(cfg.warmup, LossConfig)
    assert isinstance(cfg.complete, LossConfig)
    assert cfg.warmup is not cfg.complete


def test_loss_curriculum_initial_stage_follows_enabled():
    cfg = LossCurriculumConfig()

    cfg.enabled = True
    assert cfg.initial_stage is cfg.warmup
    assert cfg.active_stages() == [cfg.warmup, cfg.complete]

    cfg.enabled = False
    assert cfg.initial_stage is cfg.complete
    assert cfg.active_stages() == [cfg.complete]


def test_curriculum_inheritance_copies_explicit_complete_edits_into_warmup():
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.use_l1_curve": True, "curriculum.complete.weight_l1_curve": 0.3}

    cfg.complete.use_l1_curve    = True
    cfg.complete.weight_l1_curve = 0.3

    inherited = CurriculumInheritance(cfg, default_curriculum(), overrides).apply()

    assert set(inherited) == {"use_l1_curve", "weight_l1_curve"}
    assert cfg.warmup.use_l1_curve    is True
    assert cfg.warmup.weight_l1_curve == pytest.approx(0.3)


def test_curriculum_inheritance_respects_explicit_warmup_overrides():
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.use_cosine_curve": False, "curriculum.warmup.use_cosine_curve": True}

    cfg.complete.use_cosine_curve = False

    inherited = CurriculumInheritance(cfg, default_curriculum(), overrides).apply()

    assert inherited == {}
    assert cfg.warmup.use_cosine_curve is True


def test_curriculum_inheritance_skips_fields_with_divergent_stage_defaults():
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.presence_balance": False}

    cfg.complete.presence_balance = False

    inherited = CurriculumInheritance(cfg, default_curriculum(), overrides).apply()

    assert inherited == {}
    assert cfg.warmup.presence_balance is False


def test_curriculum_inheritance_disabled_by_inherit_flag():
    cfg       = default_curriculum()
    overrides = {"curriculum.complete.use_l1_curve": True}

    cfg.inherit               = False
    cfg.complete.use_l1_curve = True

    assert CurriculumInheritance(cfg, default_curriculum(), overrides).apply() == {}
    assert cfg.warmup.use_l1_curve is False


def test_backbone_entry_default_subconfigs():
    cfg = BackboneEntryConfig()
    assert cfg.backbone_name == "resunet"
    assert isinstance(cfg.curriculum, LossCurriculumConfig)
    assert isinstance(cfg.inference, InferenceConfig)
    assert isinstance(cfg.warmup_losses, dict)
    assert isinstance(cfg.complete_losses, dict)


def test_backbone_entry_default_complete_losses_nonempty():
    cfg = BackboneEntryConfig()
    assert len(cfg.complete_losses) > 0
    for spec in cfg.complete_losses.values():
        assert spec["use_param_l1"] is True


def test_backbone_trainer_subconfig_factories():
    cfg = BackboneTrainerConfig(gaussian=make_gaussian())
    assert isinstance(cfg.optimizer, OptimizerConfig)
    assert isinstance(cfg.scheduler, SchedulerConfig)
    assert isinstance(cfg.gradient_clipper, GradientClipperConfig)


def test_every_seeded_entry_defaults_to_the_standard_seeds():
    from configuration.benchmark.general        import BenchmarkConfig
    from configuration.cross_validation.general import CrossValidationConfig
    from configuration.training.general.run     import standard_seeds

    assert standard_seeds() == [0, 1]

    seeded = [
        BackboneEntryConfig,
        CrossValidationConfig,
        BenchmarkConfig,
    ]
    for config_cls in seeded:
        assert config_cls().seeds == standard_seeds(), config_cls.__name__
