
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
from configuration.training.general.loss import LossConfig

from configuration.training.backbone import (
    default_loss,
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


def test_backbone_entry_default_subconfigs():
    cfg = BackboneEntryConfig()
    assert cfg.backbone_name == "resunet"
    assert isinstance(cfg.loss, LossConfig)
    assert isinstance(cfg.inference, InferenceConfig)


def test_backbone_entry_default_loss_trains_the_parameter_l1_term():
    assert default_loss().use_param_l1 is True
    assert BackboneEntryConfig().loss == default_loss()


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
