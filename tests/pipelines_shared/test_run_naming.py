from __future__ import annotations

from configuration.dataset                import AugmentationConfig
from configuration.training               import LossConfig, LossKind
from pipelines.shared.training.run_naming import RunNaming


def _aug(**probabilities) -> AugmentationConfig:
    return AugmentationConfig(**probabilities)


def test_loss_tag_names_the_loss_kind():
    assert RunNaming.loss_tag(LossConfig())                     == "mse"
    assert RunNaming.loss_tag(LossConfig(kind=LossKind.HUBER)) == "huber"


def test_augmentation_tag_letters_follow_enabled_probabilities():
    assert RunNaming.augmentation_tag(_aug())                                        == "hv"
    assert RunNaming.augmentation_tag(_aug(p_flip_h=0.0, p_flip_v=0.0))             == "noaug"
    assert RunNaming.augmentation_tag(_aug(p_rot90=0.25, p_noise=0.1))              == "hvrn"
    assert RunNaming.augmentation_tag(_aug(p_flip_h=0.0, p_flip_v=0.0, p_rot90=1.0)) == "r"


def test_tag_orders_model_head_loss_and_augmentation():
    assert RunNaming.tag("resunet", "conv", LossConfig(kind=LossKind.L1), _aug()) == "resunet-conv-l1-hv"


def test_benchmark_unit_splits_the_model_key():
    assert RunNaming.benchmark_unit("unet", LossConfig(), _aug(p_flip_h=0.0, p_flip_v=0.0)) == "unet-conv-mse-noaug"


def test_compose_appends_the_suffix():
    assert RunNaming.compose("unet-conv-mse-hv", "seed0") == "unet-conv-mse-hv_seed0"
    assert RunNaming.compose("unet-conv-mse-hv", None)    == "unet-conv-mse-hv"
