from __future__ import annotations

import pytest

from configuration.dataset                import AugmentationConfig
from configuration.training               import LossConfig, LossCurriculumConfig, ParamMatching
from pipelines.shared.training.run_naming import RunNaming


def _loss(**flags) -> LossConfig:
    return LossConfig(**flags)


def _aug(**probabilities) -> AugmentationConfig:
    return AugmentationConfig(**probabilities)


def test_loss_tag_lists_enabled_terms_with_weights_param_first():
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_cosine_curve=True, weight_cosine_curve=0.05, use_covariance_match=True, weight_covariance_match=0.05)

    assert RunNaming.loss_tag(loss) == "param_l1_1-covariance_match_0.05-cosine_curve_0.05"


def test_loss_tag_rejects_empty_loss():
    with pytest.raises(ValueError):
        RunNaming.loss_tag(_loss())


def test_loss_tag_names_the_legacy_term():
    loss = _loss(use_param_legacy=True, weight_param_legacy=1.0, param_matching=ParamMatching.SORTED_GT)

    assert RunNaming.loss_tag(loss) == "param_legacy_1"


def test_matching_tag_names_the_strategy():
    assert RunNaming.matching_tag(_loss())                                          == "hungarian"
    assert RunNaming.matching_tag(_loss(param_matching=ParamMatching.SORTED_GT))    == "sorted_gt"


def test_gaussians_tag_counts_the_slots():
    assert RunNaming.gaussians_tag(3) == "K_3"
    assert RunNaming.gaussians_tag(5) == "K_5"


def test_presence_tag_letters_follow_enabled_flags():
    assert RunNaming.presence_tag(_loss())                                                     == "none"
    assert RunNaming.presence_tag(_loss(use_active_normalization=True))                        == "A"
    assert RunNaming.presence_tag(_loss(presence_balance=True))                                == "B"
    assert RunNaming.presence_tag(_loss(use_active_normalization=True, presence_balance=True)) == "AB"
    assert RunNaming.presence_tag(_loss(amp_focal_gamma=2.0))                                  == "none"


def test_augmentation_tag_letters_follow_enabled_probabilities():
    assert RunNaming.augmentation_tag(_aug())                                       == "hv"
    assert RunNaming.augmentation_tag(_aug(p_flip_h=0.0, p_flip_v=0.0))             == "noaug"
    assert RunNaming.augmentation_tag(_aug(p_rot90=0.25, p_noise=0.1))              == "hvrn"
    assert RunNaming.augmentation_tag(_aug(p_flip_h=0.0, p_flip_v=0.0, p_rot90=1.0)) == "r"


def test_tag_orders_model_head_matching_gaussians_augmentation_presence_loss():
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, param_matching=ParamMatching.SORTED_GT, use_active_normalization=True)

    assert RunNaming.tag("resunet", "set_pred", loss, 3, _aug()) == "resunet-set_pred-sorted_gt-K_3-hv-A-param_l1_1"


def test_training_tag_names_the_final_stage():
    warmup   = _loss(use_param_l1=True, weight_param_l1=1.0)
    complete = _loss(use_param_l1=True, weight_param_l1=1.0, use_covariance_match=True, weight_covariance_match=0.05)

    enabled  = LossCurriculumConfig(enabled=True,  warmup=warmup, complete=complete)
    disabled = LossCurriculumConfig(enabled=False, warmup=warmup, complete=complete)

    assert RunNaming.training_tag("unet", "conv", enabled, 3, _aug())  == "unet-conv-hungarian-K_3-hv-none-param_l1_1-covariance_match_0.05"
    assert RunNaming.training_tag("unet", "conv", disabled, 3, _aug()) == "unet-conv-hungarian-K_3-hv-none-param_l1_1-covariance_match_0.05"


def test_benchmark_unit_keeps_the_component_separator():
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_active_normalization=True)

    assert RunNaming.benchmark_unit("resunet-multihead", "mse_curve", loss, 3, _aug()) == "resunet-multihead-hungarian-K_3-hv-A__mse_curve"
    assert RunNaming.benchmark_unit("unet", "param_l1", loss, 3, _aug())               == "unet-conv-hungarian-K_3-hv-A__param_l1"


def test_benchmark_unit_without_component_uses_the_full_tag():
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_cosine_curve=True, weight_cosine_curve=0.05)

    assert RunNaming.benchmark_unit("resunet-set_pred", None, loss, 5, _aug()) == "resunet-set_pred-hungarian-K_5-hv-none-param_l1_1-cosine_curve_0.05"


def test_compose_appends_the_suffix():
    assert RunNaming.compose("unet-conv-hungarian-K_3-hv-A-param_l1_1", "seed0") == "unet-conv-hungarian-K_3-hv-A-param_l1_1_seed0"
    assert RunNaming.compose("unet-conv-hungarian-K_3-hv-A-param_l1_1", None)    == "unet-conv-hungarian-K_3-hv-A-param_l1_1"


def test_extras_slot_in_between_presence_and_loss():
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_active_normalization=True)

    assert RunNaming.tag("dual_resunet", "set_pred", loss, 2, _aug(), extras=("full.ifg",)) == "dual_resunet-set_pred-hungarian-K_2-hv-A-full.ifg-param_l1_1"
    assert RunNaming.tag("dual_resunet", "set_pred", loss, 2, _aug())                       == "dual_resunet-set_pred-hungarian-K_2-hv-A-param_l1_1"


def test_dual_model_tag_collapses_matching_trunks():
    assert RunNaming.dual_model_tag("unet_skip", "unet_skip") == "dual_unet_skip"
    assert RunNaming.dual_model_tag("resunet", "resunet")     == "dual_resunet"


def test_dual_model_tag_names_both_trunks_when_they_differ():
    assert RunNaming.dual_model_tag("unet_skip", "local_cnn") == "dual_unet_skip.local_cnn"
