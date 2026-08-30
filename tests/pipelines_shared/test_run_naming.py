"""Tests for the run-name tags that encode model, head, matching, loss, and autoencoder choices."""

from __future__ import annotations

import pytest

from configuration.dataset                import AugmentationConfig
from configuration.training               import EmbeddingLossConfig, LossConfig, LossCurriculumConfig, ParamMatching
from configuration.training.jepa          import JepaMaskingConfig
from pipelines.shared.training.run_naming import JepaNamingSpec, RunNaming


def _loss(**flags) -> LossConfig:
    """Returns a loss config with the given term flags and weights."""
    return LossConfig(**flags)


def _aug(**probabilities) -> AugmentationConfig:
    """Returns an augmentation config with the given per-transform probabilities."""
    return AugmentationConfig(**probabilities)


def _jepa(**overrides) -> JepaNamingSpec:
    """Returns a JEPA naming spec defaulting to a frozen GRU profile autoencoder with stop-gradient targets."""
    spec = dict(
        profile_ae      = "gru_ae",
        profile_mode    = "frozen",
        target_provider = "stopgrad",
        image_ae        = None,
        image_mode      = "frozen",
        embedding_loss  = EmbeddingLossConfig(),
        param_loss      = _loss(use_param_l1=True, weight_param_l1=1.0, param_matching=ParamMatching.SORTED_GT, use_active_normalization=True),
        masking         = JepaMaskingConfig(),
    )
    spec.update(overrides)

    return JepaNamingSpec(**spec)


def test_loss_tag_lists_enabled_terms_with_weights_param_first():
    """The loss tag lists enabled terms with their weights, parameter term first."""
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_cosine_curve=True, weight_cosine_curve=0.05, use_moments=True, weight_moments=0.05)

    assert RunNaming.loss_tag(loss) == "param_l1_1-moments_0.05-cosine_curve_0.05"


def test_loss_tag_rejects_empty_loss():
    """A loss config with no enabled term raises ValueError."""
    with pytest.raises(ValueError):
        RunNaming.loss_tag(_loss())




def test_matching_tag_names_the_strategy():
    """The matching tag names the Hungarian or sorted-ground-truth strategy."""
    assert RunNaming.matching_tag(_loss())                                          == "hungarian"
    assert RunNaming.matching_tag(_loss(param_matching=ParamMatching.SORTED_GT))    == "sorted_gt"


def test_gaussians_tag_counts_the_slots():
    """The Gaussian tag records the number of slots as K_n."""
    assert RunNaming.gaussians_tag(3) == "K_3"
    assert RunNaming.gaussians_tag(5) == "K_5"


def test_presence_tag_letters_follow_enabled_flags():
    """The presence tag spells A for active normalization and B for presence balance."""
    assert RunNaming.presence_tag(_loss())                                                     == "none"
    assert RunNaming.presence_tag(_loss(use_active_normalization=True))                        == "A"
    assert RunNaming.presence_tag(_loss(presence_balance=True))                                == "B"
    assert RunNaming.presence_tag(_loss(use_active_normalization=True, presence_balance=True)) == "AB"
    assert RunNaming.presence_tag(_loss(amp_focal_gamma=2.0))                                  == "none"


def test_augmentation_tag_letters_follow_enabled_probabilities():
    """The augmentation tag spells one letter per transform with a non-zero probability."""
    assert RunNaming.augmentation_tag(_aug())                                       == "hv"
    assert RunNaming.augmentation_tag(_aug(p_flip_h=0.0, p_flip_v=0.0))             == "noaug"
    assert RunNaming.augmentation_tag(_aug(p_rot90=0.25, p_noise=0.1))              == "hvrn"
    assert RunNaming.augmentation_tag(_aug(p_flip_h=0.0, p_flip_v=0.0, p_rot90=1.0)) == "r"


def test_tag_orders_model_head_matching_gaussians_augmentation_presence_loss():
    """The full tag orders model, head, matching, slots, augmentation, presence, then loss."""
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, param_matching=ParamMatching.SORTED_GT, use_active_normalization=True)

    assert RunNaming.tag("resunet", "set_pred", loss, 3, _aug()) == "resunet-set_pred-sorted_gt-K_3-hv-A-param_l1_1"


def test_training_tag_names_the_final_stage():
    """The training tag names the complete curriculum stage whether or not the curriculum is enabled."""
    warmup   = _loss(use_param_l1=True, weight_param_l1=1.0)
    complete = _loss(use_param_l1=True, weight_param_l1=1.0, use_moments=True, weight_moments=0.05)

    enabled  = LossCurriculumConfig(enabled=True,  warmup=warmup, complete=complete)
    disabled = LossCurriculumConfig(enabled=False, warmup=warmup, complete=complete)

    assert RunNaming.training_tag("unet", "conv", enabled, 3, _aug())  == "unet-conv-hungarian-K_3-hv-none-param_l1_1-moments_0.05"
    assert RunNaming.training_tag("unet", "conv", disabled, 3, _aug()) == "unet-conv-hungarian-K_3-hv-none-param_l1_1-moments_0.05"


def test_benchmark_unit_keeps_the_component_separator():
    """A benchmark unit appends the component after a double underscore and defaults the head to conv."""
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_active_normalization=True)

    assert RunNaming.benchmark_unit("resunet-multihead", "mse_curve", loss, 3, _aug()) == "resunet-multihead-hungarian-K_3-hv-A__mse_curve"
    assert RunNaming.benchmark_unit("unet", "param_l1", loss, 3, _aug())               == "unet-conv-hungarian-K_3-hv-A__param_l1"


def test_benchmark_unit_without_component_uses_the_full_tag():
    """Without a component the benchmark unit is the full tag including the loss terms."""
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_cosine_curve=True, weight_cosine_curve=0.05)

    assert RunNaming.benchmark_unit("resunet-set_pred", None, loss, 5, _aug()) == "resunet-set_pred-hungarian-K_5-hv-none-param_l1_1-cosine_curve_0.05"


def test_compose_appends_the_suffix():
    """Compose appends an underscored suffix and is a no-op when the suffix is missing."""
    assert RunNaming.compose("unet-conv-hungarian-K_3-hv-A-param_l1_1", "seed0") == "unet-conv-hungarian-K_3-hv-A-param_l1_1_seed0"
    assert RunNaming.compose("unet-conv-hungarian-K_3-hv-A-param_l1_1", None)    == "unet-conv-hungarian-K_3-hv-A-param_l1_1"


def test_extras_slot_in_between_presence_and_loss():
    """Extra tags are inserted between the presence letters and the loss terms."""
    loss = _loss(use_param_l1=True, weight_param_l1=1.0, use_active_normalization=True)

    assert RunNaming.tag("dual_resunet", "set_pred", loss, 2, _aug(), extras=("full.derived",)) == "dual_resunet-set_pred-hungarian-K_2-hv-A-full.derived-param_l1_1"
    assert RunNaming.tag("dual_resunet", "set_pred", loss, 2, _aug())                       == "dual_resunet-set_pred-hungarian-K_2-hv-A-param_l1_1"


def test_dual_model_tag_collapses_matching_trunks():
    """Two identical trunks collapse into a single dual_ prefixed name."""
    assert RunNaming.dual_model_tag("unet_skip", "unet_skip") == "dual_unet_skip"
    assert RunNaming.dual_model_tag("resunet", "resunet")     == "dual_resunet"


def test_dual_model_tag_names_both_trunks_when_they_differ():
    """Two different trunks are joined by a dot after the dual_ prefix."""
    assert RunNaming.dual_model_tag("unet_skip", "local_cnn") == "dual_unet_skip.local_cnn"


def test_embedding_loss_tag_lists_enabled_terms_with_weights():
    """The default embedding loss tag names the embedding and curve mean-squared terms."""
    assert RunNaming.embedding_loss_tag(EmbeddingLossConfig()) == "emb_mse_1-curve_mse_1"


def test_embedding_loss_tag_names_every_term_and_the_curve_kind():
    """Each enabled embedding term is abbreviated with its weight and the curve loss kind."""
    embedding = EmbeddingLossConfig(use_embedding_mse=False, use_embedding_cosine=True, weight_embedding_cosine=0.5, use_embedding_smoothl1=True, weight_embedding_smoothl1=2.0, use_curve_recon=True, weight_curve_recon=0.25, curve_kind="charbonnier")

    assert RunNaming.embedding_loss_tag(embedding) == "emb_cos_0.5-emb_sl1_2-curve_charbonnier_0.25"


def test_embedding_loss_tag_rejects_empty_loss():
    """An embedding loss config with no enabled term raises ValueError."""
    with pytest.raises(ValueError):
        RunNaming.embedding_loss_tag(EmbeddingLossConfig(use_embedding_mse=False, use_curve_recon=False))


def test_autoencoder_tags_strip_the_ae_suffix_and_carry_the_mode():
    """Autoencoder tags drop the _ae suffix and append the frozen or finetune mode."""
    assert RunNaming.profile_ae_tag("gru_ae", "frozen")      == "pae_gru.frozen"
    assert RunNaming.profile_ae_tag("mlp_ae", "finetune")    == "pae_mlp.finetune"
    assert RunNaming.image_ae_tag("conv2d_ae", "frozen")     == "iae_conv2d.frozen"
    assert RunNaming.image_ae_tag("conv2d_ae", "finetune")   == "iae_conv2d.finetune"


def test_jepa_tag_profile_coupled_carries_target_ae_and_embedding_loss():
    """A profile-coupled JEPA tag names the target provider, the profile autoencoder, and the embedding loss."""
    assert RunNaming.jepa_tag("resunet", "conv", _jepa(), 5, _aug()) == "resunet-conv-stopgrad-K_5-hv-none-pae_gru.frozen-emb_mse_1-curve_mse_1"


def test_jepa_tag_orders_profile_before_image_autoencoder():
    """The profile autoencoder tag precedes the image autoencoder tag."""
    naming = _jepa(profile_mode="finetune", target_provider="live", image_ae="conv2d_ae", image_mode="finetune")

    assert RunNaming.jepa_tag("resunet", "conv", naming, 5, _aug()) == "resunet-conv-live-K_5-hv-none-pae_gru.finetune-iae_conv2d.finetune-emb_mse_1-curve_mse_1"


def test_jepa_tag_image_only_carries_the_param_loss_it_trains_on():
    """An image-only JEPA tag falls back to the matching and parameter loss it trains on."""
    naming = _jepa(profile_ae=None, image_ae="conv2d_ae")

    assert RunNaming.jepa_tag("resunet", "set_pred", naming, 5, _aug()) == "resunet-set_pred-sorted_gt-K_5-hv-A-iae_conv2d.frozen-param_l1_1"


def test_jepa_tag_rejects_a_spec_without_any_autoencoder():
    """A JEPA spec naming neither autoencoder raises ValueError."""
    with pytest.raises(ValueError):
        RunNaming.jepa_tag("resunet", "conv", _jepa(profile_ae=None, image_ae=None), 5, _aug())


def test_embedding_loss_tag_appends_the_sobolev_term():
    """An enabled curve Sobolev term is appended to the embedding loss tag."""
    embedding = EmbeddingLossConfig(use_curve_sobolev=True, weight_curve_sobolev=0.5)

    assert RunNaming.embedding_loss_tag(embedding) == "emb_mse_1-curve_mse_1-curve_sobolev_0.5"
