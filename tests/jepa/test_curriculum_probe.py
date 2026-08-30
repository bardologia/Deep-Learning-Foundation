"""Tests for the JEPA curriculum schedule and the embedding loss-scale probe.

Covers the stage selection of the trainer, the curriculum swap through the shared
controller, the resume re-application, and the embedding-regime probe measuring
every term at unit weight.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from configuration.training            import JepaCurriculumConfig, LossScaleProbeConfig
from configuration.training.jepa       import EmbeddingLossConfig, default_param_loss
from pipelines.jepa.training.coupling  import TargetProvider
from pipelines.jepa.training.loss      import Loss as EmbeddingLoss
from pipelines.jepa.training.loss_probe import JepaEmbeddingLossProbe
from pipelines.jepa.training.trainer   import Trainer
from pipelines.shared.training.curriculum import CurriculumController

from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats

from tests.conftest import SilentLogger
from tests.jepa.conftest import EMBEDDING_DIM, N_GAUSSIANS, PROFILE_LENGTH, SPATIAL, IdentityNormStats, make_autoencoder
from tests.jepa.test_trainer import build_module


def make_stage_shim(has_profile, curriculum):
    """Returns a Trainer shim exposing only the stage-selection surface.

    Args:
        has_profile : Whether the shim represents the embedding regime.
        curriculum  : JepaCurriculumConfig of the shim.

    Returns:
        The shim with `config`, `has_profile` and `curriculum` set.
    """
    trainer             = Trainer.__new__(Trainer)
    trainer.has_profile = has_profile
    trainer.curriculum  = curriculum
    trainer.config      = SimpleNamespace(embedding_loss=EmbeddingLossConfig(), param_loss=default_param_loss(), curriculum=curriculum)
    return trainer


def test_disabled_curriculum_starts_on_the_main_stage():
    """With the curriculum off, the initial stage is the flat embedding or param loss."""
    embedding = make_stage_shim(True,  JepaCurriculumConfig(enabled=False))
    param     = make_stage_shim(False, JepaCurriculumConfig(enabled=False))

    assert embedding._initial_loss_stage() is embedding.config.embedding_loss
    assert param._initial_loss_stage()     is param.config.param_loss


def test_enabled_curriculum_starts_on_the_warmup_stage_of_the_regime():
    """With the curriculum on, the initial stage is the warmup stage of the active regime."""
    curriculum = JepaCurriculumConfig(enabled=True, swap_epoch=3)
    embedding  = make_stage_shim(True,  curriculum)
    param      = make_stage_shim(False, curriculum)

    assert embedding._initial_loss_stage() is curriculum.warmup_embedding
    assert param._initial_loss_stage()     is curriculum.warmup_param


def test_embedding_loss_set_curriculum_swaps_the_stage():
    """The embedding criterion adopts the new stage configuration on set_curriculum."""
    autoencoder = make_autoencoder("none")
    z_axis      = torch.linspace(-4.0, 4.0, PROFILE_LENGTH)
    warmup      = EmbeddingLossConfig(use_embedding_mse=True, use_curve_recon=False)
    complete    = EmbeddingLossConfig(use_embedding_mse=False, use_embedding_cosine=True, weight_embedding_cosine=1.0, use_curve_recon=False)

    criterion = EmbeddingLoss(autoencoder, TargetProvider("stopgrad"), warmup, z_axis, IdentityNormStats(), 3, ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0)))
    criterion.set_curriculum(complete)

    assert criterion.emb_cfg is complete

    z_hat  = torch.rand(2, EMBEDDING_DIM, SPATIAL, SPATIAL)
    gt     = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    result = criterion(z_hat, gt)

    assert set(result["components"]) == {"embedding_cosine"}


class RecordingStub:
    """Callable stub recording every invocation of its named methods."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args))


def make_controller(curriculum, complete_stage, criterion):
    """Returns a CurriculumController over recording stubs for the optimization state."""
    return CurriculumController(
        curriculum       = curriculum,
        complete_stage   = complete_stage,
        criterion        = criterion,
        early_stopping   = RecordingStub(),
        lr_scheduler     = RecordingStub(),
        warmup           = RecordingStub(),
        optimizer        = SimpleNamespace(param_groups=[], state={}),
        update_optimizer = lambda lrs: None,
        logger           = SilentLogger(),
    )


def test_controller_swaps_only_at_the_swap_epoch():
    """The controller installs the complete stage exactly at the configured epoch."""
    autoencoder = make_autoencoder("none")
    z_axis      = torch.linspace(-4.0, 4.0, PROFILE_LENGTH)
    warmup      = EmbeddingLossConfig()
    complete    = EmbeddingLossConfig(use_embedding_cosine=True)

    criterion  = EmbeddingLoss(autoencoder, TargetProvider("stopgrad"), warmup, z_axis, IdentityNormStats(), 3, ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0)))
    controller = make_controller(JepaCurriculumConfig(enabled=True, swap_epoch=2), complete, criterion)

    assert not controller.maybe_swap(1)
    assert criterion.emb_cfg is warmup

    assert controller.maybe_swap(2)
    assert criterion.emb_cfg is complete


def test_controller_never_swaps_when_disabled():
    """A disabled curriculum leaves the criterion untouched at every epoch."""
    autoencoder = make_autoencoder("none")
    z_axis      = torch.linspace(-4.0, 4.0, PROFILE_LENGTH)
    warmup      = EmbeddingLossConfig()

    criterion  = EmbeddingLoss(autoencoder, TargetProvider("stopgrad"), warmup, z_axis, IdentityNormStats(), 3, ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0)))
    controller = make_controller(JepaCurriculumConfig(enabled=False, swap_epoch=0), EmbeddingLossConfig(), criterion)

    assert not any(controller.maybe_swap(epoch) for epoch in range(5))
    assert criterion.emb_cfg is warmup


def test_on_state_restored_reapplies_the_main_stage_past_the_swap():
    """Resuming past the swap epoch re-installs the main loss stage on the criterion."""
    curriculum = JepaCurriculumConfig(enabled=True, swap_epoch=1)
    trainer    = make_stage_shim(True, curriculum)

    swaps = []
    trainer.criterion = SimpleNamespace(set_curriculum=lambda stage, context="": swaps.append(stage))
    trainer.logger    = SilentLogger()

    trainer._on_state_restored({"epoch": 0})
    assert swaps == []

    trainer._on_state_restored({"epoch": 1})
    assert swaps == [trainer.config.embedding_loss]


def test_embedding_probe_measures_every_term_at_unit_weight():
    """The embedding probe reports one suggested weight per probed term."""
    module = build_module()
    z_axis = torch.linspace(-4.0, 4.0, PROFILE_LENGTH)

    probe = JepaEmbeddingLossProbe(
        probe_cfg           = LossScaleProbeConfig(enabled=True, n_batches=1, reference="embedding_mse", exit_after=False),
        embedding_cfg       = EmbeddingLossConfig(),
        autoencoder         = module.profile_autoencoder,
        target_provider     = TargetProvider("stopgrad"),
        norm_stats          = IdentityNormStats(),
        params_per_gaussian = 3,
        profile_normalizer  = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0)),
        logger              = SilentLogger(),
    )

    for use_flag, weight_key in probe.EMBEDDING_TERMS:
        assert getattr(probe.loss_cfg, use_flag)
        assert getattr(probe.loss_cfg, weight_key) == 1.0

    images  = torch.rand(2, 2, SPATIAL, SPATIAL)
    gt      = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    weights = probe.run([(images, gt)], module, torch.device("cpu"), z_axis)

    assert set(weights) == {"embedding_mse", "embedding_cosine", "embedding_smoothl1", "embedding_huber", "embedding_charbonnier", "embedding_variance", "embedding_covariance", "curve_recon", "curve_sobolev"}
    assert weights["embedding_mse"] == pytest.approx(1.0)


def test_embedding_probe_rejects_unknown_flags():
    """An enabled_losses key outside the embedding term registry raises."""
    module = build_module()

    with pytest.raises(ValueError):
        JepaEmbeddingLossProbe(
            probe_cfg           = LossScaleProbeConfig(enabled=True, enabled_losses={"use_param_l1": True}),
            embedding_cfg       = EmbeddingLossConfig(),
            autoencoder         = module.profile_autoencoder,
            target_provider     = TargetProvider("stopgrad"),
            norm_stats          = IdentityNormStats(),
            params_per_gaussian = 3,
            profile_normalizer  = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0)),
        )
