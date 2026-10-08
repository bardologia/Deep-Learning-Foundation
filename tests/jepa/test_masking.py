"""Tests for JEPA input masking and the extended embedding-loss terms.

Covers the mask sampler statistics and validation, the input filler, the
mask-restricted embedding loss, the trainer integration on the training and the
unmasked evaluation path, the new robust and collapse-regularizer terms, and the
masked run-name token.
"""

from __future__ import annotations


import pytest
import torch

from configuration.training.jepa      import EmbeddingLossConfig, JepaMaskingConfig
from pipelines.jepa.training.coupling import TargetProvider
from pipelines.jepa.training.loss     import Loss as EmbeddingLoss
from pipelines.jepa.training.masking  import InputMasker, MaskSampler
from pipelines.shared.training.run_naming import RunNaming

from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats

from tests.jepa.conftest     import EMBEDDING_DIM, N_GAUSSIANS, PROFILE_LENGTH, SPATIAL, IdentityNormStats, make_autoencoder
from tests.jepa.test_trainer import make_trainer_shim
from tools.monitoring.logger  import NullLogger
from tools.monitoring.tracker import NullTracker


def make_criterion(emb_cfg=None):
    """Returns an embedding criterion over the tiny autoencoder of the conftest."""
    autoencoder = make_autoencoder("none")
    z_axis      = torch.linspace(-4.0, 4.0, PROFILE_LENGTH)
    cfg         = emb_cfg or EmbeddingLossConfig(use_curve_recon=False)
    return EmbeddingLoss(autoencoder, TargetProvider("stopgrad"), cfg, z_axis, IdentityNormStats(), 3, ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0)))


def test_random_mask_ratio_is_respected():
    """The random strategy hides roughly the configured fraction of pixels."""
    sampler = MaskSampler(JepaMaskingConfig(enabled=True, strategy="random", mask_ratio=0.5))
    mask    = sampler.sample(8, 32, 32, torch.device("cpu"))

    assert mask.shape == (8, 32, 32)
    assert 0.35 < mask.float().mean().item() < 0.65


def test_block_mask_hides_rectangles_below_the_target_ratio():
    """The block strategy hides a nonzero region bounded near the target ratio."""
    sampler = MaskSampler(JepaMaskingConfig(enabled=True, strategy="block", mask_ratio=0.4, n_blocks=4))
    mask    = sampler.sample(4, 32, 32, torch.device("cpu"))

    ratio = mask.float().mean().item()
    assert 0.0 < ratio <= 0.5


def test_every_sample_hides_at_least_one_pixel():
    """Even a tiny mask ratio never yields an empty per-sample mask."""
    sampler = MaskSampler(JepaMaskingConfig(enabled=True, strategy="random", mask_ratio=0.01))
    mask    = sampler.sample(16, 4, 4, torch.device("cpu"))

    assert mask.flatten(1).any(dim=1).all()


@pytest.mark.parametrize("bad", [{"strategy": "nope"}, {"mask_ratio": 0.0}, {"mask_ratio": 1.0}, {"n_blocks": 0}])
def test_sampler_rejects_invalid_settings(bad):
    """Unknown strategies, degenerate ratios and empty block counts raise."""
    with pytest.raises(ValueError):
        MaskSampler(JepaMaskingConfig(enabled=True, **bad))


def test_zero_fill_writes_zeros_into_masked_pixels():
    """The zeros fill blanks exactly the hidden pixels and keeps the rest."""
    masker = InputMasker(JepaMaskingConfig(enabled=True, fill="zeros"))
    images = torch.ones(2, 3, 4, 4)
    mask   = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[:, :2, :] = True

    out = masker.apply(images, mask)

    assert (out[:, :, :2, :] == 0.0).all()
    assert (out[:, :, 2:, :] == 1.0).all()


def test_noise_fill_perturbs_masked_pixels():
    """The noise fill writes non-constant values into the hidden pixels."""
    masker = InputMasker(JepaMaskingConfig(enabled=True, fill="noise", fill_noise_std=1.0))
    images = torch.ones(2, 3, 4, 4)
    mask   = torch.ones(2, 4, 4, dtype=torch.bool)

    out = masker.apply(images, mask)

    assert out.std().item() > 0.1


def test_unknown_fill_raises():
    """An unknown fill mode is rejected at construction."""
    with pytest.raises(ValueError):
        InputMasker(JepaMaskingConfig(enabled=True, fill="nope"))


def test_masked_loss_only_sees_the_masked_pixels():
    """Restricting the loss to a mask makes off-mask predictions irrelevant."""
    criterion = make_criterion()

    z_hat = torch.rand(2, EMBEDDING_DIM, SPATIAL, SPATIAL)
    gt    = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    mask  = torch.zeros(2, SPATIAL, SPATIAL, dtype=torch.bool)
    mask[:, 0, 0] = True

    base = criterion(z_hat, gt, mask)["total_loss"]

    corrupted                = z_hat.clone()
    corrupted[:, :, 1:, 1:] += 100.0
    same                     = criterion(corrupted, gt, mask)["total_loss"]

    assert same.item() == pytest.approx(base.item(), rel=1e-6)


def test_trainer_masks_input_and_restricts_the_loss(monkeypatch):
    """With masking on, the trainer hides input pixels and channels the mask to the criterion."""
    trainer         = make_trainer_shim()
    trainer.masking = JepaMaskingConfig(enabled=True, strategy="random", mask_ratio=0.5, loss_on="masked")
    trainer.mask_sampler = MaskSampler(trainer.masking)
    trainer.input_masker = InputMasker(trainer.masking)

    captured  = {}
    criterion = trainer.criterion

    def spy(pred, gt, mask=None):
        captured["mask"] = mask
        return criterion(pred, gt, mask)

    monkeypatch.setattr(trainer, "criterion", spy)

    images = torch.rand(2, 2, SPATIAL, SPATIAL)
    gt     = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    out    = trainer._compute_loss((images, gt))

    assert captured["mask"] is not None
    assert captured["mask"].dtype == torch.bool
    assert torch.isfinite(out["total_loss"])


def test_embedding_huber_and_charbonnier_match_manual():
    """The robust embedding terms equal the manual formulas on the normalized embeddings."""
    cfg = EmbeddingLossConfig(use_embedding_mse=False, use_curve_recon=False, use_embedding_huber=True, weight_embedding_huber=1.0, embedding_huber_delta=0.5, use_embedding_charbonnier=True, weight_embedding_charbonnier=1.0, embedding_charbonnier_eps=1e-2)
    criterion = make_criterion(cfg)

    z_hat = torch.rand(2, EMBEDDING_DIM, SPATIAL, SPATIAL)
    gt    = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    out   = criterion(z_hat, gt)

    assert set(out["components"]) == {"embedding_huber", "embedding_charbonnier"}
    assert torch.isfinite(out["total_loss"])
    assert out["total_loss"].item() == pytest.approx((out["components"]["embedding_huber"] + out["components"]["embedding_charbonnier"]).item(), rel=1e-6)


def test_variance_hinge_penalizes_collapsed_embeddings():
    """A constant predicted embedding pays the full variance hinge; a spread one pays less."""
    cfg = EmbeddingLossConfig(use_embedding_mse=False, use_curve_recon=False, use_embedding_variance=True, weight_embedding_variance=1.0, variance_gamma=1.0)
    criterion = make_criterion(cfg)

    gt        = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    collapsed = torch.zeros(2, EMBEDDING_DIM, SPATIAL, SPATIAL)
    spread    = torch.randn(2, EMBEDDING_DIM, SPATIAL, SPATIAL) * 3.0

    v_collapsed = criterion(collapsed, gt)["components"]["embedding_variance"].item()
    v_spread    = criterion(spread, gt)["components"]["embedding_variance"].item()

    assert v_collapsed == pytest.approx(1.0, abs=1e-2)
    assert v_spread < v_collapsed


def test_covariance_penalty_vanishes_for_decorrelated_dimensions():
    """Correlated embedding dimensions pay more covariance penalty than independent ones."""
    cfg = EmbeddingLossConfig(use_embedding_mse=False, use_curve_recon=False, use_embedding_covariance=True, weight_embedding_covariance=1.0)
    criterion = make_criterion(cfg)

    gt   = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    base = torch.randn(2, 1, SPATIAL, SPATIAL)

    correlated  = base.expand(-1, EMBEDDING_DIM, -1, -1).contiguous()
    independent = torch.randn(2, EMBEDDING_DIM, SPATIAL, SPATIAL)

    c_corr = criterion(correlated, gt)["components"]["embedding_covariance"].item()
    c_ind  = criterion(independent, gt)["components"]["embedding_covariance"].item()

    assert c_corr > c_ind


def test_masking_tag_encodes_strategy_and_ratio():
    """The run-name masking fragment carries the strategy and ratio."""
    assert RunNaming.masking_tag(JepaMaskingConfig(enabled=True, strategy="block", mask_ratio=0.5)) == "mask_block0.5"


def test_trainer_shim_defaults_to_masking_off():
    """The plain trainer path stays unmasked and loss-complete."""
    trainer = make_trainer_shim()

    images = torch.rand(2, 2, SPATIAL, SPATIAL)
    gt     = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    out    = trainer._compute_loss((images, gt))

    assert torch.isfinite(out["total_loss"])


def _masked_trainer(loss_on):
    """Returns a trainer shim with random masking enabled and the given loss scope."""
    trainer              = make_trainer_shim()
    trainer.masking      = JepaMaskingConfig(enabled=True, strategy="random", mask_ratio=0.5, loss_on=loss_on)
    trainer.mask_sampler = MaskSampler(trainer.masking)
    trainer.input_masker = InputMasker(trainer.masking)

    return trainer


def test_eval_pass_is_unmasked_and_takes_the_loss_over_the_full_input(monkeypatch):
    """In eval mode no mask reaches the criterion and the input is not blanked."""
    trainer = _masked_trainer("masked")
    trainer.model.eval()

    captured  = {}
    criterion = trainer.criterion
    forward   = trainer.model.forward

    def spy(pred, gt, mask=None):
        captured["mask"] = mask
        return criterion(pred, gt, mask)

    def forward_spy(images):
        captured["images"] = images
        return forward(images)

    monkeypatch.setattr(trainer, "criterion", spy)
    monkeypatch.setattr(trainer.model, "forward", forward_spy)

    images = torch.rand(2, 2, SPATIAL, SPATIAL) + 1.0
    gt     = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)
    out    = trainer._compute_loss((images, gt))

    assert captured["mask"] is None
    assert torch.equal(captured["images"], images)
    assert torch.isfinite(out["total_loss"])


def test_eval_loss_is_deterministic_across_repeated_passes():
    """Two eval channels over the same batch return the same loss, unlike the masked training path."""
    trainer = _masked_trainer("masked")
    trainer.model.eval()

    images = torch.rand(2, 2, SPATIAL, SPATIAL) + 1.0
    gt     = torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)

    first  = trainer._compute_loss((images, gt))["total_loss"].item()
    second = trainer._compute_loss((images, gt))["total_loss"].item()

    assert first == pytest.approx(second, rel=1e-6)


def test_training_pass_still_masks_after_an_eval_pass():
    """Returning to train mode restores masking, so the eval guard is not sticky."""
    trainer = _masked_trainer("all")
    trainer.model.eval()
    trainer._compute_loss((torch.rand(2, 2, SPATIAL, SPATIAL), torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)))

    trainer.model.train()

    seen    = []
    forward = trainer.model.forward

    def forward_spy(images):
        seen.append(images)
        return forward(images)

    trainer.model.forward = forward_spy

    images = torch.full((2, 2, SPATIAL, SPATIAL), 1.0)
    trainer._compute_loss((images, torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)))

    assert not torch.equal(seen[-1], images)


def test_epoch_start_restores_masking_after_an_evaluation_pass():
    """The trainer's own epoch start puts the module back into train mode after evaluate() left it in eval, so masking applies beyond the first epoch, while a frozen autoencoder stays in eval."""
    trainer              = make_trainer_shim(trainable=False)
    trainer.masking      = JepaMaskingConfig(enabled=True, strategy="random", mask_ratio=0.5, loss_on="all")
    trainer.mask_sampler = MaskSampler(trainer.masking)
    trainer.input_masker = InputMasker(trainer.masking)

    trainer.model.eval()
    assert not trainer.model.training

    trainer._set_train_mode()

    assert trainer.model.training
    assert trainer.model.backbone.training
    assert not trainer.model.profile_autoencoder.training

    trainer._compute_loss((torch.rand(2, 2, SPATIAL, SPATIAL), torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)))

    assert trainer.masked_batches == 1
    assert trainer.train_batches  == 1


def test_masked_batches_monitor_the_realised_hidden_fraction():
    """Each masked training batch reports the realised hidden-pixel fraction of its mask among the loss monitors."""
    trainer = _masked_trainer("all")
    trainer.model.train()

    out = trainer._compute_loss((torch.rand(2, 2, SPATIAL, SPATIAL), torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)))

    fraction = float(out["monitor"]["mask_hidden_fraction"])
    assert 0.0 < fraction <= 1.0


def test_unmasked_evaluation_batches_carry_no_hidden_fraction():
    """An evaluation pass is unmasked, so no hidden fraction is reported for it."""
    trainer = _masked_trainer("all")
    trainer.model.eval()

    out = trainer._compute_loss((torch.rand(2, 2, SPATIAL, SPATIAL), torch.rand(2, 3 * N_GAUSSIANS, SPATIAL, SPATIAL)))

    assert "mask_hidden_fraction" not in out["monitor"]


def test_masking_table_states_the_nominal_ratio_and_the_unmasked_validation():
    """The [Input Masking] rows call the ratio nominal, point at the tracked realised fraction and say validation is unmasked over all pixels."""
    trainer         = make_trainer_shim()
    trainer.masking = JepaMaskingConfig(enabled=True, strategy="block", mask_ratio=0.4, n_blocks=3, loss_on="masked")

    rows = trainer._masking_table()

    assert rows["Mask ratio"].startswith("0.4 nominal (3 blocks")
    assert "mask_hidden_fraction" in rows["Mask ratio"]
    assert rows["Validation"].startswith("unmasked, loss over all pixels")
    assert rows["Loss on"] == "masked"


def test_epoch_log_refuses_an_unmasked_epoch():
    """An epoch that trained batches without masking any of them raises instead of passing as a masked run."""
    trainer         = _masked_trainer("all")
    trainer.logger  = NullLogger()
    trainer.tracker = NullTracker()

    trainer.train_batches, trainer.masked_batches = 3, 3
    trainer._log_train_epoch_extra(0.1, epoch=0)

    trainer.train_batches, trainer.masked_batches = 3, 0
    with pytest.raises(RuntimeError, match="none of the 3 training batches"):
        trainer._log_train_epoch_extra(0.1, epoch=1)
