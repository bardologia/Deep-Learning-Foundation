"""Tests for JepaModule forward routing and the JEPA Trainer's coupling validation and step.

Covers routing through the backbone and the optional image-autoencoder front end, the rules
relating coupling mode, target provider, embedding normalisation and the curve reconstruction
term, the loss step on a tiny batch in both the embedding and parameter regimes, an optimizer
step updating the online backbone, and checkpoint round-tripping."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from configuration.architectures                         import Conv2dImageAutoencoderConfig
from configuration.training.jepa                         import EmbeddingLossConfig, JepaMaskingConfig
from models.image_autoencoder                            import get_image_autoencoder
from pipelines.jepa.training.coupling                    import CouplingMode, TargetProvider
from pipelines.jepa.training.loss                        import Loss as EmbeddingLoss
from pipelines.jepa.training.trainer                     import JepaModule, Trainer
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats

from tests.conftest      import SilentLogger
from tests.jepa.conftest import EMBEDDING_DIM, IdentityNormStats, N_GAUSSIANS, SPATIAL, make_autoencoder


def build_module():
    """Builds a JepaModule pairing a one-by-one convolution backbone with an unnormalised profile autoencoder."""
    autoencoder = make_autoencoder("none")
    backbone    = nn.Conv2d(2, EMBEDDING_DIM, kernel_size=1)
    return JepaModule(backbone, profile_autoencoder=autoencoder, image_autoencoder=None)


def test_jepa_module_forward_routes_through_backbone():
    """The module maps images to an embedding map of shape (batch, EMBEDDING_DIM, height, width)."""
    module = build_module()
    images = torch.randn(2, 2, SPATIAL, SPATIAL)

    out = module(images)

    assert out.shape == (2, EMBEDDING_DIM, SPATIAL, SPATIAL)


def test_jepa_module_image_autoencoder_disabled_by_default():
    """Building with only a profile autoencoder leaves the image branch absent."""
    module = build_module()

    assert module.image_autoencoder   is None
    assert module.profile_autoencoder is not None


def test_jepa_module_forward_routes_through_image_autoencoder():
    """With an image autoencoder attached the backbone consumes its embedding and the output keeps the input resolution."""
    image_cfg   = Conv2dImageAutoencoderConfig(in_channels=2, embedding_dim=4, base_channels=4, depth=1, downsample_factor=2)
    image_ae, _ = get_image_autoencoder("conv2d_ae", image_cfg)
    backbone    = nn.Conv2d(4, EMBEDDING_DIM, kernel_size=1)
    module      = JepaModule(backbone, profile_autoencoder=make_autoencoder("none"), image_autoencoder=image_ae)
    module.eval()

    images = torch.randn(2, 2, 8, 8)

    with torch.no_grad():
        out = module(images)

    assert out.shape == (2, EMBEDDING_DIM, 8, 8)


def test_validate_coupling_live_requires_finetune():
    """A live target provider over a frozen encoder is rejected."""
    frozen = CouplingMode("frozen", "profile autoencoder")
    cfg    = EmbeddingLossConfig(use_curve_recon=True)

    with pytest.raises(ValueError):
        Trainer.validate_coupling(frozen, "live", cfg, make_autoencoder("none"))


def test_validate_coupling_live_requires_curve_recon():
    """A live target provider without the curve reconstruction term is rejected."""
    finetune = CouplingMode("finetune", "profile autoencoder")
    cfg      = EmbeddingLossConfig(use_curve_recon=False)

    with pytest.raises(ValueError):
        Trainer.validate_coupling(finetune, "live", cfg, make_autoencoder("none"))


def test_validate_coupling_stopgrad_frozen_is_valid():
    """Stopgrad over a frozen encoder passes validation."""
    frozen = CouplingMode("frozen", "profile autoencoder")
    cfg    = EmbeddingLossConfig(use_curve_recon=True)

    Trainer.validate_coupling(frozen, "stopgrad", cfg, make_autoencoder("none"))


def test_validate_coupling_finetune_layernorm_requires_curve_recon():
    """Finetuning a LayerNorm-normalised embedding without curve reconstruction is rejected."""
    finetune = CouplingMode("finetune", "profile autoencoder")
    cfg      = EmbeddingLossConfig(use_curve_recon=False)

    with pytest.raises(ValueError, match="LayerNorm"):
        Trainer.validate_coupling(finetune, "stopgrad", cfg, make_autoencoder("layernorm"))


def test_validate_coupling_finetune_layernorm_valid_with_curve_recon():
    """Finetuning a LayerNorm embedding passes once curve reconstruction is on."""
    finetune = CouplingMode("finetune", "profile autoencoder")
    cfg      = EmbeddingLossConfig(use_curve_recon=True)

    Trainer.validate_coupling(finetune, "stopgrad", cfg, make_autoencoder("layernorm"))


def test_validate_coupling_frozen_layernorm_valid_without_curve_recon():
    """A frozen LayerNorm embedding needs no curve reconstruction term."""
    frozen = CouplingMode("frozen", "profile autoencoder")
    cfg    = EmbeddingLossConfig(use_curve_recon=False)

    Trainer.validate_coupling(frozen, "stopgrad", cfg, make_autoencoder("layernorm"))


def test_validate_coupling_finetune_parameterless_norm_valid_without_curve_recon():
    """Finetuning a parameterless L2 embedding needs no curve reconstruction term."""
    finetune = CouplingMode("finetune", "profile autoencoder")
    cfg      = EmbeddingLossConfig(use_curve_recon=False)

    Trainer.validate_coupling(finetune, "stopgrad", cfg, make_autoencoder("l2"))


def make_trainer_shim(target_kind="stopgrad", trainable=True):
    """Assembles a Trainer with only the fields the loss step touches.

    Args:
        target_kind : Target provider kind handed to the embedding loss.
        trainable   : Whether the profile autoencoder is finetuned rather than frozen.

    Returns:
        A Trainer instance carrying the module, device, coupling mode and criterion.
    """
    module = build_module()
    mode   = CouplingMode("finetune" if trainable else "frozen", "profile autoencoder")
    mode.apply(module.profile_autoencoder)

    z_axis = torch.linspace(-4.0, 4.0, module.profile_autoencoder.config.profile_length)

    norm_stats         = IdentityNormStats()
    profile_normalizer = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    emb_cfg            = EmbeddingLossConfig(use_embedding_mse=True, weight_embedding_mse=1.0, use_curve_recon=False)

    provider  = TargetProvider(target_kind)
    criterion = EmbeddingLoss(module.profile_autoencoder, provider, emb_cfg, z_axis, norm_stats, 3, profile_normalizer, SilentLogger())

    trainer                = Trainer.__new__(Trainer)
    trainer.model          = module
    trainer.device         = torch.device("cpu")
    trainer.has_profile    = True
    trainer.has_image      = False
    trainer.profile_mode   = mode
    trainer.criterion      = criterion
    trainer.masking        = JepaMaskingConfig()
    trainer.masked_batches = 0
    trainer.train_batches  = 0
    return trainer


def test_compute_loss_runs_on_tiny_batch():
    """The loss step returns a finite total on a two-sample batch."""
    trainer = make_trainer_shim()
    images  = torch.randn(2, 2, SPATIAL, SPATIAL)
    gt      = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    out = trainer._compute_loss((images, gt))

    assert torch.isfinite(out["total_loss"])


def test_compute_loss_param_path_calls_the_criterion():
    """On the parameter path the criterion receives the prediction and targets."""
    trainer             = make_trainer_shim()
    trainer.has_profile = False

    captured = {}

    def criterion(pred, gt):
        """Records the targets it received and returns a zero total."""
        captured["gt"] = gt
        return {"total_loss": torch.zeros(())}

    trainer.criterion = criterion

    images = torch.randn(2, 2, SPATIAL, SPATIAL)
    gt     = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    trainer._compute_loss((images, gt))

    assert torch.equal(captured["gt"], gt)


def test_optimizer_step_updates_online_branch():
    """One SGD step changes at least one backbone parameter."""
    trainer   = make_trainer_shim()
    params    = [p for p in trainer.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.1)

    images = torch.randn(2, 2, SPATIAL, SPATIAL)
    gt     = torch.rand(2, N_GAUSSIANS * 3, SPATIAL, SPATIAL)

    before = [p.detach().clone() for p in trainer.model.backbone.parameters()]

    optimizer.zero_grad(set_to_none=True)
    loss = trainer._compute_loss((images, gt))["total_loss"]
    loss.backward()
    optimizer.step()

    after   = list(trainer.model.backbone.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed


def test_checkpoint_state_dict_round_trip(tmp_path):
    """A module restored from a saved state dict reproduces the original output."""
    module = build_module()
    images = torch.randn(2, 2, SPATIAL, SPATIAL)

    state = {"epoch": 1, "params": module.state_dict(), "z_axis": torch.linspace(-4, 4, 8).numpy()}
    path  = tmp_path / "best_model.pt"
    torch.save(state, path)

    loaded   = torch.load(path, map_location="cpu", weights_only=False)
    restored = build_module()
    restored.load_state_dict(loaded["params"])

    with torch.no_grad():
        assert torch.allclose(module(images), restored(images))


def test_validate_coupling_live_rejects_zero_curve_recon_weight():
    """A live provider with the curve reconstruction weight at zero is rejected."""
    finetune = CouplingMode("finetune", "profile autoencoder")
    cfg      = EmbeddingLossConfig(use_curve_recon=True, weight_curve_recon=0.0)

    with pytest.raises(ValueError, match="weight_curve_recon"):
        Trainer.validate_coupling(finetune, "live", cfg, make_autoencoder("none"))


def test_validate_coupling_finetune_layernorm_rejects_zero_curve_recon_weight():
    """Finetuning a LayerNorm embedding with a zero curve reconstruction weight is rejected."""
    finetune = CouplingMode("finetune", "profile autoencoder")
    cfg      = EmbeddingLossConfig(use_curve_recon=True, weight_curve_recon=0.0)

    with pytest.raises(ValueError, match="weight_curve_recon"):
        Trainer.validate_coupling(finetune, "stopgrad", cfg, make_autoencoder("layernorm"))
