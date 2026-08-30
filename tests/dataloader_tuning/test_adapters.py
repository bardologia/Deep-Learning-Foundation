"""Tests for the feed adapters that supply data to the dataloader tuner.

Covers the adapter registry and default models per mode, the synthetic Gaussian
curve dataset, the cheap MSE surrogate losses that stand in for the real trainer
loss, and the synthetic adapter's feed target construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from pipelines.dataloader_tuning.adapters import (
    BackboneFeedAdapter,
    FeedLosses,
    FeedTarget,
    FeedTargetFactory,
    ImageFeedAdapter,
    ProfileFeedAdapter,
    SyntheticCurveDataset,
    SyntheticFeedAdapter,
)
from tools.data.gaussians import GaussianMixture

from tests.conftest import SilentLogger


def test_feed_modes_match_adapter_registry():
    """The advertised feed modes are exactly the registered adapter keys."""
    assert set(FeedTargetFactory.modes()) == set(FeedTargetFactory.ADAPTERS)


def test_default_model_defined_for_each_mode():
    """Every feed mode's adapter carries a non-empty default model name."""
    assert all(FeedTargetFactory.adapter(mode).DEFAULT_MODEL for mode in FeedTargetFactory.modes())


def test_synthetic_dataset_length_and_curve_shape():
    """The synthetic dataset yields the requested count of float32 curves of the requested length."""
    dataset = SyntheticCurveDataset(n_samples=12, profile_length=48, seed=0)

    assert len(dataset) == 12

    curve = dataset[0]
    assert curve.shape == (48,)
    assert curve.dtype == np.float32


def test_synthetic_dataset_x_axis_is_linspace_zero_to_one():
    """The synthetic sampling axis spans the unit interval inclusively."""
    dataset = SyntheticCurveDataset(n_samples=4, profile_length=16, seed=0)

    assert dataset.z_axis.shape == (16,)
    assert dataset.z_axis[0]    == pytest.approx(0.0)
    assert dataset.z_axis[-1]   == pytest.approx(1.0)


def test_synthetic_dataset_uses_gaussian_mixture_synthesis():
    """A synthetic curve matches a three-component Gaussian mixture drawn from the same seeded generator."""
    dataset = SyntheticCurveDataset(n_samples=4, profile_length=24, seed=7)

    rng        = np.random.default_rng(7 + 0)
    amplitudes = rng.uniform(0.2, 1.0, size=(1, 3)).astype(np.float32)
    means      = rng.uniform(0.2, 0.8, size=(1, 3)).astype(np.float32)
    sigmas     = rng.uniform(0.02, 0.08, size=(1, 3)).astype(np.float32)
    expected   = GaussianMixture.evaluate_batch(dataset.z_axis, amplitudes, means, sigmas)[0]

    assert np.allclose(dataset[0], expected)


def test_synthetic_dataset_is_deterministic_per_index():
    """Repeated access to the same index returns the identical curve."""
    dataset = SyntheticCurveDataset(n_samples=4, profile_length=24, seed=3)

    assert np.array_equal(dataset[2], dataset[2])


def test_feed_losses_are_mse_surrogates_not_trainer_loss():
    """The feed losses compute plain MSE against the reconstruction or zeros, never invoking the model's own loss."""
    class _Exploding:
        """Callable that fails the test if the trainer criterion is ever invoked."""
        def __call__(self, *args, **kwargs):
            """Raises to signal that the trainer loss was called from the feed path."""
            raise AssertionError("the dataloader feed must stay a cheap MSE surrogate, never the trainer loss")

    class _Offset(torch.nn.Module):
        """Module whose reconstruction and forward outputs are fixed offsets of the input."""
        def __init__(self):
            """Attaches the exploding criterion that must never be called."""
            super().__init__()
            self.criterion = _Exploding()

        def reconstruct(self, x):
            """Returns the input shifted by two and no latent."""
            return x + 2.0, None

        def forward(self, x):
            """Returns the input shifted by three."""
            return x + 3.0

        def _compute_loss(self, *args, **kwargs):
            """Raises to signal that the trainer loss was called from the feed path."""
            raise AssertionError("the dataloader feed must stay a cheap MSE surrogate, never the trainer loss")

    model  = _Offset()
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert float(FeedLosses.reconstruction(model, inputs)) == pytest.approx(4.0)
    assert float(FeedLosses.supervised(model, inputs))     == pytest.approx(float(((inputs + 3.0) ** 2).mean()))


def test_reconstruction_loss_is_zero_for_perfect_model():
    """The reconstruction surrogate is zero when the model returns its input unchanged."""
    class Identity(torch.nn.Module):
        """Module whose reconstruction returns the input unchanged."""
        def reconstruct(self, x):
            """Returns the input unchanged and no latent."""
            return x, None

    loss = FeedLosses.reconstruction(Identity(), torch.ones(2, 4))

    assert float(loss) == pytest.approx(0.0)


def test_supervised_loss_targets_zeros():
    """The supervised surrogate is zero when the model outputs all zeros."""
    class Constant(torch.nn.Module):
        """Module whose forward output is a fixed zero tensor of shape (3, 5)."""
        def forward(self, x):
            """Returns a zero tensor of shape (3, 5)."""
            return torch.zeros(3, 5)

    loss = FeedLosses.supervised(Constant(), torch.ones(3, 5))

    assert float(loss) == pytest.approx(0.0)


def test_synthetic_adapter_builds_feed_target(logger_stub, tmp_path):
    """The synthetic adapter builds a feed target with the default model, reconstruction loss and requested sample count."""
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=32, synthetic_samples=20)

    target = SyntheticFeedAdapter(config, tmp_path, logger_stub).build()

    assert isinstance(target, FeedTarget)
    assert target.model_name   == FeedTargetFactory.adapter("synthetic").DEFAULT_MODEL
    assert target.forward_loss == FeedLosses.reconstruction
    assert len(target.dataset) == 20


def test_synthetic_adapter_to_model_input_adds_spatial_dims(logger_stub, tmp_path):
    """Curves are reshaped to (batch, profile_length, 1, 1) before entering the model."""
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=16, synthetic_samples=8)
    target = SyntheticFeedAdapter(config, tmp_path, logger_stub).build()

    batch       = torch.stack([torch.from_numpy(target.dataset[i]) for i in range(4)])
    model_input = target.to_model_input(batch, torch.device("cpu"))

    assert model_input.shape == (4, 16, 1, 1)


def test_synthetic_adapter_loss_runs_through_model(logger_stub, tmp_path):
    """The feed target's loss evaluates to a finite value on a synthetic batch."""
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=16, synthetic_samples=8)
    target = SyntheticFeedAdapter(config, tmp_path, logger_stub).build()

    batch       = torch.stack([torch.from_numpy(target.dataset[i]) for i in range(4)])
    model_input = target.to_model_input(batch, torch.device("cpu"))
    loss        = target.forward_loss(target.model, model_input)

    assert torch.isfinite(loss)


def test_synthetic_adapter_respects_explicit_model_name(logger_stub, tmp_path):
    """An explicit model name in the config overrides the mode default."""
    config = DataLoaderTuningEntryConfig(mode="synthetic", model_name="conv1d_ae", synthetic_length=32)

    adapter = SyntheticFeedAdapter(config, tmp_path, logger_stub)

    assert adapter.model_name == "conv1d_ae"


def test_feed_target_factory_unknown_mode_raises(logger_stub, tmp_path):
    """Building a feed target for an unregistered mode raises ValueError."""
    config = DataLoaderTuningEntryConfig(mode="not_a_mode")

    with pytest.raises(ValueError):
        FeedTargetFactory.build(config, tmp_path, logger_stub)


def test_feed_target_factory_dispatches_to_synthetic(logger_stub, tmp_path):
    """The factory routes the synthetic mode to the synthetic adapter."""
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=16, synthetic_samples=8)

    target = FeedTargetFactory.build(config, tmp_path, logger_stub)

    assert "synthetic" in target.item_source.lower()


def test_real_data_adapters_default_models_are_distinct():
    """Profile, image and backbone adapters each adopt their own mode's default model."""
    assert ProfileFeedAdapter(DataLoaderTuningEntryConfig(mode="profile_autoencoder"), ".", SilentLogger()).model_name == FeedTargetFactory.adapter("profile_autoencoder").DEFAULT_MODEL
    assert ImageFeedAdapter(DataLoaderTuningEntryConfig(mode="image_autoencoder"), ".", SilentLogger()).model_name     == FeedTargetFactory.adapter("image_autoencoder").DEFAULT_MODEL
    assert BackboneFeedAdapter(DataLoaderTuningEntryConfig(mode="backbone"), ".", SilentLogger()).model_name           == FeedTargetFactory.adapter("backbone").DEFAULT_MODEL
