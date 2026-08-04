from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from pipelines.dataloader_tuning.adapters import (
    DEFAULT_MODEL,
    FEED_ADAPTERS,
    FEED_MODES,
    BackboneFeedAdapter,
    FeedLosses,
    FeedTarget,
    ImageFeedAdapter,
    ProfileFeedAdapter,
    SyntheticCurveDataset,
    SyntheticFeedAdapter,
    build_feed_target,
)
from tools.data.gaussians import GaussianMixture

from tests.conftest import SilentLogger


def test_feed_modes_match_adapter_registry():
    assert set(FEED_MODES) == set(FEED_ADAPTERS)


def test_default_model_defined_for_each_mode():
    assert set(DEFAULT_MODEL) == set(FEED_MODES)


def test_synthetic_dataset_length_and_curve_shape():
    dataset = SyntheticCurveDataset(n_samples=12, profile_length=48, seed=0)

    assert len(dataset) == 12

    curve = dataset[0]
    assert curve.shape == (48,)
    assert curve.dtype == np.float32


def test_synthetic_dataset_x_axis_is_linspace_zero_to_one():
    dataset = SyntheticCurveDataset(n_samples=4, profile_length=16, seed=0)

    assert dataset.x_axis.shape == (16,)
    assert dataset.x_axis[0]    == pytest.approx(0.0)
    assert dataset.x_axis[-1]   == pytest.approx(1.0)


def test_synthetic_dataset_uses_gaussian_mixture_synthesis():
    dataset = SyntheticCurveDataset(n_samples=4, profile_length=24, seed=7)

    rng        = np.random.default_rng(7 + 0)
    amplitudes = rng.uniform(0.2, 1.0, size=(1, 3)).astype(np.float32)
    means      = rng.uniform(0.2, 0.8, size=(1, 3)).astype(np.float32)
    sigmas     = rng.uniform(0.02, 0.08, size=(1, 3)).astype(np.float32)
    expected   = GaussianMixture.evaluate_batch(dataset.x_axis, amplitudes, means, sigmas)[0]

    assert np.allclose(dataset[0], expected)


def test_synthetic_dataset_is_deterministic_per_index():
    dataset = SyntheticCurveDataset(n_samples=4, profile_length=24, seed=3)

    assert np.array_equal(dataset[2], dataset[2])


def test_feed_losses_are_mse_surrogates_not_trainer_loss():
    class _Exploding:
        def __call__(self, *args, **kwargs):
            raise AssertionError("the dataloader feed must stay a cheap MSE surrogate, never the trainer loss")

    class _Offset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = _Exploding()

        def reconstruct(self, x):
            return x + 2.0, None

        def forward(self, x):
            return x + 3.0

        def _compute_loss(self, *args, **kwargs):
            raise AssertionError("the dataloader feed must stay a cheap MSE surrogate, never the trainer loss")

    model  = _Offset()
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert float(FeedLosses.reconstruction(model, inputs)) == pytest.approx(4.0)
    assert float(FeedLosses.supervised(model, inputs))     == pytest.approx(float(((inputs + 3.0) ** 2).mean()))


def test_reconstruction_loss_is_zero_for_perfect_model():
    class Identity(torch.nn.Module):
        def reconstruct(self, x):
            return x, None

    loss = FeedLosses.reconstruction(Identity(), torch.ones(2, 4))

    assert float(loss) == pytest.approx(0.0)


def test_supervised_loss_targets_zeros():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(3, 5)

    loss = FeedLosses.supervised(Constant(), torch.ones(3, 5))

    assert float(loss) == pytest.approx(0.0)


def test_synthetic_adapter_builds_feed_target(logger_stub, tmp_path):
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=32, synthetic_samples=20)

    target = SyntheticFeedAdapter(config, tmp_path, logger_stub).build()

    assert isinstance(target, FeedTarget)
    assert target.model_name   == DEFAULT_MODEL["synthetic"]
    assert target.forward_loss == FeedLosses.reconstruction
    assert len(target.dataset) == 20


def test_synthetic_adapter_to_model_input_adds_spatial_dims(logger_stub, tmp_path):
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=16, synthetic_samples=8)
    target = SyntheticFeedAdapter(config, tmp_path, logger_stub).build()

    batch       = torch.stack([torch.from_numpy(target.dataset[i]) for i in range(4)])
    model_input = target.to_model_input(batch, torch.device("cpu"))

    assert model_input.shape == (4, 16, 1, 1)


def test_synthetic_adapter_loss_runs_through_model(logger_stub, tmp_path):
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=16, synthetic_samples=8)
    target = SyntheticFeedAdapter(config, tmp_path, logger_stub).build()

    batch       = torch.stack([torch.from_numpy(target.dataset[i]) for i in range(4)])
    model_input = target.to_model_input(batch, torch.device("cpu"))
    loss        = target.forward_loss(target.model, model_input)

    assert torch.isfinite(loss)


def test_synthetic_adapter_respects_explicit_model_name(logger_stub, tmp_path):
    config = DataLoaderTuningEntryConfig(mode="synthetic", model_name="conv1d_ae", synthetic_length=32)

    adapter = SyntheticFeedAdapter(config, tmp_path, logger_stub)

    assert adapter.model_name == "conv1d_ae"


def test_build_feed_target_unknown_mode_raises(logger_stub, tmp_path):
    config = DataLoaderTuningEntryConfig(mode="not_a_mode")

    with pytest.raises(ValueError):
        build_feed_target(config, tmp_path, logger_stub)


def test_build_feed_target_dispatches_to_synthetic(logger_stub, tmp_path):
    config = DataLoaderTuningEntryConfig(mode="synthetic", synthetic_length=16, synthetic_samples=8)

    target = build_feed_target(config, tmp_path, logger_stub)

    assert "synthetic" in target.item_source.lower()


def test_real_data_adapters_default_models_are_distinct():
    assert ProfileFeedAdapter(DataLoaderTuningEntryConfig(mode="profile_autoencoder"), ".", SilentLogger()).model_name == DEFAULT_MODEL["profile_autoencoder"]
    assert ImageFeedAdapter(DataLoaderTuningEntryConfig(mode="image_autoencoder"), ".", SilentLogger()).model_name     == DEFAULT_MODEL["image_autoencoder"]
    assert BackboneFeedAdapter(DataLoaderTuningEntryConfig(mode="backbone"), ".", SilentLogger()).model_name           == DEFAULT_MODEL["backbone"]
