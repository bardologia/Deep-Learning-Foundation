"""Tests for the linear probes that read scene content out of intermediate activations.

Covers the cross-validated ridge probe, the participation-ratio geometry measure, and
the probe core that hooks named layers, samples pixels from feature maps and scores
component count, presence and total amplitude against an input baseline and shuffled
controls.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.layer_probes import FeatureGeometry, FeatureSampler, LayerProbeCore, RidgeProbe


H, W = 16, 12


def test_ridge_probe_recovers_a_linear_signal():
    """A linear target is decoded with a cross-validated R2 above 0.95 and low fold spread."""
    rng = np.random.default_rng(0)
    X   = rng.normal(size=(400, 8))
    y   = X @ rng.normal(size=8) + 0.01 * rng.normal(size=400)

    score = RidgeProbe(ridge_lambda=1e-3).score(X, y)

    assert score["mean"] > 0.95
    assert score["std"]  < 0.05


def test_ridge_probe_scores_noise_near_zero():
    """An unrelated target scores near zero."""
    rng = np.random.default_rng(1)
    X   = rng.normal(size=(400, 8))
    y   = rng.normal(size=400)

    assert abs(RidgeProbe().score(X, y)["mean"]) < 0.2


def test_ridge_probe_needs_enough_samples():
    """Too few samples for the fold count raises ValueError."""
    with pytest.raises(ValueError):
        RidgeProbe().score(np.zeros((5, 3)), np.zeros(5))


def test_ridge_probe_needs_two_folds():
    """A single-fold probe is rejected at construction."""
    with pytest.raises(ValueError):
        RidgeProbe(n_folds=1)


def test_participation_ratio_separates_isotropic_from_collapsed():
    """Isotropic features fill the dimensions while a rank-one cloud scores one."""
    rng       = np.random.default_rng(2)
    isotropic = rng.normal(size=(2000, 8))
    collapsed = np.outer(rng.normal(size=2000), rng.normal(size=8))

    assert FeatureGeometry.participation_ratio(isotropic) > 7.0
    assert FeatureGeometry.participation_ratio(collapsed) == pytest.approx(1.0, abs=0.05)


class _TwoLayerNet(torch.nn.Module):
    """Network exposing one layer that carries the input and one that only sees zeros."""
    def __init__(self) -> None:
        """Builds the informative and the uninformative identity layers."""
        super().__init__()
        self.useful  = torch.nn.Identity()
        self.useless = torch.nn.Identity()

    def forward(self, x):
        """Returns x through the informative layer while feeding zeros through the other one."""
        a = self.useful(x)
        _ = self.useless(torch.zeros_like(x))
        return a


def _batches(n: int = 3):
    """Builds batches whose input encodes the component count and whose labels follow it.

    Args:
        n : Number of batches to build.

    Returns:
        List of (images, params) pairs, images of shape (2, 2, H, W) and params of
        shape (2, 6, H, W) with two Gaussian slots activated by the count channel.
    """
    rng     = np.random.default_rng(2)
    batches = []

    for _ in range(n):
        counts  = rng.integers(0, 3, size=(2, H, W)).astype(np.float32)
        images  = torch.from_numpy(np.stack([counts, rng.normal(size=(2, H, W)).astype(np.float32)], axis=1))

        gt = np.zeros((2, 6, H, W), dtype=np.float32)
        gt[:, 0] = (counts >= 1).astype(np.float32)
        gt[:, 3] = (counts >= 2).astype(np.float32)
        gt[:, 1] = 20.0
        gt[:, 4] = 40.0
        gt[:, 2] = 2.0
        gt[:, 5] = 3.0

        batches.append((images, gt))

    return batches


def _core(model, layers, samples: int = 256, seed: int = 0) -> LayerProbeCore:
    """Builds a probe core over the named layers of a model.

    Args:
        model   : Module whose named submodules are hooked.
        layers  : Names of the submodules to probe.
        samples : Number of pixels sampled per batch.
        seed    : Seed shared by the pixel sampler and the ridge probe.

    Returns:
        A probe core configured for two-slot, three-parameter-per-Gaussian labels.
    """
    return LayerProbeCore(
        model             = model,
        layers            = layers,
        ppg               = 3,
        amp_thr           = 1e-3,
        samples_per_batch = samples,
        probe             = RidgeProbe(ridge_lambda=1e-3, seed=seed),
        seed              = seed,
    )


def test_probe_core_finds_the_informative_layer():
    """The layer carrying the input decodes the component count while the zero-fed layer does not."""
    results = _core(_TwoLayerNet(), ["useful", "useless"]).collect(_batches())

    assert results["layers"]["useful"]["scores"]["count"]["mean"]  > 0.9
    assert results["layers"]["useless"]["scores"]["count"]["mean"] < 0.2
    assert results["layers"]["useful"]["n_channels"] == 2


def test_probe_core_scores_all_five_targets():
    """Every declared probe target is scored, with second-slot presence matching its base rate."""
    results = _core(_TwoLayerNet(), ["useful"]).collect(_batches())
    scores  = results["layers"]["useful"]["scores"]

    assert set(scores) == {target for target, _label in LayerProbeCore.TARGETS}
    assert scores["second_presence"]["mean"] == pytest.approx(0.75, abs=0.05)
    assert scores["total_amp"]["mean"]       > 0.9


def test_input_baseline_decodes_what_the_input_carries():
    """The raw-input baseline decodes the count from its two channels."""
    results  = _core(_TwoLayerNet(), ["useful"]).collect(_batches())
    baseline = results["input_baseline"]

    assert baseline["n_channels"] == 2
    assert baseline["scores"]["count"]["mean"] > 0.9


def test_shuffled_controls_sit_near_zero():
    """Shuffling the labels of the best layer drives its score to zero."""
    results = _core(_TwoLayerNet(), ["useful", "useless"]).collect(_batches())

    control = results["shuffled_controls"]["count"]

    assert control["layer"] == "useful"
    assert abs(control["score"]) < 0.2


def test_feature_map_layouts_are_accepted_and_rejected():
    """Only batch-first, channel-second tensors on a spatial grid count as feature maps."""
    sampler = FeatureSampler(64)
    B       = 2

    assert sampler.is_feature_map(torch.zeros(B, 8, H, W),         B, H, W)
    assert sampler.is_feature_map(torch.zeros(B, 8, H // 4, W // 4), B, H, W)

    assert not sampler.is_feature_map(torch.zeros(6 * B, 4, 49, 49), B, H, W)
    assert not sampler.is_feature_map(torch.zeros(B, H, W, 96),      B, H, W)
    assert not sampler.is_feature_map(torch.zeros(B, 8, H),          B, H, W)


def test_features_at_rejects_a_tensor_that_is_not_a_feature_map():
    """Sampling features from a non-feature-map tensor raises ValueError."""
    sampler = FeatureSampler(16)
    b, i, j = sampler.sample_coords(2, H, W)

    with pytest.raises(ValueError):
        sampler.features_at(torch.zeros(12, 4, 49, 49), b, i, j, 2, H, W)


def test_probe_core_drops_channels_last_activations():
    """Channels-last activations are skipped instead of being probed."""
    class _ChannelsLastNet(torch.nn.Module):
        """Network emitting one spatial activation and one channels-last activation."""
        def __init__(self) -> None:
            """Builds the spatial and the channels-last identity layers."""
            super().__init__()
            self.spatial      = torch.nn.Identity()
            self.channel_last = torch.nn.Identity()

        def forward(self, x):
            """Returns x through the spatial layer while passing a permuted copy through the other one."""
            a = self.spatial(x)
            _ = self.channel_last(x.permute(0, 2, 3, 1))
            return a

    results = _core(_ChannelsLastNet(), ["spatial", "channel_last"]).collect(_batches())

    assert set(results["layers"]) == {"spatial"}


def test_probe_core_seed_changes_the_sampled_pixels():
    """Different seeds sample different pixels and give different probe scores."""
    def score(seed: int) -> float:
        results = _core(_TwoLayerNet(), ["useful"], samples=32, seed=seed).collect(_batches(1))
        return results["layers"]["useful"]["scores"]["count"]["mean"]

    assert score(0) != score(3)


def test_probe_core_without_feature_layers_raises():
    """A model exposing no feature-map activation raises ValueError."""
    class _ScalarNet(torch.nn.Module):
        """Network whose only hooked layer emits a scalar rather than a feature map."""
        def __init__(self) -> None:
            """Builds the scalar identity layer."""
            super().__init__()
            self.flat = torch.nn.Identity()

        def forward(self, x):
            """Returns x unchanged while feeding its mean through the scalar layer."""
            _ = self.flat(x.mean())
            return x

    with pytest.raises(ValueError):
        _core(_ScalarNet(), ["flat"], samples=128).collect(_batches(1))
