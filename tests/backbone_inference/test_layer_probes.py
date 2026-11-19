
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.diagnostics                          import LayerProbeConfig
from pipelines.backbone.inference.analysis.layer_probes import FeatureGeometry, FeatureSampler, LayerProbeCore, LayerProbeRun, RidgeProbe

from tests.conftest import RecordingLogger


H, W = 16, 12


def test_ridge_probe_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    X   = rng.normal(size=(400, 8))
    y   = X @ rng.normal(size=8) + 0.01 * rng.normal(size=400)

    score = RidgeProbe(ridge_lambda=1e-3).score(X, y)

    assert score["mean"] > 0.95
    assert score["std"]  < 0.05


def test_ridge_probe_scores_noise_near_zero():
    rng = np.random.default_rng(1)
    X   = rng.normal(size=(400, 8))
    y   = rng.normal(size=400)

    assert abs(RidgeProbe().score(X, y)["mean"]) < 0.2


def test_ridge_probe_leaves_constant_folds_out():
    rng = np.random.default_rng(3)
    X   = rng.normal(size=(40, 3))
    y   = np.zeros(40)

    probe = RidgeProbe(n_folds=4, seed=0)
    folds = probe._folds(40)
    y[folds[0]] = X[folds[0], 0]

    score = probe.score(X, y)

    assert probe.degenerate_folds(y) == 3
    assert score["degenerate_folds"] == 3


def test_probe_core_leaves_a_constant_target_unscored_with_a_note():
    batches = [(images, np.concatenate([gt[:, :3], np.zeros_like(gt[:, 3:])], axis=1)) for images, gt in _batches()]

    results = _core(_TwoLayerNet(), ["useful"]).collect(batches)

    assert results["layers"]["useful"]["scores"]["second_presence"] is None
    assert results["input_baseline"]["scores"]["second_presence"] is None
    assert results["shuffled_controls"]["second_presence"] is None
    assert results["target_notes"]["second_presence"] == "zero variance in 4/4 held-out folds; R² undefined, not scored"
    assert results["layers"]["useful"]["scores"]["count"] is not None


def test_participation_ratio_separates_isotropic_from_collapsed():
    rng       = np.random.default_rng(2)
    isotropic = rng.normal(size=(2000, 8))
    collapsed = np.outer(rng.normal(size=2000), rng.normal(size=8))

    assert FeatureGeometry.participation_ratio(isotropic) > 7.0
    assert FeatureGeometry.participation_ratio(collapsed) == pytest.approx(1.0, abs=0.05)


class _TwoLayerNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.useful  = torch.nn.Identity()
        self.useless = torch.nn.Identity()

    def forward(self, x):
        a = self.useful(x)
        _ = self.useless(torch.zeros_like(x))
        return a


def _batches(n: int = 3):
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
    results = _core(_TwoLayerNet(), ["useful", "useless"]).collect(_batches())

    assert results["layers"]["useful"]["scores"]["count"]["mean"]  > 0.9
    assert results["layers"]["useless"]["scores"]["count"]["mean"] < 0.2
    assert results["layers"]["useful"]["n_channels"] == 2


def test_probe_core_scores_all_five_targets():
    results = _core(_TwoLayerNet(), ["useful"]).collect(_batches())
    scores  = results["layers"]["useful"]["scores"]

    assert set(scores) == {target for target, _label in LayerProbeCore.TARGETS}
    assert scores["second_presence"]["mean"] == pytest.approx(0.75, abs=0.05)
    assert scores["total_amp"]["mean"]       > 0.9


def test_input_baseline_decodes_what_the_input_carries():
    results  = _core(_TwoLayerNet(), ["useful"]).collect(_batches())
    baseline = results["input_baseline"]

    assert baseline["n_channels"] == 2
    assert baseline["scores"]["count"]["mean"] > 0.9


def test_shuffled_controls_sit_near_zero():
    results = _core(_TwoLayerNet(), ["useful", "useless"]).collect(_batches())

    control = results["shuffled_controls"]["count"]

    assert control["layer"] == "useful"
    assert abs(control["score"]) < 0.2


def test_feature_map_layouts_are_accepted_and_rejected():
    sampler = FeatureSampler(64)
    B       = 2

    assert sampler.is_feature_map(torch.zeros(B, 8, H, W),         B, H, W)
    assert sampler.is_feature_map(torch.zeros(B, 8, H // 4, W // 4), B, H, W)

    assert not sampler.is_feature_map(torch.zeros(6 * B, 4, 49, 49), B, H, W)
    assert not sampler.is_feature_map(torch.zeros(B, H, W, 96),      B, H, W)
    assert not sampler.is_feature_map(torch.zeros(B, 8, H),          B, H, W)


def test_probe_core_drops_channels_last_activations():
    class _ChannelsLastNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial      = torch.nn.Identity()
            self.channel_last = torch.nn.Identity()

        def forward(self, x):
            a = self.spatial(x)
            _ = self.channel_last(x.permute(0, 2, 3, 1))
            return a

    results = _core(_ChannelsLastNet(), ["spatial", "channel_last"]).collect(_batches())

    assert set(results["layers"]) == {"spatial"}
    assert results["dropped_layers"] == {"channel_last": f"output of shape (2, {H}, {W}, 2) is not a feature map on the 2x{H}x{W} input grid"}


def test_probe_core_names_a_layer_that_never_fires():
    class _IdleNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Identity()
            self.idle = torch.nn.Identity()

        def forward(self, x):
            return self.used(x)

    results = _core(_IdleNet(), ["used", "idle"]).collect(_batches(1))

    assert set(results["layers"]) == {"used"}
    assert results["dropped_layers"] == {"idle": "produced no tensor output"}


def test_layer_selection_logs_the_thinned_count(tmp_path):
    logger = RecordingLogger()
    net    = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(5)])
    run    = SimpleNamespace(model=SimpleNamespace(module=net))

    layers = LayerProbeRun(tmp_path, LayerProbeConfig(max_layers=3), logger)._select_layers(run)

    assert layers == ["0", "2", "4"]
    assert ("info", f"{tmp_path.name}: probing 3 of 5 leaf layers (max_layers=3)") in logger.messages


def test_probe_core_records_the_true_width_of_a_subsampled_layer():
    class _WideNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wide = torch.nn.Identity()

        def forward(self, x):
            return self.wide(x.repeat(1, 300, 1, 1))

    results = _core(_WideNet(), ["wide"], samples=64).collect(_batches(1))

    assert results["layers"]["wide"]["n_channels"] == 600
    assert results["layers"]["wide"]["n_channels_sampled"] == FeatureSampler.MAX_FEATURES
    assert results["input_baseline"]["n_channels"] == results["input_baseline"]["n_channels_sampled"] == 2


def test_probe_core_seed_changes_the_sampled_pixels():
    def score(seed: int) -> float:
        results = _core(_TwoLayerNet(), ["useful"], samples=32, seed=seed).collect(_batches(1))
        return results["layers"]["useful"]["scores"]["count"]["mean"]

    assert score(0) != score(3)

