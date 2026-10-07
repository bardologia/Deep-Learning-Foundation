"""Tests for backbone input and output normalization, the leaky output clamp, and fitted statistics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.dataset                     import InputConfig, OutputConfig, Representation
from configuration.normalization.general       import ChannelStats, ChannelStrategy, NormMethod, NormalizationConfig, OutputClampConfig
from pipelines.backbone.dataset.datasets       import PatchDataset
from pipelines.backbone.dataset.normalizer     import Normalizer
from pipelines.backbone.dataset.spatial        import Patcher
from pipelines.backbone.dataset.stats          import Stats
from pipelines.backbone.dataset.stats_computer import StatsComputer
from tools.monitoring.logger                   import Logger


def _zscore_stats() -> Stats:
    """Returns input statistics with three plain z-score phase channels."""
    input_stats = ChannelStats(
        loc        = [1.0, -2.0, 0.0],
        scale      = [3.0, 0.5, 2.0],
        names      = ["chan/phase", "chan/phase", "derived/phase"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE)] * 3,
        clampable  = [False] * 3,
    )

    return Stats(input_stats=input_stats, output_stats=None)


def _log1p_stats() -> Stats:
    """Returns input statistics with two log1p z-score magnitude channels."""
    input_stats = ChannelStats(
        loc        = [0.5, 1.2],
        scale      = [0.8, 1.5],
        names      = ["chan/mag", "derived/mag"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE, apply_log1p=True)] * 2,
        clampable  = [False] * 2,
    )

    return Stats(input_stats=input_stats, output_stats=None)


def test_zscore_roundtrip_numpy():
    """Normalizing then denormalizing a numpy input recovers the original values."""
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(0).standard_normal((3, 5, 5)).astype(np.float32)

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, atol=1e-4)


def test_zscore_normalized_output_matches_formula():
    """Z-score normalization equals the per-channel (x - loc) / scale formula."""
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(1).standard_normal((3, 4, 4)).astype(np.float32)

    out      = norm.normalize_input(x)
    expected = (x - np.array([1.0, -2.0, 0.0]).reshape(-1, 1, 1)) / np.array([3.0, 0.5, 2.0]).reshape(-1, 1, 1)

    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_log1p_roundtrip_numpy():
    """Normalizing then denormalizing a log1p magnitude input recovers the original values."""
    norm = Normalizer(_log1p_stats())
    x    = np.abs(np.random.default_rng(2).standard_normal((2, 6, 6))).astype(np.float32) * 10.0

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, atol=1e-3)


def test_log1p_inputs_are_not_clipped_by_the_output_clamp():
    """Large log1p inputs survive the round trip untouched in both numpy and torch."""
    norm = Normalizer(_log1p_stats())
    x    = np.full((2, 3, 3), 1500.0, dtype=np.float32)

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, rtol=1e-3)

    back_torch = norm.denormalize_input(norm.normalize_input(torch.from_numpy(x)))

    np.testing.assert_allclose(back_torch.numpy(), x, rtol=1e-3)


def test_torch_and_numpy_normalization_agree():
    """The torch and numpy input-normalization paths produce the same values."""
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(3).standard_normal((3, 4, 4)).astype(np.float32)

    out_np    = norm.normalize_input(x)
    out_torch = norm.normalize_input(torch.from_numpy(x))

    np.testing.assert_allclose(out_np, out_torch.numpy(), atol=1e-5)


def test_batched_4d_normalization_roundtrip():
    """A batched (batch, channel, height, width) tensor round-trips through the normalizer."""
    norm = Normalizer(_zscore_stats())
    x    = torch.randn(2, 3, 5, 5)

    back = norm.denormalize_input(norm.normalize_input(x))

    assert torch.allclose(back, x, atol=1e-4)


def test_missing_output_stats_raises():
    """Normalizing outputs without fitted output statistics raises ValueError."""
    norm = Normalizer(_zscore_stats())

    with pytest.raises(ValueError):
        norm.normalize_output(np.zeros((1, 2, 2), dtype=np.float32))


def _mixed_output_stats(clamp: OutputClampConfig | None = None) -> Stats:
    """Returns output statistics with clampable amplitude and sigma channels and an unclamped mu channel.

    Args:
        clamp : Output clamp configuration, defaulting to the enabled clamp.

    Returns:
        Stats holding only the mixed output channel statistics.
    """
    log1p  = ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=True)
    zscore = ChannelStrategy(NormMethod.ZSCORE)

    output_stats = ChannelStats(
        loc        = [2.0, 10.0, 1.0],
        scale      = [1.5,  8.0, 0.7],
        names      = ["G1_amp", "G1_mu", "G1_sigma"],
        strategies = [zscore, zscore, log1p],
        clampable  = [True, False, True],
    )
    return Stats(input_stats=None, output_stats=output_stats, clamp=clamp if clamp is not None else OutputClampConfig())


def test_nonlog_clampable_channel_is_bounded_for_extreme_outputs():
    """An extreme normalized amplitude denormalizes to a finite, bounded physical value."""
    norm = Normalizer(_mixed_output_stats())

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 1e20

    phys = norm.denormalize_output(x, leaky_slope=0.1)

    assert torch.isfinite(phys).all()
    assert phys[:, 0].max().item() < 1e6


def test_nonlog_clampable_channel_is_identity_in_range():
    """Inside its range a clampable channel denormalizes by the plain affine formula."""
    norm = Normalizer(_mixed_output_stats())

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 40.0

    phys     = norm.denormalize_output(x, leaky_slope=0.1)
    expected = 40.0 * 1.5 + 2.0

    assert phys[:, 0].max().item() == pytest.approx(expected, rel=1e-5)


def test_mu_channel_bypasses_the_output_clamp():
    """The mu channel stays affine for both negative and very large normalized values."""
    norm = Normalizer(_mixed_output_stats())

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 1] = -5.0

    phys = norm.denormalize_output(x, leaky_slope=0.1)

    assert phys[:, 1].max().item() == pytest.approx(10.0 + 8.0 * -5.0, rel=1e-5)

    x[:, 1] = 1e6
    phys    = norm.denormalize_output(x, leaky_slope=0.1)

    assert phys[:, 1].max().item() == pytest.approx(10.0 + 8.0 * 1e6, rel=1e-5)


def test_clamp_disabled_leaves_nonlog_channels_affine():
    """With the clamp disabled an out-of-range amplitude denormalizes affinely."""
    norm = Normalizer(_mixed_output_stats(clamp=OutputClampConfig(enabled=False)))

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 1e4

    phys = norm.denormalize_output(x, leaky_slope=0.1)

    assert phys[:, 0].max().item() == pytest.approx(1e4 * 1.5 + 2.0, rel=1e-5)


def test_extreme_out_of_range_values_backpropagate_finite_gradients():
    """The leaky clamp keeps gradients finite and non-zero for far out-of-range outputs."""
    norm = Normalizer(_mixed_output_stats())

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 1e20
    x[:, 1] = 1e6
    x[:, 2] = 60.0
    x.requires_grad_(True)

    norm.denormalize_output(x, leaky_slope=0.1).sum().backward()

    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0.0).all()


def test_output_denormalization_torch_and_numpy_agree():
    """The torch and numpy output-denormalization paths produce the same values."""
    norm = Normalizer(_mixed_output_stats())

    x       = np.zeros((1, 3, 2, 2), dtype=np.float32)
    x[:, 0] = 1e10
    x[:, 1] = -3.0
    x[:, 2] = 25.0

    out_np    = norm.denormalize_output(x, leaky_slope=0.1)
    out_torch = norm.denormalize_output(torch.from_numpy(x), leaky_slope=0.1)

    np.testing.assert_allclose(out_np, out_torch.numpy(), rtol=1e-5)


@pytest.mark.dataset
@pytest.mark.slow
def test_fit_real_window_stats_finite_and_roundtrip(data_dir, derived, parameters, param_extraction_meta, tmp_path):
    """Statistics fitted on a real 24x24 window are finite, positive in scale, and round-trip."""
    derived     = np.ascontiguousarray(np.asarray(derived[:4, :24, :24]))
    primary = derived[:1]
    inputs  = np.concatenate([primary, derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :24, :24]))

    patcher = Patcher.build(spatial_size=(24, 24), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_primary=True, primary_representation=Representation.MAG_ONLY,
                          use_secondaries=False,
                          use_derived=True, derived_representation=Representation.ANGLE_ONLY)
    oc      = OutputConfig()

    ds = PatchDataset(
        inputs=inputs, gt_parameters=params, grid=patcher,
        input_config=ic, output_config=oc, split_name="train",
        n_secondaries=0, n_derived=4, n_gaussians=params.shape[0] // 3,
    )

    logger = Logger(log_dir=str(tmp_path / "logs"), name="norm_real", level="ERROR")
    stats  = StatsComputer.compute(
        dataset=ds, logger=logger, input_config=ic, output_config=oc,
        n_secondaries=0, n_derived=4, n_gaussians=params.shape[0] // 3, activity_threshold=float(param_extraction_meta["activity_threshold"]),
    )
    logger.close()

    assert np.all(np.isfinite(stats.input_stats.loc))
    assert np.all(np.isfinite(stats.input_stats.scale))
    assert np.all(np.asarray(stats.input_stats.scale) > 0.0)
    assert np.all(np.isfinite(stats.output_stats.loc))
    assert np.all(np.isfinite(stats.output_stats.scale))

    norm          = Normalizer(stats)
    ds.normalizer = norm

    out   = ds[0][0]
    back  = norm.denormalize_input(out)
    again = norm.normalize_input(back)

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(again, out, atol=1e-2)


def _fixed_bounds_output_config() -> OutputConfig:
    """Returns an output config whose amplitude, mean, and sigma use fixed-bounds normalization."""
    norm = NormalizationConfig(out_amp="fixed_bounds", out_mu="fixed_bounds", out_sigma="fixed_bounds")
    return OutputConfig(output_strategies={key: norm.strategy("output", key) for key in ("out/amp", "out/mu", "out/sigma")})


def test_fixed_bounds_output_stats_resolve_per_slot():
    """Fixed-bounds output statistics expand into per-slot loc, scale, method, and clampable flags."""
    pools = {"out/amp": np.array([1.0]), "out/mu": np.array([1.0]), "out/sigma": np.array([1.0])}
    stats = StatsComputer._fit_output(logger=None, role_pools=pools, output_config=_fixed_bounds_output_config(), n_gaussians=2)

    assert stats.loc == pytest.approx([1e-05, -20.0, 0.01, 1e-05, -20.0, 0.01])
    assert stats.scale == pytest.approx([10.0 - 1e-05, 100.0, 49.99, 10.0 - 1e-05, 100.0, 49.99])
    assert [strat.norm_method for strat in stats.strategies] == [NormMethod.FIXED_BOUNDS] * 6
    assert stats.clampable == [True, False, True, True, False, True]


def test_fixed_bounds_output_stats_reject_mismatched_gaussian_count():
    """Bounds covering two slots raise ValueError when three Gaussians are requested."""
    pools = {"out/amp": np.array([1.0]), "out/mu": np.array([1.0]), "out/sigma": np.array([1.0])}

    with pytest.raises(ValueError, match="exactly 9 entries"):
        StatsComputer._fit_output(logger=None, role_pools=pools, output_config=_fixed_bounds_output_config(), n_gaussians=3)


def test_fixed_bounds_normalization_matches_the_legacy_formula():
    """Fixed-bounds normalization is (x - min) / (max - min) and inverts exactly."""
    pools = {"out/amp": np.array([1.0]), "out/mu": np.array([1.0]), "out/sigma": np.array([1.0])}
    stats = StatsComputer._fit_output(logger=None, role_pools=pools, output_config=_fixed_bounds_output_config(), n_gaussians=2)
    norm  = Normalizer(Stats(input_stats=None, output_stats=stats))

    phys = torch.tensor([2.0, -5.0, 1.5, 4.0, 20.0, 10.0]).reshape(1, 6, 1, 1)
    out  = norm.normalize_output(phys)

    mins = torch.tensor([1e-05, -20.0, 0.01, 1e-05, -20.0, 0.01]).reshape(1, 6, 1, 1)
    maxs = torch.tensor([10.0, 80.0, 50.0, 10.0, 80.0, 50.0]).reshape(1, 6, 1, 1)
    torch.testing.assert_close(out, (phys - mins) / (maxs - mins))

    back = norm.denormalize_output(out)
    torch.testing.assert_close(back, phys, atol=1e-5, rtol=1e-5)


def test_fixed_input_presets_are_data_free_constants():
    """Fixed input presets return constants without data, and fixed bounds are rejected for inputs."""
    assert ChannelStrategy(NormMethod.FIXED_LOG1P, apply_log1p=True).fit(np.array([])) == (0.0, 1.0)

    loc, scale = ChannelStrategy(NormMethod.FIXED_ANGLE_01).fit(np.array([]))
    assert loc == pytest.approx(-np.pi)
    assert scale == pytest.approx(2.0 * np.pi)

    with pytest.raises(ValueError, match="not valid for input"):
        ChannelStrategy(NormMethod.FIXED_BOUNDS).fit(np.array([1.0]))
