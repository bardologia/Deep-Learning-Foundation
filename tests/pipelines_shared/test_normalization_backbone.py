
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
    input_stats = ChannelStats(
        loc        = [1.0, -2.0, 0.0],
        scale      = [3.0, 0.5, 2.0],
        names      = ["chan/phase", "chan/phase", "derived/phase"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE)] * 3,
        clampable  = [False] * 3,
    )

    return Stats(input_stats=input_stats, output_stats=None)


def _log1p_stats() -> Stats:
    input_stats = ChannelStats(
        loc        = [0.5, 1.2],
        scale      = [0.8, 1.5],
        names      = ["chan/mag", "derived/mag"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE, apply_log1p=True)] * 2,
        clampable  = [False] * 2,
    )

    return Stats(input_stats=input_stats, output_stats=None)


def test_zscore_roundtrip_numpy():
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(0).standard_normal((3, 5, 5)).astype(np.float32)

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, atol=1e-4)


def test_zscore_normalized_output_matches_formula():
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(1).standard_normal((3, 4, 4)).astype(np.float32)

    out      = norm.normalize_input(x)
    expected = (x - np.array([1.0, -2.0, 0.0]).reshape(-1, 1, 1)) / np.array([3.0, 0.5, 2.0]).reshape(-1, 1, 1)

    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_log1p_roundtrip_numpy():
    norm = Normalizer(_log1p_stats())
    x    = np.abs(np.random.default_rng(2).standard_normal((2, 6, 6))).astype(np.float32) * 10.0

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, atol=1e-3)


def test_log1p_inputs_are_not_clipped_by_the_output_clamp():
    norm = Normalizer(_log1p_stats())
    x    = np.full((2, 3, 3), 1500.0, dtype=np.float32)

    back = norm.denormalize_input(norm.normalize_input(x))

    np.testing.assert_allclose(back, x, rtol=1e-3)

    back_torch = norm.denormalize_input(norm.normalize_input(torch.from_numpy(x)))

    np.testing.assert_allclose(back_torch.numpy(), x, rtol=1e-3)


def test_torch_and_numpy_normalization_agree():
    norm = Normalizer(_zscore_stats())
    x    = np.random.default_rng(3).standard_normal((3, 4, 4)).astype(np.float32)

    out_np    = norm.normalize_input(x)
    out_torch = norm.normalize_input(torch.from_numpy(x))

    np.testing.assert_allclose(out_np, out_torch.numpy(), atol=1e-5)


def test_batched_4d_normalization_roundtrip():
    norm = Normalizer(_zscore_stats())
    x    = torch.randn(2, 3, 5, 5)

    back = norm.denormalize_input(norm.normalize_input(x))

    assert torch.allclose(back, x, atol=1e-4)


def _mixed_output_stats(clamp: OutputClampConfig | None = None) -> Stats:
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
    norm = Normalizer(_mixed_output_stats())

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 1e20

    phys = norm.denormalize_output(x, leaky_slope=0.1)

    assert torch.isfinite(phys).all()
    assert phys[:, 0].max().item() < 1e6


def test_nonlog_clampable_channel_is_identity_in_range():
    norm = Normalizer(_mixed_output_stats())

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 40.0

    phys     = norm.denormalize_output(x, leaky_slope=0.1)
    expected = 40.0 * 1.5 + 2.0

    assert phys[:, 0].max().item() == pytest.approx(expected, rel=1e-5)


def test_mu_channel_bypasses_the_output_clamp():
    norm = Normalizer(_mixed_output_stats())

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 1] = -5.0

    phys = norm.denormalize_output(x, leaky_slope=0.1)

    assert phys[:, 1].max().item() == pytest.approx(10.0 + 8.0 * -5.0, rel=1e-5)

    x[:, 1] = 1e6
    phys    = norm.denormalize_output(x, leaky_slope=0.1)

    assert phys[:, 1].max().item() == pytest.approx(10.0 + 8.0 * 1e6, rel=1e-5)


def test_clamp_disabled_leaves_nonlog_channels_affine():
    norm = Normalizer(_mixed_output_stats(clamp=OutputClampConfig(enabled=False)))

    x       = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 1e4

    phys = norm.denormalize_output(x, leaky_slope=0.1)

    assert phys[:, 0].max().item() == pytest.approx(1e4 * 1.5 + 2.0, rel=1e-5)


def test_extreme_out_of_range_values_backpropagate_finite_gradients():
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
    norm = Normalizer(_mixed_output_stats())

    x       = np.zeros((1, 3, 2, 2), dtype=np.float32)
    x[:, 0] = 1e10
    x[:, 1] = -3.0
    x[:, 2] = 25.0

    out_np    = norm.denormalize_output(x, leaky_slope=0.1)
    out_torch = norm.denormalize_output(torch.from_numpy(x), leaky_slope=0.1)

    np.testing.assert_allclose(out_np, out_torch.numpy(), rtol=1e-5)


def _window_dataset(derived, parameters) -> tuple[PatchDataset, InputConfig, OutputConfig]:
    window  = np.ascontiguousarray(np.asarray(derived[:4, :24, :24]))
    primary = window[:1]
    inputs  = np.concatenate([primary, window], axis=0)
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

    return ds, ic, oc


@pytest.mark.dataset
@pytest.mark.slow
def test_fit_real_window_stats_finite_and_roundtrip(data_dir, derived, parameters, param_extraction_meta, tmp_path):
    ds, ic, oc = _window_dataset(derived, parameters)

    logger = Logger(log_dir=str(tmp_path / "logs"), name="norm_real", level="ERROR")
    stats  = StatsComputer.compute(
        dataset=ds, logger=logger, input_config=ic, output_config=oc,
        n_secondaries=0, n_derived=4, n_gaussians=ds.n_gaussians, activity_threshold=float(param_extraction_meta["activity_threshold"]),
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


@pytest.mark.dataset
@pytest.mark.slow
def test_compute_logs_the_effective_clamp_and_the_pool_sizes(data_dir, derived, parameters, param_extraction_meta, tmp_path):
    ds, ic, oc = _window_dataset(derived, parameters)

    logger = Logger(log_dir=str(tmp_path / "logs"), name="norm_clamp", level="ERROR")
    StatsComputer.compute(
        dataset=ds, logger=logger, input_config=ic, output_config=oc,
        n_secondaries=0, n_derived=4, n_gaussians=ds.n_gaussians, activity_threshold=float(param_extraction_meta["activity_threshold"]),
        normalization=NormalizationConfig(clamp_output=False),
    )
    logger.close()

    text = (tmp_path / "logs" / "norm_clamp.log").read_text(encoding="utf-8")

    assert "Output clamp : disabled" in text
    assert f"Active pixels per slot (train split, 1 region(s), amplitude > {float(param_extraction_meta['activity_threshold']):g})" in text
    assert "of 576 pixels active" in text


def test_output_stats_table_reports_sample_counts_and_clampable_roles(tmp_path):
    logger = Logger(log_dir=str(tmp_path / "logs"), name="norm_table", level="ERROR")
    pools  = {"out/amp": np.array([1.0, 2.0, 3.0]), "out/mu": np.array([1.0, 2.0, 3.0]), "out/sigma": np.array([1.0, 2.0, 3.0])}

    StatsComputer._fit_output(logger=logger, role_pools=pools, output_config=OutputConfig(), n_gaussians=1)
    logger.close()

    text = (tmp_path / "logs" / "norm_table.log").read_text(encoding="utf-8")

    assert "Samples" in text
    assert "clampable" in text
    assert "clamped" not in text.replace("clampable", "")


def test_clamp_rows_describe_the_effective_clamp():
    assert Stats.clamp_rows(OutputClampConfig(enabled=False))["Output clamp"] == "disabled"
    assert Stats.clamp_rows(OutputClampConfig(floor=0.0, ceil=50.0, amp_max=20.0)) == {"Output clamp": "0.0 to 50.0", "Clamp leak": 0.1, "Param leak": 0.1, "Amp max": 20.0}
