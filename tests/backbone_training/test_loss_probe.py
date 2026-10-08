"""Tests for the loss-scale probe used to calibrate per-term loss weights.

Covers how the probe forces terms on, honours per-term overrides from the entry config,
reports suggested weights relative to a reference term, filters outlier ratios, logs the
batches it measured, states that exit_after trains nothing, and is refused in fan-outs
that would mark every probe-only unit DONE.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training                 import BackboneEntryConfig, LossScaleProbeConfig
from configuration.training.general.loss    import LossConfig
from pipelines.backbone.training.launcher   import BackboneTrainingLauncher, SingleTrainRunner
from pipelines.backbone.training.loss_probe import LossScaleProbe
from tools.monitoring.logger                import Logger, NullLogger

from tests.backbone_training._helpers import gaussian_config, identity_normalizer, tiny_model, x_axis_tensor


def _loader(in_channels: int = 2, n_gaussians: int = 2, n: int = 4, hw: int = 12) -> DataLoader:
    """Returns a loader of random (n, in_channels, hw, hw) inputs and (n, 3*n_gaussians, hw, hw) targets."""
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(n, in_channels, hw, hw, generator=gen)
    tgt  = torch.randn(n, n_gaussians * 3, hw, hw, generator=gen)
    return DataLoader(TensorDataset(imgs, tgt), batch_size=2)


def _probe(probe_cfg: LossScaleProbeConfig, n_channels: int = 6, logger=None) -> LossScaleProbe:
    """Returns a loss-scale probe wired to an identity normalizer over n_channels parameter channels.

    Args:
        probe_cfg  : Probe configuration under test.
        n_channels : Number of output parameter channels.
        logger     : Logger receiving the report; a null logger when omitted, which suits the configuration-only tests.
    """
    return LossScaleProbe(
        probe_cfg    = probe_cfg,
        loss_cfg     = LossConfig(),
        gaussian_cfg = gaussian_config(n_channels // 3),
        norm_stats   = identity_normalizer(n_channels),
        logger       = logger if logger is not None else NullLogger(),
    )


@pytest.fixture
def probe_logger(tmp_path):
    """Yields a file logger under tmp_path for the tests that run the probe, and closes it afterwards."""
    logger = Logger(log_dir=str(tmp_path / "logs"), name="loss_probe_test")
    yield logger
    logger.close()


def _log_text(tmp_path) -> str:
    """Returns the contents of the probe test log file."""
    return (tmp_path / "logs" / "loss_probe_test.log").read_text(encoding="utf-8")


def test_probe_forces_all_terms_on_with_unit_weight():
    """Verifies the probe enables every sweepable loss term at unit weight."""
    probe = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=False))

    assert probe.loss_cfg.use_mse_curve    is True
    assert probe.loss_cfg.weight_mse_curve == 1.0
    assert probe.loss_cfg.use_moments      is True


def test_entry_config_forwards_the_per_term_probe_overrides():
    """Verifies the entry config's probe block reaches the runner's resolved probe config intact."""
    config                      = BackboneEntryConfig()
    config.probe.enabled        = True
    config.probe.enabled_losses = {"use_moments": False}

    probe_cfg = SingleTrainRunner(config).config.probe

    assert probe_cfg.enabled        is True
    assert probe_cfg.enabled_losses == {"use_moments": False}
    assert probe_cfg.reference      == "param_l1"


def test_probe_respects_enabled_losses_override():
    """Verifies a per-term override switches that term off while leaving the rest enabled."""
    overrides = {"use_moments": False}
    probe     = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=False, enabled_losses=overrides))

    assert probe.loss_cfg.use_moments is False
    assert probe.loss_cfg.use_mse_curve        is True






def test_probe_runs_and_reports_suggested_weights(probe_logger):
    """Verifies the probe returns suggested weights with the reference term pinned at one."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    loader   = _loader()
    probe    = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, reference="param_l1", exit_after=False), logger=probe_logger)

    suggested = probe.run(loader, model, torch.device("cpu"), x_axis_tensor())

    assert isinstance(suggested, dict)
    assert "param_l1"  in suggested
    assert "mse_curve" in suggested
    assert suggested["param_l1"] == pytest.approx(1.0, rel=1e-6)


def test_probe_logs_the_measured_batch_count_and_duration(tmp_path, probe_logger):
    """Verifies the probe announces the effective batch budget, not the configured one, and closes with the measured count and elapsed time."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    loader   = _loader()
    probe    = _probe(LossScaleProbeConfig(enabled=True, n_batches=1000, exit_after=False), logger=probe_logger)

    probe.run(loader, model, torch.device("cpu"), x_axis_tensor())
    probe_logger.close()

    text = _log_text(tmp_path)

    assert "Averaging over 2 batches (1000 configured, 2 in the train loader)" in text
    assert "Probe measured 2 batches in" in text


def test_probe_exit_after_warns_that_nothing_trained(tmp_path, probe_logger):
    """Verifies exit_after still exits cleanly but first states that no training ran and no run artifacts were written."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    probe    = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=True), logger=probe_logger)

    with pytest.raises(SystemExit) as stop:
        probe.run(_loader(), model, torch.device("cpu"), x_axis_tensor())
    probe_logger.close()

    assert stop.value.code == 0
    assert "no training ran and no checkpoint, test metrics or complete.json were written" in _log_text(tmp_path)


def test_launcher_refuses_a_probe_only_trial_sweep(tmp_path):
    """Verifies a trial sweep with the probe enabled and exit_after left on is refused before any unit is planned."""
    launcher = BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py"))

    with pytest.raises(ValueError, match="probe.exit_after=True would end every unit of the patch trial sweep"):
        launcher.run(["--trials_enabled", "true", "--trials_mode", "patch", "--probe.enabled", "true", "--logdir", str(tmp_path)])


def test_launcher_refuses_a_probe_only_seed_fanout(tmp_path):
    """Verifies a multi-seed fan-out with the probe enabled and exit_after left on is refused before any seed job is queued."""
    launcher = BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py"))

    with pytest.raises(ValueError, match="2-seed fan-out"):
        launcher.run(["--seeds", "[0, 1]", "--probe.enabled", "true", "--logdir", str(tmp_path)])


def test_probe_disabled_returns_empty():
    """Verifies a disabled probe returns no suggestions."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    loader   = _loader()
    probe    = _probe(LossScaleProbeConfig(enabled=False, exit_after=False))

    assert probe.run(loader, model, torch.device("cpu"), x_axis_tensor()) == {}


def test_probe_suggested_weights_finite_for_positive_terms(probe_logger):
    """Verifies the suggested weights for positive-valued terms are finite."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    loader   = _loader()
    probe    = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=False), logger=probe_logger)

    suggested = probe.run(loader, model, torch.device("cpu"), x_axis_tensor())

    assert math.isfinite(suggested["param_l1"])
    assert math.isfinite(suggested["mse_curve"])


def test_probe_iqr_filter_drops_outlier():
    """Verifies the IQR filter removes a far-outlier ratio."""
    values   = [1.0, 1.1, 0.9, 1.05, 100.0]
    filtered = LossScaleProbe._iqr_filter(values)

    assert 100.0 not in filtered
    assert len(filtered) == 4


def test_probe_iqr_filter_keeps_short_lists():
    """Verifies the IQR filter passes lists too short to estimate quartiles through unchanged."""
    values = [1.0, 2.0, 3.0]

    assert LossScaleProbe._iqr_filter(values) == values
