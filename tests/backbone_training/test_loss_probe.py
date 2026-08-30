"""Tests for the loss-scale probe used to calibrate per-term loss weights.

Covers how the probe forces terms on, honours per-term overrides from the entry config,
reports suggested weights relative to a reference term, and filters outlier ratios.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training                 import BackboneEntryConfig, LossScaleProbeConfig
from configuration.training.general.loss    import LossConfig
from pipelines.backbone.training.launcher   import SingleTrainRunner
from pipelines.backbone.training.loss_probe import LossScaleProbe
from tools.monitoring.logger                import NullLogger

from tests.backbone_training._helpers import gaussian_config, identity_normalizer, tiny_model, x_axis_tensor


def _loader(in_channels: int = 2, n_gaussians: int = 2, n: int = 4, hw: int = 12) -> DataLoader:
    """Returns a loader of random (n, in_channels, hw, hw) inputs and (n, 3*n_gaussians, hw, hw) targets."""
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(n, in_channels, hw, hw, generator=gen)
    tgt  = torch.randn(n, n_gaussians * 3, hw, hw, generator=gen)
    return DataLoader(TensorDataset(imgs, tgt), batch_size=2)


def _probe(probe_cfg: LossScaleProbeConfig, n_channels: int = 6) -> LossScaleProbe:
    """Returns a loss-scale probe wired to an identity normalizer over n_channels parameter channels."""
    return LossScaleProbe(
        probe_cfg    = probe_cfg,
        loss_cfg     = LossConfig(),
        gaussian_cfg = gaussian_config(n_channels // 3),
        norm_stats   = identity_normalizer(n_channels),
        logger       = NullLogger(),
    )


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


def test_probe_leaves_legacy_term_off_by_default():
    """Verifies the legacy parameter term stays off unless explicitly requested."""
    probe = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=False))

    assert probe.loss_cfg.use_param_legacy is False


def test_probe_enables_legacy_term_only_on_explicit_override():
    """Verifies an explicit override switches the legacy parameter term on."""
    overrides = {"use_param_legacy": True}
    probe     = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=False, enabled_losses=overrides))

    assert probe.loss_cfg.use_param_legacy is True


def test_probe_runs_and_reports_suggested_weights():
    """Verifies the probe returns suggested weights with the reference term pinned at one."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    loader   = _loader()
    probe    = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, reference="param_l1", exit_after=False))

    suggested = probe.run(loader, model, torch.device("cpu"), x_axis_tensor())

    assert isinstance(suggested, dict)
    assert "param_l1"  in suggested
    assert "mse_curve" in suggested
    assert suggested["param_l1"] == pytest.approx(1.0, rel=1e-6)


def test_probe_disabled_returns_empty():
    """Verifies a disabled probe returns no suggestions."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    loader   = _loader()
    probe    = _probe(LossScaleProbeConfig(enabled=False, exit_after=False))

    assert probe.run(loader, model, torch.device("cpu"), x_axis_tensor()) == {}


def test_probe_suggested_weights_finite_for_positive_terms():
    """Verifies the suggested weights for positive-valued terms are finite."""
    model, _ = tiny_model(in_channels=2, n_gaussians=2)
    loader   = _loader()
    probe    = _probe(LossScaleProbeConfig(enabled=True, n_batches=2, exit_after=False))

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
