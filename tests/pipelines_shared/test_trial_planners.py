"""Tests for the shared optimization trial planners: EMA, mixed precision and the stability ladder.

Covers the run names and configuration overrides each ladder emits, the validation errors
raised for malformed trial settings, and the round trip of the emitted override paths
through `ConfigCli` onto a `BackboneEntryConfig`.
"""

from __future__ import annotations

import pytest

from configuration.training.backbone          import BackboneEntryConfig, EmaTrialsConfig, StabilityTrialsConfig
from pipelines.shared.training.trial_planners import AmpTrialPlanner, EmaTrialPlanner, StabilityTrialPlanner
from tools.runtime.config_cli                 import ConfigCli


WARMUP_MODES = ("linear", "cosine", "exponential", "polynomial")
CLIP_MODES   = ("disabled", "fixed", "adaptive_percentile", "adaptive_mean_std")


def test_stability_planner_default_walks_the_ladder():
    """Verifies the stability ladder plans the bare, warmup-only and warmup-plus-clipping rungs."""
    planner = StabilityTrialPlanner(StabilityTrialsConfig(), WARMUP_MODES, CLIP_MODES)

    plans = dict(planner.plan())

    assert list(plans) == ["stab-0-none", "stab-1-warmup-w0.03", "stab-2-clip-n1"]
    assert planner.summary()["Total runs"] == 3

    assert plans["stab-0-none"] == {
        "training.warmup_enabled" : False,
        "training.clip_mode"      : "disabled",
    }
    assert plans["stab-1-warmup-w0.03"] == {
        "training.warmup_enabled"  : True,
        "training.warmup_fraction" : 0.03,
        "training.warmup_mode"     : "linear",
        "training.clip_mode"       : "disabled",
    }
    assert plans["stab-2-clip-n1"] == {
        "training.warmup_enabled"  : True,
        "training.warmup_fraction" : 0.03,
        "training.warmup_mode"     : "linear",
        "training.clip_mode"       : "fixed",
        "training.max_grad_norm"   : 1.0,
    }


def test_stability_planner_plans_extra_warmup_and_clipping_rungs():
    """Verifies extra fractions and norms each add a rung, clipping on the first warmup."""
    planner = StabilityTrialPlanner(StabilityTrialsConfig(warmup_fractions=[0.03, 0.1], max_grad_norms=[1.0, 0.5]), WARMUP_MODES, CLIP_MODES)

    plans = dict(planner.plan())

    assert list(plans) == ["stab-0-none", "stab-1-warmup-w0.03", "stab-1-warmup-w0.1", "stab-2-clip-n1", "stab-2-clip-n0.5"]
    assert planner.summary()["Total runs"] == 5

    assert plans["stab-1-warmup-w0.1"]["training.warmup_fraction"]  == 0.1
    assert plans["stab-2-clip-n0.5"]["training.warmup_fraction"]    == 0.03
    assert plans["stab-2-clip-n0.5"]["training.max_grad_norm"]      == 0.5


def test_stability_planner_paths_are_entry_config_leaves():
    """Verifies every stability override path resolves to a leaf of the backbone entry config."""
    leaves = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    for _, overrides in StabilityTrialPlanner(StabilityTrialsConfig(), WARMUP_MODES, CLIP_MODES).plan():
        for path in overrides:
            assert path in leaves, path


def test_stability_planner_rejects_bad_settings():
    """Verifies empty, out-of-range, duplicated or unknown warmup and clipping settings are rejected."""
    with pytest.raises(ValueError, match="at least one fraction"):
        StabilityTrialPlanner(StabilityTrialsConfig(warmup_fractions=[]), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="warmup_fractions"):
        StabilityTrialPlanner(StabilityTrialsConfig(warmup_fractions=[0.0]), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="warmup_fractions"):
        StabilityTrialPlanner(StabilityTrialsConfig(warmup_fractions=[1.5]), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="unique"):
        StabilityTrialPlanner(StabilityTrialsConfig(warmup_fractions=[0.03, 0.03]), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="warmup_mode"):
        StabilityTrialPlanner(StabilityTrialsConfig(warmup_mode="sigmoid"), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="clip_mode"):
        StabilityTrialPlanner(StabilityTrialsConfig(clip_mode="quantile"), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="active clipping strategy"):
        StabilityTrialPlanner(StabilityTrialsConfig(clip_mode="disabled"), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="at least one threshold"):
        StabilityTrialPlanner(StabilityTrialsConfig(max_grad_norms=[]), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="max_grad_norms"):
        StabilityTrialPlanner(StabilityTrialsConfig(max_grad_norms=[0.0]), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="unique"):
        StabilityTrialPlanner(StabilityTrialsConfig(max_grad_norms=[1.0, 1.0]), WARMUP_MODES, CLIP_MODES)

    with pytest.raises(ValueError, match="fixed strategy"):
        StabilityTrialPlanner(StabilityTrialsConfig(clip_mode="adaptive_percentile", max_grad_norms=[1.0, 0.5]), WARMUP_MODES, CLIP_MODES)


def test_ema_planner_default_walks_the_ladder():
    """Verifies the EMA ladder plans the off rung and one on-rung per decay."""
    planner = EmaTrialPlanner(EmaTrialsConfig(decays=[0.99, 0.999]))

    plans = dict(planner.plan())

    assert list(plans) == ["ema-0-off", "ema-1-d0.99", "ema-1-d0.999"]
    assert planner.summary()["Total runs"] == 3

    assert plans["ema-0-off"]    == {"training.use_ema": False}
    assert plans["ema-1-d0.99"]  == {"training.use_ema": True, "training.ema_decay": 0.99}
    assert plans["ema-1-d0.999"] == {"training.use_ema": True, "training.ema_decay": 0.999}


def test_ema_planner_paths_are_entry_config_leaves():
    """Verifies every EMA override path resolves to a leaf of the backbone entry config."""
    leaves = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    for _, overrides in EmaTrialPlanner(EmaTrialsConfig()).plan():
        for path in overrides:
            assert path in leaves, path


def test_ema_planner_rejects_bad_decays():
    """Verifies empty, out-of-range and duplicated decay lists are rejected."""
    with pytest.raises(ValueError, match="at least one"):
        EmaTrialPlanner(EmaTrialsConfig(decays=[]))

    with pytest.raises(ValueError, match="between 0 and 1"):
        EmaTrialPlanner(EmaTrialsConfig(decays=[1.0]))

    with pytest.raises(ValueError, match="between 0 and 1"):
        EmaTrialPlanner(EmaTrialsConfig(decays=[0.0]))

    with pytest.raises(ValueError, match="unique"):
        EmaTrialPlanner(EmaTrialsConfig(decays=[0.999, 0.999]))


def test_amp_planner_walks_the_pair():
    """Verifies the mixed-precision pair plans the off and on rungs."""
    planner = AmpTrialPlanner()

    plans = dict(planner.plan())

    assert list(plans) == ["amp-0-off", "amp-1-on"]
    assert planner.summary()["Total runs"] == 2

    assert plans["amp-0-off"] == {"training.use_amp": False}
    assert plans["amp-1-on"]  == {"training.use_amp": True}


def test_amp_planner_paths_are_entry_config_leaves():
    """Verifies every amp override path resolves to a leaf of the backbone entry config."""
    leaves = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    for _, overrides in AmpTrialPlanner().plan():
        for path in overrides:
            assert path in leaves, path


def test_stability_plan_round_trips_through_config_cli():
    """Verifies stability overrides applied through ConfigCli land on the training queue config."""
    plans = dict(StabilityTrialPlanner(StabilityTrialsConfig(), WARMUP_MODES, CLIP_MODES).plan())

    cli   = ConfigCli(BackboneEntryConfig())
    trial = cli.apply(ConfigCli.to_argv(plans["stab-2-clip-n1"]) + ["--trial"])

    assert trial.training.warmup_enabled  is True
    assert trial.training.warmup_fraction == 0.03
    assert trial.training.warmup_mode.value == "linear"
    assert trial.training.clip_mode.value   == "fixed"
    assert trial.training.max_grad_norm  == 1.0
