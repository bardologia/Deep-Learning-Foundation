
from __future__ import annotations

from configuration.training.backbone          import BackboneEntryConfig, EmaTrialsConfig, StabilityTrialsConfig
from pipelines.shared.training.trial_planners import AmpTrialPlanner, EmaTrialPlanner, StabilityTrialPlanner
from tools.runtime.config_cli                 import ConfigCli


def test_stability_planner_default_walks_the_ladder():
    planner = StabilityTrialPlanner(StabilityTrialsConfig())

    plans = dict(planner.plan())

    assert list(plans) == ["stab-0-none", "stab-1-warmup-w0.03", "stab-2-clip-n1", "stab-3-both-w0.03-n1"]
    assert planner.summary()["Total runs"] == 4

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
        "training.warmup_enabled" : False,
        "training.clip_mode"      : "fixed",
        "training.max_grad_norm"  : 1.0,
    }
    assert plans["stab-3-both-w0.03-n1"] == {
        "training.warmup_enabled"  : True,
        "training.warmup_fraction" : 0.03,
        "training.warmup_mode"     : "linear",
        "training.clip_mode"       : "fixed",
        "training.max_grad_norm"   : 1.0,
    }


def test_stability_planner_labels_a_non_fixed_clip_arm_by_strategy():
    planner = StabilityTrialPlanner(StabilityTrialsConfig(clip_mode="adaptive_percentile"))

    plans = dict(planner.plan())

    assert list(plans) == ["stab-0-none", "stab-1-warmup-w0.03", "stab-2-clip-adaptive_percentile", "stab-3-both-w0.03-adaptive_percentile"]
    assert plans["stab-2-clip-adaptive_percentile"]["training.clip_mode"]      == "adaptive_percentile"
    assert plans["stab-2-clip-adaptive_percentile"]["training.warmup_enabled"] is False


def test_stability_planner_plans_extra_warmup_and_clipping_rungs():
    planner = StabilityTrialPlanner(StabilityTrialsConfig(warmup_fractions=[0.03, 0.1], max_grad_norms=[1.0, 0.5]))

    plans = dict(planner.plan())

    assert list(plans) == ["stab-0-none", "stab-1-warmup-w0.03", "stab-1-warmup-w0.1", "stab-2-clip-n1", "stab-2-clip-n0.5", "stab-3-both-w0.03-n1", "stab-3-both-w0.03-n0.5"]
    assert planner.summary()["Total runs"] == 7

    assert plans["stab-1-warmup-w0.1"]["training.warmup_fraction"]     == 0.1
    assert plans["stab-2-clip-n0.5"]["training.warmup_enabled"]        is False
    assert plans["stab-2-clip-n0.5"]["training.max_grad_norm"]         == 0.5
    assert plans["stab-3-both-w0.03-n0.5"]["training.warmup_fraction"] == 0.03
    assert plans["stab-3-both-w0.03-n0.5"]["training.max_grad_norm"]   == 0.5


def test_stability_planner_paths_are_entry_config_leaves():
    leaves = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    for _, overrides in StabilityTrialPlanner(StabilityTrialsConfig()).plan():
        for path in overrides:
            assert path in leaves, path


def test_ema_planner_default_walks_the_ladder():
    planner = EmaTrialPlanner(EmaTrialsConfig(decays=[0.99, 0.999]))

    plans = dict(planner.plan())

    assert list(plans) == ["ema-0-off", "ema-1-raw-d0.99", "ema-1-raw-d0.999", "ema-2-warmup-d0.99", "ema-2-warmup-d0.999"]
    assert planner.summary()["Total runs"] == 5

    assert plans["ema-0-off"]           == {"training.use_ema": False}
    assert plans["ema-1-raw-d0.99"]     == {"training.use_ema": True, "training.ema_decay": 0.99, "training.ema_warmup": False}
    assert plans["ema-2-warmup-d0.999"] == {"training.use_ema": True, "training.ema_decay": 0.999, "training.ema_warmup": True}


def test_ema_planner_paths_are_entry_config_leaves():
    leaves = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    for _, overrides in EmaTrialPlanner(EmaTrialsConfig()).plan():
        for path in overrides:
            assert path in leaves, path


def test_amp_planner_walks_the_pair():
    planner = AmpTrialPlanner()

    plans = dict(planner.plan())

    assert list(plans) == ["amp-0-off", "amp-1-on"]
    assert planner.summary()["Total runs"] == 2

    assert plans["amp-0-off"] == {"training.use_amp": False}
    assert plans["amp-1-on"]  == {"training.use_amp": True}


def test_amp_planner_paths_are_entry_config_leaves():
    leaves = {path for path, _ in ConfigCli._leaves(BackboneEntryConfig())}

    for _, overrides in AmpTrialPlanner().plan():
        for path in overrides:
            assert path in leaves, path


def test_stability_plan_round_trips_through_config_cli():
    plans = dict(StabilityTrialPlanner(StabilityTrialsConfig()).plan())

    cli   = ConfigCli(BackboneEntryConfig())
    trial = cli.apply(ConfigCli.to_argv(plans["stab-3-both-w0.03-n1"]) + ["--trial"])

    assert trial.training.warmup_enabled  is True
    assert trial.training.warmup_fraction == 0.03
    assert trial.training.warmup_mode.value == "linear"
    assert trial.training.clip_mode.value   == "fixed"
    assert trial.training.max_grad_norm  == 1.0
