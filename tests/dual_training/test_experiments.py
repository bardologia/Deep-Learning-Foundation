"""Tests for the dual training trial planners, scheduler and launcher.

Covers the routing catalogue of trunk-input pairs, the capacity-matched ratio
ladders and their validation rules, the context, head/matching, reach and
ablation planners, the run directory and override forwarding of the dual
scheduler, and the launcher's choice between a single run, a seed fan-out and
a trial scheduler.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipelines.dual.training.launcher as dual_launcher
from configuration.synthesis                 import SynthesisConfig
from configuration.training                  import DualEntryConfig, DualRatioTrialsConfig, DualRoutingTrialsConfig, dual_curriculum
from configuration.training.backbone         import HeadMatchingTrialsConfig, ReachTrialsConfig
from configuration.training.dual             import _default_dual_ratio_trials, _default_dual_routing_trials
from configuration.training.general.ablation import AblationCatalog
from pipelines.dual.training.experiments     import DualContextTrialPlanner, DualHeadMatchingTrialPlanner, DualRatioTrialPlanner, DualReachTrialPlanner, DualRoutingTrialPlanner
from pipelines.dual.training.pipeline    import TrunkChannelMap
from tools.runtime.config_cli            import ConfigCli


PARITY_FEATURES = [48, 96, 184, 352]

TINY_LADDER = [8, 16, 24, 32]

RATIO_OUT_CHANNELS = 15


def _planner(trials_config=None, model_overrides=None):
    """Returns a routing trial planner over the given trials config and model overrides."""
    return DualRoutingTrialPlanner(trials_config or DualRoutingTrialsConfig(), model_overrides or {}, TrunkChannelMap.GROUPS)


def _tiny_ratio_trials(**overrides) -> DualRatioTrialsConfig:
    """Returns a ratio trials config with a single 50-50 split over a four-rung ladder."""
    defaults = {"trials": {"50-50": {"params": list(TINY_LADDER), "existence": list(TINY_LADDER)}}, "match_tolerance": 0.1}
    return DualRatioTrialsConfig(**{**defaults, **overrides})


def _ratio_planner(trials_config=None, model_overrides=None, out_channels=RATIO_OUT_CHANNELS, trunk_overrides=({}, {})):
    """Returns a ratio trial planner for a nine-channel dual_resunet at a 32x32 patch with the given output width and entry-level trunk overrides."""
    return DualRatioTrialPlanner(
        trials_config or _tiny_ratio_trials(),
        model_overrides or {},
        "dual_resunet",
        ("resunet", "resunet"),
        ((0, 1, 2, 3, 4, 5, 6, 7, 8), (5, 6, 7, 8)),
        trunk_overrides,
        (32, 32),
        9,
        out_channels,
    )


def test_dual_routing_planner_default_catalog_covers_the_pairs_with_and_without_dem():
    """The default routing catalogue doubles every ordered trunk-input pair as a no-aux field and a with-aux field variant."""
    plans = dict(_planner().plan())

    pairs = ["full-full", "chan-full", "full-chan", "derived-full", "full-derived", "chan-derived", "derived-chan"]
    assert list(plans) == [f"di-{pair}-{aux}" for pair in pairs for aux in ("noaux", "aux")]

    full    = ["chan", "derived"]
    chan    = ["chan"]
    derived = ["derived"]

    assert plans["di-full-full-noaux"]["params_input"]    == full
    assert plans["di-full-full-noaux"]["existence_input"] == full
    assert plans["di-full-full-aux"]["params_input"]      == full + ["aux"]
    assert plans["di-full-full-aux"]["existence_input"]   == full + ["aux"]
    assert plans["di-chan-full-noaux"]["params_input"]    == chan
    assert plans["di-chan-full-noaux"]["existence_input"] == full
    assert plans["di-chan-full-aux"]["params_input"]      == chan + ["aux"]
    assert plans["di-chan-full-aux"]["existence_input"]   == full + ["aux"]
    assert plans["di-derived-chan-noaux"]["params_input"]     == derived
    assert plans["di-derived-chan-noaux"]["existence_input"]  == chan
    assert plans["di-derived-chan-aux"]["params_input"]       == derived + ["aux"]
    assert plans["di-derived-chan-aux"]["existence_input"]    == chan + ["aux"]


def test_dual_routing_planner_switches_the_dem_channel_with_the_selection():
    """`input.use_aux` is on exactly when a trunk selects the aux group, aux variants routing it into both trunks."""
    for run_name, overrides in _planner().plan():
        selected = "aux" in overrides["params_input"] or "aux" in overrides["existence_input"]

        assert overrides["input.use_aux"] is selected
        assert selected == run_name.endswith("-aux")

        if selected:
            assert "aux" in overrides["params_input"] and "aux" in overrides["existence_input"]


def test_dual_routing_planner_fixes_half_width_trunks_on_every_trial():
    """Every routing trial pins both trunks to the parity feature ladder."""
    for _, overrides in _planner().plan():
        assert overrides["model_overrides"] == {"params_features": PARITY_FEATURES, "existence_features": PARITY_FEATURES}


def test_dual_routing_planner_merges_entry_model_overrides():
    """Entry-level model overrides survive alongside the planner's feature ladders."""
    for _, overrides in _planner(model_overrides={"head_activation": "gelu"}).plan():
        assert overrides["model_overrides"]["head_activation"] == "gelu"
        assert overrides["model_overrides"]["params_features"] == PARITY_FEATURES


def test_dual_routing_planner_rejects_feature_keys_in_model_overrides():
    """Setting trunk feature widths through model_overrides raises ValueError."""
    with pytest.raises(ValueError, match="routing_trials fields"):
        _planner(model_overrides={"params_features": [8, 16]})


def test_dual_routing_planner_rejects_empty_or_invalid_features():
    """Empty or non-positive feature ladders raise ValueError."""
    with pytest.raises(ValueError, match="at least one feature width"):
        _planner(DualRoutingTrialsConfig(params_features=[]))

    with pytest.raises(ValueError, match="positive integers"):
        _planner(DualRoutingTrialsConfig(existence_features=[32, 0]))


def test_dual_routing_planner_rejects_empty_trials():
    """A routing catalogue without trials raises ValueError."""
    with pytest.raises(ValueError, match="at least one trunk-input variant"):
        _planner(DualRoutingTrialsConfig(trials={}))


def test_dual_routing_planner_rejects_missing_trunk_key():
    """A trial missing one trunk's input list raises ValueError."""
    with pytest.raises(ValueError, match="must set \\['existence'\\]"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["chan"]}}))


def test_dual_routing_planner_rejects_unknown_keys():
    """A trial carrying a key beyond the two trunks raises ValueError."""
    with pytest.raises(ValueError, match="unknown keys \\['gate'\\]"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["chan"], "existence": ["derived"], "gate": ["derived"]}}))


def test_dual_routing_planner_rejects_unknown_groups():
    """A trial naming an unregistered input group raises ValueError."""
    with pytest.raises(ValueError, match="unknown 'params' groups \\['phase'\\]"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["chan", "phase"], "existence": ["derived"]}}))


def test_dual_routing_planner_accepts_the_dem_group():
    """A trial routing the Aux into a trunk validates against the registered groups."""
    plans = dict(_planner(DualRoutingTrialsConfig(trials={"aux-derived": {"params": ["chan", "aux"], "existence": ["derived"]}})).plan())

    assert plans["di-aux-derived"]["params_input"] == ["chan", "aux"]


def test_dual_routing_planner_rejects_empty_selection():
    """A trunk with an empty group list raises ValueError."""
    with pytest.raises(ValueError, match="selects no channel groups for 'existence'"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["chan"], "existence": []}}))


def test_dual_routing_planner_rejects_duplicate_groups():
    """A trunk repeating an input group raises ValueError."""
    with pytest.raises(ValueError, match="repeats 'params' groups"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["chan", "chan"], "existence": ["derived"]}}))


def test_dual_routing_planner_paths_are_entry_config_leaves():
    """Every override path emitted by the routing planner exists on the dual entry config."""
    config = DualEntryConfig()

    for _, overrides in _planner().plan():
        for path in overrides:
            section, _, leaf = path.partition(".")
            target = getattr(config, section)
            assert not leaf or hasattr(target, leaf), path


def test_dual_trial_overrides_round_trip_through_the_cli():
    """Routing overrides survive conversion to argv and back through the config CLI."""
    plans = dict(_planner().plan())

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(plans["di-full-derived-aux"]))

    assert config.params_input    == ["chan", "derived", "aux"]
    assert config.existence_input == ["derived", "aux"]
    assert config.input.use_aux   is True
    assert config.model_overrides == {"params_features": PARITY_FEATURES, "existence_features": PARITY_FEATURES}

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(plans["di-full-derived-noaux"]))

    assert config.params_input    == ["chan", "derived"]
    assert config.existence_input == ["derived"]
    assert config.input.use_aux   is False


def test_dual_default_trials_match_the_config_factory():
    """The routing config's default trials come from the shared factory."""
    assert DualRoutingTrialsConfig().trials == _default_dual_routing_trials()


def test_dual_ratio_planner_emits_ladder_overrides_and_leaves_inputs_alone():
    """A ratio trial emits only the two feature ladders, without touching trunk inputs."""
    plans = _ratio_planner().plan()

    assert [name for name, _ in plans] == ["dr-50-50"]
    assert plans[0][1] == {"model_overrides": {"params_features": TINY_LADDER, "existence_features": TINY_LADDER}}


def test_dual_ratio_planner_merges_entry_model_overrides():
    """Entry-level model overrides survive alongside the ratio ladders."""
    _, overrides = _ratio_planner(model_overrides={"head_activation": "gelu"}).plan()[0]

    assert overrides["model_overrides"]["head_activation"] == "gelu"
    assert overrides["model_overrides"]["params_features"] == TINY_LADDER


def test_dual_ratio_planner_rejects_feature_keys_in_model_overrides():
    """Setting trunk feature widths through model_overrides raises ValueError."""
    with pytest.raises(ValueError, match="ratio_trials ladders"):
        _ratio_planner(model_overrides={"existence_features": [8, 16]})


def test_dual_ratio_planner_rejects_empty_trials_and_bad_tolerance():
    """An empty split catalogue or a non-positive match tolerance raises ValueError."""
    with pytest.raises(ValueError, match="at least one arm-split variant"):
        _ratio_planner(_tiny_ratio_trials(trials={}))

    with pytest.raises(ValueError, match="match_tolerance"):
        _ratio_planner(_tiny_ratio_trials(match_tolerance=0.0))


def test_dual_ratio_planner_rejects_missing_or_unknown_arm_keys():
    """A split missing an arm ladder or carrying an extra key raises ValueError."""
    with pytest.raises(ValueError, match="must set \\['existence'\\]"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER}}))

    with pytest.raises(ValueError, match="unknown keys \\['gate'\\]"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER, "existence": TINY_LADDER, "gate": TINY_LADDER}}))


def test_dual_ratio_planner_rejects_empty_or_invalid_ladders():
    """An empty ladder or a non-positive feature width raises ValueError."""
    with pytest.raises(ValueError, match="lists no feature widths"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER, "existence": []}}))

    with pytest.raises(ValueError, match="positive integers"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": [8, 0, 24, 32], "existence": TINY_LADDER}}))


def test_dual_ratio_planner_rejects_mixed_ladder_depths():
    """Arm ladders of unequal depth raise ValueError."""
    with pytest.raises(ValueError, match="mix ladder depths"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER, "existence": [8, 16, 24]}}))


def test_dual_ratio_planner_rejects_malformed_share_labels():
    """Split labels that are not two percentages summing to 100 with the detection arm smaller raise ValueError."""
    with pytest.raises(ValueError, match="percentage shares"):
        _ratio_planner(_tiny_ratio_trials(trials={"half": {"params": TINY_LADDER, "existence": TINY_LADDER}}))

    with pytest.raises(ValueError, match="sum to 100"):
        _ratio_planner(_tiny_ratio_trials(trials={"60-41": {"params": TINY_LADDER, "existence": TINY_LADDER}}))

    with pytest.raises(ValueError, match="detection arm is always the smaller"):
        _ratio_planner(_tiny_ratio_trials(trials={"40-60": {"params": TINY_LADDER, "existence": TINY_LADDER}}))


def test_dual_ratio_planner_rejects_a_split_the_ladders_do_not_realize():
    """Ladders whose realised parameter split misses the labelled share raise ValueError."""
    with pytest.raises(ValueError, match="match tolerance; retune the ladders"):
        _ratio_planner(_tiny_ratio_trials(trials={"90-10": {"params": list(TINY_LADDER), "existence": list(TINY_LADDER)}}, match_tolerance=0.01))


def test_dual_ratio_planner_rejects_trials_with_unequal_budgets():
    """Splits whose total parameter counts differ raise ValueError."""
    trials = {
        "50-50" : {"params": list(TINY_LADDER), "existence": list(TINY_LADDER)},
        "60-40" : {"params": [48, 96, 144, 192], "existence": [40, 80, 120, 160]},
    }

    with pytest.raises(ValueError, match="total parameter count"):
        _ratio_planner(_tiny_ratio_trials(trials=trials, match_tolerance=0.2))


def test_dual_ratio_overrides_round_trip_through_the_cli():
    """Ratio overrides survive conversion to argv and back, leaving the default trunk inputs."""
    _, overrides = _ratio_planner().plan()[0]

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(overrides))

    assert config.model_overrides == {"params_features": TINY_LADDER, "existence_features": TINY_LADDER}
    assert config.params_input    == ["chan", "derived", "aux"]
    assert config.existence_input == ["chan", "derived", "aux"]


def test_dual_ratio_default_catalog_holds_the_parity_budget_at_every_split():
    """The five default splits share one four-rung ladder depth and the same total parameter budget."""
    planner = _ratio_planner(DualRatioTrialsConfig())

    assert DualRatioTrialsConfig().trials == _default_dual_ratio_trials()
    assert [name for name, _ in planner.plan()] == ["dr-50-50", "dr-60-40", "dr-70-30", "dr-80-20", "dr-90-10"]
    assert planner.arm_counts["50-50"]["params"] + planner.arm_counts["50-50"]["existence"] == 31_193_689

    for spec in _default_dual_ratio_trials().values():
        assert len(spec["params"])    == 4
        assert len(spec["existence"]) == 4


def test_dual_ratio_planner_counts_the_heads_at_the_resolved_output_width():
    """A wider resolved output width raises the counted parameters of both arms."""
    two_slots  = _ratio_planner(out_channels=6)
    five_slots = _ratio_planner(out_channels=RATIO_OUT_CHANNELS)

    assert five_slots.arm_counts["50-50"]["params"]    > two_slots.arm_counts["50-50"]["params"]
    assert five_slots.arm_counts["50-50"]["existence"] > two_slots.arm_counts["50-50"]["existence"]


def test_dual_ratio_planner_rejects_an_unresolved_output_width():
    """A zero output width raises ValueError."""
    with pytest.raises(ValueError, match="resolved output width"):
        _ratio_planner(out_channels=0)


def test_dual_ratio_planner_counts_the_entry_trunk_overrides():
    """The arm budgets are counted on trunks built with the entry-level trunk overrides, as training builds them, and the summary names them."""
    plain  = _ratio_planner()
    shaped = _ratio_planner(trunk_overrides=({"bottleneck_factor": 4}, {"bottleneck_factor": 4}))

    assert shaped.arm_counts["50-50"]["params"]    > plain.arm_counts["50-50"]["params"]
    assert shaped.arm_counts["50-50"]["existence"] > plain.arm_counts["50-50"]["existence"]
    assert shaped.summary()["Trunk overrides"] == "params {'bottleneck_factor': 4}, existence {'bottleneck_factor': 4}"
    assert plain.summary()["Trunk overrides"]  == "params none, existence none"
    assert plain.summary()["Counted at"]       == "15 output channels, patch (32, 32)"


def test_dual_single_runner_builds_the_model_config_with_trunk_fields():
    """The single runner copies trunk backbone and per-trunk override fields into the model config."""
    config                     = DualEntryConfig()
    config.params_backbone     = "nafnet"
    config.params_overrides    = {"width": 8}
    config.existence_overrides = {"dropout": 0.1}
    config.model_overrides     = {"params_features": [], "head_activation": "gelu"}

    model_config = dual_launcher.DualSingleTrainRunner(config)._model_config()

    assert model_config.params_backbone     == "nafnet"
    assert model_config.params_overrides    == {"width": 8}
    assert model_config.existence_overrides == {"dropout": 0.1}
    assert model_config.params_features     == []
    assert model_config.head_activation     == "gelu"


def test_dual_single_runner_rejects_trunk_fields_in_model_overrides():
    """Passing a dedicated trunk field through model_overrides raises ValueError."""
    config                 = DualEntryConfig()
    config.model_overrides = {"params_overrides": {"width": 8}}

    with pytest.raises(ValueError, match="dedicated entry fields"):
        dual_launcher.DualSingleTrainRunner(config)._model_config()


def test_dual_scheduler_houses_runs_in_routing_dir(tmp_path):
    """Routing mode places runs and per-trial logs under the routing subdirectory."""
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "routing"

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    assert scheduler.runs_root == tmp_path / "routing"
    assert scheduler.log_dir   == tmp_path / "routing" / "batch_train_logs"

    job = scheduler._job("di-full-derived", {})

    assert job.command[-2:] == ["--logdir", str(tmp_path / "routing")]
    assert job.log_path     == tmp_path / "routing" / "batch_train_logs" / "di-full-derived.log"


def test_dual_scheduler_plans_the_trunk_input_grid(tmp_path):
    """The scheduler's routing planner emits the trunk-input trials, aux field-doubled, with parity ladders."""
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "routing"

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    plans = dict(scheduler.planner().plan())

    pairs = ["full-full", "chan-full", "full-chan", "derived-full", "full-derived", "chan-derived", "derived-chan"]
    assert list(plans) == [f"di-{pair}-{aux}" for pair in pairs for aux in ("noaux", "aux")]
    assert plans["di-full-derived-aux"]["model_overrides"]["params_features"] == PARITY_FEATURES


def test_dual_scheduler_houses_ratio_runs_in_ratio_dir_and_plans_the_split_grid(tmp_path, test_data_dir, params_dir):
    """Ratio mode places runs under the ratio subdirectory and plans the five splits at the dataset's resolved output width."""
    config                       = DualEntryConfig()
    config.logdir                = tmp_path
    config.trials_mode           = "ratio"
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    assert scheduler.runs_root == tmp_path / "ratio"

    planner = scheduler.planner()
    plans   = dict(planner.plan())

    assert planner.out_channels == 3 * SynthesisConfig().k_max
    assert list(plans) == ["dr-50-50", "dr-60-40", "dr-70-30", "dr-80-20", "dr-90-10"]
    assert plans["dr-50-50"]["model_overrides"]["params_features"]    == PARITY_FEATURES
    assert plans["dr-50-50"]["model_overrides"]["existence_features"] == PARITY_FEATURES
    assert plans["dr-90-10"]["model_overrides"]["params_features"]    == [64, 128, 248, 472]
    assert plans["dr-90-10"]["model_overrides"]["existence_features"] == [24, 48, 84, 156]


def test_dual_scheduler_inherits_the_exhaustive_secondary_sweep(tmp_path):
    """Secondary mode enumerates every two-channel selection from the secondary labels the dataset records, not the configured selection."""
    data = tmp_path / "dataset" / "data"
    data.mkdir(parents=True)
    (data / "dataset.json").write_text(json.dumps({"channel_labels": ["CH00", "CH01", "CH02", "CH03"]}))

    config                                = DualEntryConfig()
    config.logdir                         = tmp_path / "runs"
    config.trials_mode                    = "secondary"
    config.secondary_trials.strategy      = "exhaustive"
    config.secondary_trials.n_secondaries = 2
    config.paths.dataset_path             = tmp_path / "dataset"
    config.paths.secondary_labels         = ("CH01",)

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))
    plans     = scheduler.planner().plan()

    selections = [overrides["paths.secondary_labels"] for _, overrides in plans]

    assert scheduler.runs_root == tmp_path / "runs" / "secondary"
    assert len(plans) == 3
    assert selections[0]  == ("CH01", "CH02")
    assert selections[-1] == ("CH02", "CH03")


def test_dual_scheduler_rejects_unknown_mode(tmp_path):
    """An unregistered trials mode raises ValueError."""
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "bogus"

    with pytest.raises(ValueError, match="Unknown trials_mode 'bogus'"):
        dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))


def test_dual_scheduler_forwards_only_non_scheduler_overrides(tmp_path):
    """Scheduler-owned CLI overrides are consumed and only the training ones are forwarded to workers."""
    config        = DualEntryConfig()
    config.logdir = tmp_path

    overrides = {
        "trials_enabled"               : True,
        "trials_mode"                  : "routing",
        "routing_trials.params_features" : [16, 32],
        "ratio_trials.match_tolerance" : 0.02,
        "gpus"                         : [0, 1],
        "poll_interval_s"              : 1.0,
        "training.max_epochs"          : 5,
        "params_backbone"              : "unet",
    }

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides=overrides, entry_script=Path("/entry/train_dual.py"))

    assert scheduler.forward_overrides == {"training.max_epochs": 5, "params_backbone": "unet"}


def test_dual_launcher_trial_flag_runs_single_runner(monkeypatch):
    """The trial flag runs a single training run instead of the trial scheduler."""
    ran = {}

    class FakeSingleRunner:
        """Single-run runner stub recording the config it received."""
        def __init__(self, cfg):
            """Records the config handed to the runner."""
            ran["single"] = cfg
        def run(self):
            """Records that the single run executed."""
            ran["ran"] = True

    class FakeScheduler:
        """Trial scheduler stub recording that it was constructed."""
        def __init__(self, **kwargs):
            """Records that the scheduler was constructed."""
            ran["scheduler"] = True
        def run(self):
            """Records that the scheduler executed."""
            ran["scheduler_ran"] = True

    class FakeCli:
        """Config CLI stub returning a single-seed dual entry config."""
        def __init__(self, config, description):
            """Starts with no recorded CLI overrides."""
            self.overrides = {}
        def apply(self, argv):
            """Returns a dual entry config pinned to a single seed."""
            config       = DualEntryConfig()
            config.seeds = [0]
            return config

    monkeypatch.setattr(dual_launcher, "DualSingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(dual_launcher, "DualTrainScheduler", FakeScheduler)
    monkeypatch.setattr(dual_launcher, "ConfigCli", FakeCli)

    dual_launcher.DualTrainingLauncher(entry_script=Path("/entry/train_dual.py")).run(["--trial"])

    assert ran.get("ran") is True
    assert "scheduler" not in ran


def test_dual_launcher_fans_multi_seed_runs_across_the_pool(monkeypatch):
    """A multi-seed config is handed to the seed fan-out scheduler rather than run directly."""
    ran = {}

    class FakeSingleRunner:
        """Single-run runner stub recording whether it was constructed or run."""
        def __init__(self, cfg):
            """Records that the runner was constructed."""
            ran["single"] = True
        def run(self):
            """Records that the single run executed."""
            ran["single_ran"] = True

    class FakeFanout:
        """Seed fan-out scheduler stub recording the entry script, runner factory and inference flags."""
        @classmethod
        def for_runner(cls, config, cli_overrides, entry_script, runner_factory, base_label=None, infer_at_end=False, infer_after=False):
            """Records the fan-out arguments and returns a stub instance."""
            ran["for_runner"]   = (entry_script, runner_factory)
            ran["infer_at_end"] = infer_at_end
            ran["infer_after"]  = infer_after
            return cls()
        def run(self):
            """Records that the fan-out executed."""
            ran["fanout_ran"] = True

    class FakeCli:
        """Config CLI stub returning a default dual entry config."""
        def __init__(self, config, description):
            """Starts with no recorded CLI overrides."""
            self.overrides = {}
        def apply(self, argv):
            """Returns a default dual entry config."""
            return DualEntryConfig()

    monkeypatch.setattr(dual_launcher, "DualSingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(dual_launcher, "SeedFanoutScheduler", FakeFanout)
    monkeypatch.setattr(dual_launcher, "ConfigCli", FakeCli)

    entry = Path("/entry/train_dual.py")
    dual_launcher.DualTrainingLauncher(entry_script=entry).run([])

    assert ran.get("fanout_ran") is True
    assert ran["for_runner"]     == (entry, FakeSingleRunner)
    assert ran["infer_after"]    == DualEntryConfig().infer_after
    assert "single_ran" not in ran


def test_dual_launcher_fans_out_when_trials_enabled(monkeypatch):
    """A config with trials enabled is handed to the trial scheduler with the entry script."""
    ran = {}

    class FakeSingleRunner:
        """Single-run runner stub recording whether it was constructed or run."""
        def __init__(self, cfg):
            """Records that the runner was constructed."""
            ran["single"] = True
        def run(self):
            """Records that the single run executed."""
            ran["single_ran"] = True

    class FakeScheduler:
        """Trial scheduler stub recording the entry script it was given."""
        def __init__(self, **kwargs):
            """Records the entry script passed to the scheduler."""
            ran["entry_script"] = kwargs["entry_script"]
        def run(self):
            """Records that the scheduler executed."""
            ran["scheduler_ran"] = True

    class FakeConfig:
        """Minimal config exposing trials as enabled with a dual curriculum, no trailing inference and no resume."""
        trials_enabled = True
        curriculum     = dual_curriculum()
        infer_after    = False
        infer_at_end   = False
        resume_from    = ""

    class FakeCli:
        """Config CLI stub returning the trials-enabled fake config."""
        def __init__(self, config, description):
            """Starts with no recorded CLI overrides."""
            self.overrides = {}
        def apply(self, argv):
            """Returns the trials-enabled fake config."""
            return FakeConfig()

    monkeypatch.setattr(dual_launcher, "DualSingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(dual_launcher, "DualTrainScheduler", FakeScheduler)
    monkeypatch.setattr(dual_launcher, "ConfigCli", FakeCli)

    entry = Path("/entry/train_dual.py")
    dual_launcher.DualTrainingLauncher(entry_script=entry).run([])

    assert ran.get("scheduler_ran") is True
    assert ran["entry_script"] == entry
    assert "single_ran" not in ran


def _scheduler(tmp_path, mode):
    """Returns a dual train scheduler in the given trials mode logging under the temporary path."""
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = mode

    return dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))


def test_dual_scheduler_opens_every_backbone_mode():
    """The dual scheduler supports every backbone trials mode plus routing and ratio."""
    modes = set(dual_launcher.DualTrainScheduler.MODE_SUBDIRS)

    assert set(dual_launcher.TrainScheduler.MODE_SUBDIRS) <= modes
    assert {"routing", "ratio"} <= modes


def test_dual_scheduler_reuses_backbone_planners_for_shared_modes(tmp_path):
    """Curriculum mode reuses the backbone planner and its run directory layout."""
    scheduler = _scheduler(tmp_path, "curriculum")

    assert scheduler.runs_root == tmp_path / "curriculum"

    plans = dict(scheduler.planner().plan())

    assert "w-pL11_c-pL11-cos0.05" in plans
    assert plans["w-pL11_c-pL11-cos0.05"]["curriculum.enabled"] is True


def test_dual_context_planner_applies_the_rung_to_both_trunks():
    """A context rung sets the same backbone, feature ladder and block kernels on both trunks."""
    rung    = {"label": "cnn09", "backbone": "local_cnn", "features": [8] * 10, "block_kernels": [3] * 2 + [1] * 8}
    planner = DualContextTrialPlanner([rung], ("local_cnn",))

    plans     = dict(planner.plan())
    overrides = plans["ctx-cnn09"]

    assert overrides["params_backbone"]     == "local_cnn"
    assert overrides["existence_backbone"]  == "local_cnn"
    assert overrides["model_overrides"]     == {"params_features": [8] * 10, "existence_features": [8] * 10}
    assert overrides["params_overrides"]    == {"block_kernels": [3] * 2 + [1] * 8}
    assert overrides["existence_overrides"] == {"block_kernels": [3] * 2 + [1] * 8}


def test_dual_context_overrides_round_trip_through_the_cli(tmp_path):
    """Context overrides survive conversion to argv and back with matching trunk ladders."""
    scheduler          = _scheduler(tmp_path, "context")
    run_name, override = scheduler.planner().plan()[2]

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(override))

    assert run_name                  == "ctx-cnn09"
    assert config.params_backbone    == "local_cnn"
    assert config.existence_backbone == "local_cnn"
    assert config.params_overrides   == {"block_kernels": [3] * 2 + [1] * 8}
    assert config.model_overrides["params_features"] == config.model_overrides["existence_features"]


def test_dual_head_planner_holds_set_pred_and_sweeps_matchings():
    """The head planner keeps the set-prediction head fixed and varies the matching in both curriculum stages."""
    trials  = HeadMatchingTrialsConfig(heads=["set_pred"], matchings=["sorted_gt", "hungarian"])
    planner = DualHeadMatchingTrialPlanner(trials, "unet_skip", ("unet_skip",), ("set_pred",), ("sorted_gt", "hungarian"))

    plans = dict(planner.plan())

    assert list(plans) == ["hm-set_pred-sorted_gt", "hm-set_pred-hungarian"]
    assert plans["hm-set_pred-hungarian"] == {
        "params_backbone"                    : "unet_skip",
        "existence_backbone"                 : "unet_skip",
        "curriculum.warmup.param_matching"   : "hungarian",
        "curriculum.complete.param_matching" : "hungarian",
    }


def test_dual_head_planner_rejects_heads_the_dual_model_lacks():
    """Requesting a head the dual model does not register raises ValueError."""
    trials = HeadMatchingTrialsConfig(heads=["conv", "set_pred"], matchings=["hungarian"])

    with pytest.raises(ValueError, match="registered heads"):
        DualHeadMatchingTrialPlanner(trials, "unet_skip", ("unet_skip",), ("set_pred",), ("hungarian",))


def test_dual_reach_planner_counts_whole_dual_models_and_emits_trunk_settings():
    """Reach rungs carry positive whole-model parameter counts and set both trunks plus the patch size."""
    rungs  = [
        {"label": "cnn",  "backbone": "local_cnn", "features": [64] * 2},
        {"label": "unet", "backbone": "unet",      "features": [8, 16]},
    ]
    trials = ReachTrialsConfig(rungs=rungs, match_tolerance=10.0)

    planner = DualReachTrialPlanner(trials, ("local_cnn", "unet"), 9, 6, "dual_resunet", (tuple(range(9)), tuple(range(9))), ({}, {}))

    assert set(planner.parameter_counts) == {"cnn", "unet"}
    assert all(count > 0 for count in planner.parameter_counts.values())

    plans     = dict(planner.plan())
    overrides = plans["reach-cnn"]

    assert overrides["params_backbone"]    == "local_cnn"
    assert overrides["existence_backbone"] == "local_cnn"
    assert overrides["model_overrides"]    == {"params_features": [64] * 2, "existence_features": [64] * 2}
    assert overrides["training.patch_size"] == (32, 32)


def test_dual_reach_default_rungs_hold_the_capacity_match():
    """The default reach rungs pass the capacity match and keep their labels."""
    config  = DualEntryConfig()
    planner = DualReachTrialPlanner(config.reach_trials, ("local_cnn", "unet"), 9, 6, "dual_resunet", (tuple(range(9)), tuple(range(9))), ({}, {}))

    assert set(planner.parameter_counts) == {"cnn33", "unet"}


def test_dual_reach_planner_counts_entry_trunk_overrides_unless_the_rung_sets_its_own():
    """A rung without overrides is counted with the entry-level trunk overrides its unit trains with; a rung with its own is counted with those alone, as its settings replace the entry-level dicts."""
    rungs = [
        {"label": "plain",  "backbone": "unet", "features": [8, 16]},
        {"label": "shaped", "backbone": "unet", "features": [8, 16], "bottleneck_factor": 4},
    ]
    trials   = ReachTrialsConfig(rungs=rungs, match_tolerance=10.0)
    channels = (tuple(range(9)), tuple(range(9)))

    entry_plain  = DualReachTrialPlanner(trials, ("unet",), 9, 6, "dual_resunet", channels, ({}, {}))
    entry_shaped = DualReachTrialPlanner(trials, ("unet",), 9, 6, "dual_resunet", channels, ({"bottleneck_factor": 4}, {"bottleneck_factor": 4}))

    assert entry_shaped.parameter_counts["plain"]  >  entry_plain.parameter_counts["plain"]
    assert entry_shaped.parameter_counts["plain"]  == entry_plain.parameter_counts["shaped"]
    assert entry_shaped.parameter_counts["shaped"] == entry_plain.parameter_counts["shaped"]
    assert entry_shaped.summary()["Trunk overrides"].startswith("params {'bottleneck_factor': 4}, existence {'bottleneck_factor': 4}")
    assert entry_plain.summary()["Trunk overrides"].startswith("params none, existence none")


def test_dual_ablation_catalog_swaps_the_backbone_only_rungs():
    """The dual ablation catalogue replaces the backbone rungs with trunk backbones and per-group learning rates while preserving feature order."""
    features = {feature["label"]: feature for feature in AblationCatalog.dual_default_features()}

    assert features["architecture_param_loss"]["enable"]["params_backbone"]     == "unet_skip"
    assert features["architecture_param_loss"]["enable"]["existence_backbone"]  == "unet_skip"
    assert features["architecture_param_loss"]["degrade"]["params_backbone"]    == "unet"
    assert features["architecture_param_loss"]["degrade"]["existence_backbone"] == "unet"
    assert list(features["lr_per_group"]["enable"]["model_overrides"]) == ["params_trunk_lr", "existence_trunk_lr", "output_head_lr"]

    dual_order    = [feature["label"] for feature in AblationCatalog.dual_default_features()]
    default_order = [feature["label"] for feature in AblationCatalog.default_features()]

    assert dual_order == default_order


def test_dual_ablation_override_paths_are_dual_entry_leaves():
    """Every ablation override path resolves to a leaf attribute of the dual entry config."""
    config = DualEntryConfig()

    for feature in config.ablation_features:
        for overrides in (feature["enable"], feature["degrade"]):
            for path in overrides:
                node = config
                for part in path.split(".")[:-1]:
                    node = getattr(node, part)
                assert hasattr(node, path.split(".")[-1]), path
