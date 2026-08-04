from __future__ import annotations

from pathlib import Path

import pytest

import pipelines.dual.training.launcher as dual_launcher
from configuration.training                  import DualEntryConfig, DualRatioTrialsConfig, DualRoutingTrialsConfig, dual_curriculum
from configuration.training.backbone         import HeadMatchingTrialsConfig, ReachTrialsConfig
from configuration.training.dual             import _default_dual_ratio_trials, _default_dual_routing_trials
from configuration.training.general.ablation import AblationCatalog
from pipelines.dual.training.experiments     import DualContextTrialPlanner, DualHeadMatchingTrialPlanner, DualRatioTrialPlanner, DualReachTrialPlanner, DualRoutingTrialPlanner
from pipelines.dual.training.pipeline    import TrunkChannelMap
from tools.runtime.config_cli            import ConfigCli


PARITY_FEATURES = [48, 96, 184, 352]

TINY_LADDER = [8, 16, 24, 32]


def _planner(trials_config=None, model_overrides=None):
    return DualRoutingTrialPlanner(trials_config or DualRoutingTrialsConfig(), model_overrides or {}, TrunkChannelMap.GROUPS)


def _tiny_ratio_trials(**overrides) -> DualRatioTrialsConfig:
    defaults = {"trials": {"50-50": {"params": list(TINY_LADDER), "existence": list(TINY_LADDER)}}, "match_tolerance": 0.1}
    return DualRatioTrialsConfig(**{**defaults, **overrides})


def _ratio_planner(trials_config=None, model_overrides=None):
    return DualRatioTrialPlanner(
        trials_config or _tiny_ratio_trials(),
        model_overrides or {},
        "dual_resunet",
        ("resunet", "resunet"),
        ((0, 1, 2, 3, 4, 5, 6, 7, 8), (5, 6, 7, 8)),
        9,
    )


def test_dual_routing_planner_default_catalog_covers_the_seven_ordered_pairs():
    plans = dict(_planner().plan())

    assert list(plans) == ["di-full-full", "di-pass-full", "di-full-pass", "di-ifg-full", "di-full-ifg", "di-pass-ifg", "di-ifg-pass"]

    full   = ["pass", "ifg"]
    passes = ["pass"]
    ifg    = ["ifg"]

    assert plans["di-full-full"]["params_input"]    == full
    assert plans["di-full-full"]["existence_input"] == full
    assert plans["di-pass-full"]["params_input"]    == passes
    assert plans["di-pass-full"]["existence_input"] == full
    assert plans["di-full-pass"]["params_input"]    == full
    assert plans["di-full-pass"]["existence_input"] == passes
    assert plans["di-ifg-full"]["params_input"]     == ifg
    assert plans["di-ifg-full"]["existence_input"]  == full
    assert plans["di-full-ifg"]["params_input"]     == full
    assert plans["di-full-ifg"]["existence_input"]  == ifg
    assert plans["di-pass-ifg"]["params_input"]     == passes
    assert plans["di-pass-ifg"]["existence_input"]  == ifg
    assert plans["di-ifg-pass"]["params_input"]     == ifg
    assert plans["di-ifg-pass"]["existence_input"]  == passes


def test_dual_routing_planner_default_catalog_never_selects_the_dem():
    for _, overrides in _planner().plan():
        assert "dem" not in overrides["params_input"]
        assert "dem" not in overrides["existence_input"]


def test_dual_routing_planner_fixes_half_width_trunks_on_every_trial():
    for _, overrides in _planner().plan():
        assert overrides["model_overrides"] == {"params_features": PARITY_FEATURES, "existence_features": PARITY_FEATURES}


def test_dual_routing_planner_merges_entry_model_overrides():
    for _, overrides in _planner(model_overrides={"head_activation": "gelu"}).plan():
        assert overrides["model_overrides"]["head_activation"] == "gelu"
        assert overrides["model_overrides"]["params_features"] == PARITY_FEATURES


def test_dual_routing_planner_rejects_feature_keys_in_model_overrides():
    with pytest.raises(ValueError, match="routing_trials fields"):
        _planner(model_overrides={"params_features": [8, 16]})


def test_dual_routing_planner_rejects_empty_or_invalid_features():
    with pytest.raises(ValueError, match="at least one feature width"):
        _planner(DualRoutingTrialsConfig(params_features=[]))

    with pytest.raises(ValueError, match="positive integers"):
        _planner(DualRoutingTrialsConfig(existence_features=[32, 0]))


def test_dual_routing_planner_rejects_empty_trials():
    with pytest.raises(ValueError, match="at least one trunk-input variant"):
        _planner(DualRoutingTrialsConfig(trials={}))


def test_dual_routing_planner_rejects_missing_trunk_key():
    with pytest.raises(ValueError, match="must set \\['existence'\\]"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["pass"]}}))


def test_dual_routing_planner_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys \\['gate'\\]"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["pass"], "existence": ["ifg"], "gate": ["ifg"]}}))


def test_dual_routing_planner_rejects_unknown_groups():
    with pytest.raises(ValueError, match="unknown 'params' groups \\['phase'\\]"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["pass", "phase"], "existence": ["ifg"]}}))


def test_dual_routing_planner_rejects_the_dem_group():
    with pytest.raises(ValueError, match="unknown 'params' groups \\['dem'\\]"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["pass", "ifg", "dem"], "existence": ["ifg"]}}))


def test_dual_routing_planner_rejects_empty_selection():
    with pytest.raises(ValueError, match="selects no channel groups for 'existence'"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["pass"], "existence": []}}))


def test_dual_routing_planner_rejects_duplicate_groups():
    with pytest.raises(ValueError, match="repeats 'params' groups"):
        _planner(DualRoutingTrialsConfig(trials={"bad": {"params": ["pass", "pass"], "existence": ["ifg"]}}))


def test_dual_routing_planner_paths_are_entry_config_leaves():
    config = DualEntryConfig()

    for _, overrides in _planner().plan():
        for path in overrides:
            assert hasattr(config, path), path


def test_dual_trial_overrides_round_trip_through_the_cli():
    _, overrides = _planner().plan()[4]

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(overrides))

    assert config.params_input    == ["pass", "ifg"]
    assert config.existence_input == ["ifg"]
    assert config.model_overrides == {"params_features": PARITY_FEATURES, "existence_features": PARITY_FEATURES}


def test_dual_default_trials_match_the_config_factory():
    assert DualRoutingTrialsConfig().trials == _default_dual_routing_trials()


def test_dual_ratio_planner_emits_ladder_overrides_and_leaves_inputs_alone():
    plans = _ratio_planner().plan()

    assert [name for name, _ in plans] == ["dr-50-50"]
    assert plans[0][1] == {"model_overrides": {"params_features": TINY_LADDER, "existence_features": TINY_LADDER}}


def test_dual_ratio_planner_merges_entry_model_overrides():
    _, overrides = _ratio_planner(model_overrides={"head_activation": "gelu"}).plan()[0]

    assert overrides["model_overrides"]["head_activation"] == "gelu"
    assert overrides["model_overrides"]["params_features"] == TINY_LADDER


def test_dual_ratio_planner_rejects_feature_keys_in_model_overrides():
    with pytest.raises(ValueError, match="ratio_trials ladders"):
        _ratio_planner(model_overrides={"existence_features": [8, 16]})


def test_dual_ratio_planner_rejects_empty_trials_and_bad_tolerance():
    with pytest.raises(ValueError, match="at least one arm-split variant"):
        _ratio_planner(_tiny_ratio_trials(trials={}))

    with pytest.raises(ValueError, match="match_tolerance"):
        _ratio_planner(_tiny_ratio_trials(match_tolerance=0.0))


def test_dual_ratio_planner_rejects_missing_or_unknown_arm_keys():
    with pytest.raises(ValueError, match="must set \\['existence'\\]"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER}}))

    with pytest.raises(ValueError, match="unknown keys \\['gate'\\]"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER, "existence": TINY_LADDER, "gate": TINY_LADDER}}))


def test_dual_ratio_planner_rejects_empty_or_invalid_ladders():
    with pytest.raises(ValueError, match="lists no feature widths"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER, "existence": []}}))

    with pytest.raises(ValueError, match="positive integers"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": [8, 0, 24, 32], "existence": TINY_LADDER}}))


def test_dual_ratio_planner_rejects_mixed_ladder_depths():
    with pytest.raises(ValueError, match="mix ladder depths"):
        _ratio_planner(_tiny_ratio_trials(trials={"50-50": {"params": TINY_LADDER, "existence": [8, 16, 24]}}))


def test_dual_ratio_planner_rejects_malformed_share_labels():
    with pytest.raises(ValueError, match="percentage shares"):
        _ratio_planner(_tiny_ratio_trials(trials={"half": {"params": TINY_LADDER, "existence": TINY_LADDER}}))

    with pytest.raises(ValueError, match="sum to 100"):
        _ratio_planner(_tiny_ratio_trials(trials={"60-41": {"params": TINY_LADDER, "existence": TINY_LADDER}}))

    with pytest.raises(ValueError, match="detection arm is always the smaller"):
        _ratio_planner(_tiny_ratio_trials(trials={"40-60": {"params": TINY_LADDER, "existence": TINY_LADDER}}))


def test_dual_ratio_planner_rejects_a_split_the_ladders_do_not_realize():
    with pytest.raises(ValueError, match="match tolerance; retune the ladders"):
        _ratio_planner(_tiny_ratio_trials(trials={"90-10": {"params": list(TINY_LADDER), "existence": list(TINY_LADDER)}}, match_tolerance=0.01))


def test_dual_ratio_planner_rejects_trials_with_unequal_budgets():
    trials = {
        "50-50" : {"params": list(TINY_LADDER), "existence": list(TINY_LADDER)},
        "60-40" : {"params": [48, 96, 144, 192], "existence": [40, 80, 120, 160]},
    }

    with pytest.raises(ValueError, match="total parameter count"):
        _ratio_planner(_tiny_ratio_trials(trials=trials, match_tolerance=0.2))


def test_dual_ratio_overrides_round_trip_through_the_cli():
    _, overrides = _ratio_planner().plan()[0]

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(overrides))

    assert config.model_overrides == {"params_features": TINY_LADDER, "existence_features": TINY_LADDER}
    assert config.params_input    == ["pass", "ifg"]
    assert config.existence_input == ["pass", "ifg"]


def test_dual_ratio_default_catalog_holds_the_parity_budget_at_every_split():
    planner = _ratio_planner(DualRatioTrialsConfig())

    assert DualRatioTrialsConfig().trials == _default_dual_ratio_trials()
    assert [name for name, _ in planner.plan()] == ["dr-50-50", "dr-60-40", "dr-70-30", "dr-80-20", "dr-90-10"]
    assert planner.arm_counts["50-50"]["params"] + planner.arm_counts["50-50"]["existence"] == 31_189_858

    for spec in _default_dual_ratio_trials().values():
        assert len(spec["params"])    == 4
        assert len(spec["existence"]) == 4


def test_dual_single_runner_builds_the_model_config_with_trunk_fields():
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
    config                 = DualEntryConfig()
    config.model_overrides = {"params_overrides": {"width": 8}}

    with pytest.raises(ValueError, match="dedicated entry fields"):
        dual_launcher.DualSingleTrainRunner(config)._model_config()


def test_dual_scheduler_houses_runs_in_routing_dir(tmp_path):
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "routing"

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    assert scheduler.runs_root == tmp_path / "routing"
    assert scheduler.log_dir   == tmp_path / "routing" / "batch_train_logs"

    job = scheduler._job("di-full-ifg", {})

    assert job.command[-2:] == ["--logdir", str(tmp_path / "routing")]
    assert job.log_path     == tmp_path / "routing" / "batch_train_logs" / "di-full-ifg.log"


def test_dual_scheduler_plans_the_trunk_input_grid(tmp_path):
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "routing"

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["di-full-full", "di-pass-full", "di-full-pass", "di-ifg-full", "di-full-ifg", "di-pass-ifg", "di-ifg-pass"]
    assert plans["di-full-ifg"]["model_overrides"]["params_features"] == PARITY_FEATURES


def test_dual_scheduler_houses_ratio_runs_in_ratio_dir_and_plans_the_split_grid(tmp_path):
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ratio"

    scheduler = dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))

    assert scheduler.runs_root == tmp_path / "ratio"

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["dr-50-50", "dr-60-40", "dr-70-30", "dr-80-20", "dr-90-10"]
    assert plans["dr-50-50"]["model_overrides"]["params_features"]    == PARITY_FEATURES
    assert plans["dr-50-50"]["model_overrides"]["existence_features"] == PARITY_FEATURES
    assert plans["dr-90-10"]["model_overrides"]["params_features"]    == [64, 128, 248, 472]
    assert plans["dr-90-10"]["model_overrides"]["existence_features"] == [24, 48, 84, 156]


def test_dual_scheduler_rejects_unknown_mode(tmp_path):
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "bogus"

    with pytest.raises(ValueError, match="Unknown trials_mode 'bogus'"):
        dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))


def test_dual_scheduler_forwards_only_non_scheduler_overrides(tmp_path):
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
    ran = {}

    class FakeSingleRunner:
        def __init__(self, cfg):
            ran["single"] = cfg
        def run(self):
            ran["ran"] = True

    class FakeScheduler:
        def __init__(self, **kwargs):
            ran["scheduler"] = True
        def run(self):
            ran["scheduler_ran"] = True

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
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
    ran = {}

    class FakeSingleRunner:
        def __init__(self, cfg):
            ran["single"] = True
        def run(self):
            ran["single_ran"] = True

    class FakeFanout:
        @classmethod
        def for_runner(cls, config, cli_overrides, entry_script, runner_factory, base_label=None, infer_at_end=False):
            ran["for_runner"] = (entry_script, runner_factory)
            ran["infer_at_end"] = infer_at_end
            return cls()
        def run(self):
            ran["fanout_ran"] = True

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
            return DualEntryConfig()

    monkeypatch.setattr(dual_launcher, "DualSingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(dual_launcher, "SeedFanoutScheduler", FakeFanout)
    monkeypatch.setattr(dual_launcher, "ConfigCli", FakeCli)

    entry = Path("/entry/train_dual.py")
    dual_launcher.DualTrainingLauncher(entry_script=entry).run([])

    assert ran.get("fanout_ran") is True
    assert ran["for_runner"]     == (entry, FakeSingleRunner)
    assert "single_ran" not in ran


def test_dual_launcher_fans_out_when_trials_enabled(monkeypatch):
    ran = {}

    class FakeSingleRunner:
        def __init__(self, cfg):
            ran["single"] = True
        def run(self):
            ran["single_ran"] = True

    class FakeScheduler:
        def __init__(self, **kwargs):
            ran["entry_script"] = kwargs["entry_script"]
        def run(self):
            ran["scheduler_ran"] = True

    class FakeConfig:
        trials_enabled = True
        curriculum     = dual_curriculum()
        infer_after    = False
        infer_at_end   = False

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
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
    config             = DualEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = mode

    return dual_launcher.DualTrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_dual.py"))


def test_dual_scheduler_opens_every_backbone_mode():
    modes = set(dual_launcher.DualTrainScheduler.MODE_SUBDIRS)

    assert set(dual_launcher.TrainScheduler.MODE_SUBDIRS) <= modes
    assert {"routing", "ratio"} <= modes


def test_dual_scheduler_reuses_backbone_planners_for_shared_modes(tmp_path):
    scheduler = _scheduler(tmp_path, "curriculum")

    assert scheduler.runs_root == tmp_path / "curriculum"

    plans = dict(scheduler.planner().plan())

    assert "w-pL11_c-pL11-cos0.05" in plans
    assert plans["w-pL11_c-pL11-cos0.05"]["curriculum.enabled"] is True


def test_dual_context_planner_applies_the_rung_to_both_trunks():
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
    scheduler          = _scheduler(tmp_path, "context")
    run_name, override = scheduler.planner().plan()[1]

    config = ConfigCli(DualEntryConfig()).apply(ConfigCli.to_argv(override))

    assert run_name                  == "ctx-cnn09"
    assert config.params_backbone    == "local_cnn"
    assert config.existence_backbone == "local_cnn"
    assert config.params_overrides   == {"block_kernels": [3] * 2 + [1] * 8}
    assert config.model_overrides["params_features"] == config.model_overrides["existence_features"]


def test_dual_head_planner_holds_set_pred_and_sweeps_matchings():
    trials  = HeadMatchingTrialsConfig(backbone="unet_skip", heads=["set_pred"], matchings=["sorted_gt", "hungarian"])
    planner = DualHeadMatchingTrialPlanner(trials, ("unet_skip",), ("set_pred",), ("sorted_gt", "hungarian"))

    plans = dict(planner.plan())

    assert list(plans) == ["hm-set_pred-sorted_gt", "hm-set_pred-hungarian"]
    assert plans["hm-set_pred-hungarian"] == {
        "params_backbone"                    : "unet_skip",
        "existence_backbone"                 : "unet_skip",
        "curriculum.warmup.param_matching"   : "hungarian",
        "curriculum.complete.param_matching" : "hungarian",
    }


def test_dual_head_planner_rejects_heads_the_dual_model_lacks():
    trials = HeadMatchingTrialsConfig(backbone="unet_skip", heads=["conv", "set_pred"], matchings=["hungarian"])

    with pytest.raises(ValueError, match="registered heads"):
        DualHeadMatchingTrialPlanner(trials, ("unet_skip",), ("set_pred",), ("hungarian",))


def test_dual_reach_planner_counts_whole_dual_models_and_emits_trunk_settings():
    rungs  = [
        {"label": "cnn",  "backbone": "local_cnn", "features": [64] * 2},
        {"label": "unet", "backbone": "unet",      "features": [8, 16]},
    ]
    trials = ReachTrialsConfig(rungs=rungs, match_tolerance=10.0)

    planner = DualReachTrialPlanner(trials, ("local_cnn", "unet"), 9, 6, "dual_resunet", (tuple(range(9)), tuple(range(9))))

    assert set(planner.parameter_counts) == {"cnn", "unet"}
    assert all(count > 0 for count in planner.parameter_counts.values())

    plans     = dict(planner.plan())
    overrides = plans["reach-cnn"]

    assert overrides["params_backbone"]    == "local_cnn"
    assert overrides["existence_backbone"] == "local_cnn"
    assert overrides["model_overrides"]    == {"params_features": [64] * 2, "existence_features": [64] * 2}
    assert overrides["training.patch_size"] == (32, 32)


def test_dual_reach_default_rungs_hold_the_capacity_match():
    config  = DualEntryConfig()
    planner = DualReachTrialPlanner(config.reach_trials, ("local_cnn", "unet"), 9, 6, "dual_resunet", (tuple(range(9)), tuple(range(9))))

    assert set(planner.parameter_counts) == {"cnn33", "unet"}


def test_dual_ablation_catalog_swaps_the_backbone_only_rungs():
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
    config = DualEntryConfig()

    for feature in config.ablation_features:
        for overrides in (feature["enable"], feature["degrade"]):
            for path in overrides:
                node = config
                for part in path.split(".")[:-1]:
                    node = getattr(node, part)
                assert hasattr(node, path.split(".")[-1]), path
