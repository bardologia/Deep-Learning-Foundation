"""Tests for the training launchers and the backbone trial scheduler.

Covers seed-sweep launching and GPU fan-out, the trials planners for each mode,
resume from an existing sweep folder, and the batched inference pass that runs
after all trials have trained.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pipelines.backbone.training.launcher as backbone_pipeline
from configuration.training               import BackboneEntryConfig, default_curriculum
from pipelines.shared.training            import training_launcher as mod
from pipelines.shared.training.seed_sweep import SeedFanoutScheduler
from tools.runtime.completion             import CompletionMarker


def test_seed_sweep_launcher_runs_runner_over_resolved_config(monkeypatch):
    """Verifies a single-seed launch resolves the config through the CLI and runs the seed sweep runner over it."""
    captured = {}

    class Resolved:
        """Config stand-in resolved to a single seed."""
        seed  = 4
        seeds = [4]
        ddp   = False

    resolved = Resolved()

    class FakeCli:
        """Stand-in config CLI recording the description and argv it receives."""
        def __init__(self, config, description):
            """Records the launcher description and exposes empty overrides."""
            captured["description"] = description
            self.overrides          = {}
        def apply(self, argv):
            """Records argv and returns the resolved config."""
            captured["argv"] = argv
            return resolved

    class FakeSweep:
        """Stand-in seed sweep runner recording its config and runner class."""
        def __init__(self, config, runner_class):
            """Records the config and runner class handed to the sweep."""
            captured["sweep"] = (config, runner_class)
        def run(self):
            """Marks the sweep as executed."""
            captured["ran"] = True

    runner = object()

    monkeypatch.setattr(mod, "ConfigCli", FakeCli)
    monkeypatch.setattr(mod, "SeedSweepRunner", FakeSweep)

    mod.SeedSweepLauncher(object(), runner, "desc", entry_script=Path("/entry/train.py")).run([])

    assert captured["description"] == "desc"
    assert captured["sweep"]       == (resolved, runner)
    assert captured["ran"]         is True


def test_seed_sweep_launcher_fans_multi_seed_runs_across_the_pool(monkeypatch):
    """Verifies a multi-seed launch goes through the GPU fan-out scheduler with the CLI overrides, entry script and base label."""
    captured = {}

    class Resolved:
        """Config stand-in resolved to three seeds and a base field."""
        seed       = 0
        seeds      = [0, 1, 2]
        ddp        = False
        base_field = "conv2d_ae"

    resolved = Resolved()

    class FakeCli:
        """Stand-in config CLI exposing one training override."""
        def __init__(self, config, description):
            """Exposes a single epochs override."""
            self.overrides = {"training.epochs": "3"}
        def apply(self, argv):
            """Returns the resolved config."""
            return resolved

    class FakeScheduler:
        """Stand-in fan-out scheduler recording its factory arguments."""
        @classmethod
        def for_runner(cls, config, cli_overrides, entry_script, runner_factory, base_label=None, infer_at_end=False):
            """Records the config, overrides, entry script, runner factory and base label."""
            captured["for_runner"] = (config, cli_overrides, entry_script, runner_factory, base_label)
            return cls()
        def run(self):
            """Marks the fan-out as executed."""
            captured["ran"] = True

    runner = object()

    monkeypatch.setattr(mod, "ConfigCli", FakeCli)
    monkeypatch.setattr(mod, "SeedFanoutScheduler", FakeScheduler)

    mod.SeedSweepLauncher(object(), runner, "desc", entry_script=Path("/entry/train.py"), base_attr="base_field").run([])

    assert captured["for_runner"] == (resolved, {"training.epochs": "3"}, Path("/entry/train.py"), runner, "conv2d_ae")
    assert captured["ran"]        is True


def test_backbone_launcher_trial_runs_single_runner(monkeypatch):
    """Verifies the --trial flag runs one training runner in-process instead of the scheduler."""
    ran = {}

    class FakeSingleRunner:
        """Stand-in single training runner recording its config and execution."""
        def __init__(self, cfg):
            """Records the config handed to the runner."""
            ran["single"] = cfg
        def run(self):
            """Marks the single run as executed."""
            ran["ran"] = True

    class FakeScheduler:
        """Stand-in trial scheduler that must not be constructed in trial mode."""
        def __init__(self, **kwargs):
            """Records that the scheduler was constructed."""
            ran["scheduler"] = True
        def run(self):
            """Marks the scheduler as executed."""
            ran["scheduler_ran"] = True

    class FakeCli:
        """Stand-in config CLI returning a single-seed backbone entry config."""
        def __init__(self, config, description):
            """Exposes empty CLI overrides."""
            self.overrides = {}
        def apply(self, argv):
            """Returns a backbone entry config with a single seed."""
            config       = BackboneEntryConfig()
            config.seeds = [0]
            return config

    monkeypatch.setattr(backbone_pipeline, "SingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(backbone_pipeline, "TrainScheduler", FakeScheduler)
    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    backbone_pipeline.BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py")).run(["--trial"])

    assert ran.get("ran") is True
    assert "scheduler" not in ran


def test_backbone_launcher_fans_multi_seed_runs_across_the_pool(monkeypatch):
    """Verifies a default backbone launch fans seeds across the GPU pool with the single runner as factory."""
    ran = {}

    class FakeSingleRunner:
        """Stand-in single training runner used as the fan-out runner factory."""
        def __init__(self, cfg):
            """Records that a single runner was constructed."""
            ran["single"] = True
        def run(self):
            """Marks the single run as executed."""
            ran["single_ran"] = True

    class FakeFanout:
        """Stand-in fan-out scheduler recording the entry script and runner factory."""
        @classmethod
        def for_runner(cls, config, cli_overrides, entry_script, runner_factory, base_label=None, infer_at_end=False):
            """Records the entry script, runner factory and batched-inference flag."""
            ran["for_runner"] = (entry_script, runner_factory)
            ran["infer_at_end"] = infer_at_end
            return cls()
        def run(self):
            """Marks the fan-out as executed."""
            ran["fanout_ran"] = True

    class FakeCli:
        """Stand-in config CLI returning a default backbone entry config."""
        def __init__(self, config, description):
            """Exposes empty CLI overrides."""
            self.overrides = {}
        def apply(self, argv):
            """Returns a default backbone entry config."""
            return BackboneEntryConfig()

    monkeypatch.setattr(backbone_pipeline, "SingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(backbone_pipeline, "SeedFanoutScheduler", FakeFanout)
    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    entry = Path("/entry/train_backbone.py")
    backbone_pipeline.BackboneTrainingLauncher(entry_script=entry).run([])

    assert ran.get("fanout_ran") is True
    assert ran["for_runner"]     == (entry, FakeSingleRunner)
    assert "single_ran" not in ran


def test_backbone_launcher_fans_out_when_trials_enabled(monkeypatch):
    """Verifies an enabled trials mode dispatches through the trial scheduler carrying the entry script."""
    ran = {}

    class FakeSingleRunner:
        """Stand-in single training runner that must not run under a trials sweep."""
        def __init__(self, cfg):
            """Records that a single runner was constructed."""
            ran["single"] = True
        def run(self):
            """Marks the single run as executed."""
            ran["single_ran"] = True

    class FakeScheduler:
        """Stand-in trial scheduler recording the entry script it was built with."""
        def __init__(self, **kwargs):
            """Records the entry script keyword argument."""
            ran["entry_script"] = kwargs["entry_script"]
        def run(self):
            """Marks the scheduler as executed."""
            ran["scheduler_ran"] = True

    class FakeConfig:
        """Entry config stand-in with trials enabled and inference held back."""
        trials_enabled = True
        curriculum     = default_curriculum()
        infer_after    = False
        infer_at_end   = False

    class FakeCli:
        """Stand-in config CLI returning the trials-enabled config."""
        def __init__(self, config, description):
            """Exposes empty CLI overrides."""
            self.overrides = {}
        def apply(self, argv):
            """Returns the trials-enabled config stand-in."""
            return FakeConfig()

    monkeypatch.setattr(backbone_pipeline, "SingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(backbone_pipeline, "TrainScheduler", FakeScheduler)
    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    entry = Path("/entry/train_backbone.py")
    backbone_pipeline.BackboneTrainingLauncher(entry_script=entry).run([])

    assert ran.get("scheduler_ran") is True
    assert ran["entry_script"] == entry
    assert "single_ran" not in ran


def _planned_units(monkeypatch, names):
    """Replaces the scheduler planner with one planning the given unit names without overrides."""
    class _Planner:
        """Planner stand-in yielding fixed unit names."""
        def plan(self):
            """Returns one planned unit per configured name with empty overrides."""
            return [(name, {}) for name in names]

        def summary(self):
            """Returns an empty planner summary."""
            return {}

    monkeypatch.setattr(backbone_pipeline.TrainScheduler, "planner", lambda self: _Planner())


def test_trials_scheduler_resumes_from_a_sweep_folder(tmp_path, monkeypatch):
    """Verifies resuming from a sweep folder reuses it as the runs root, skips completed units and forwards resume only into unfinished trial argvs."""
    _planned_units(monkeypatch, ["t-a", "t-b", "t-c"])

    root = tmp_path / "old_sweep"
    (root / "t-a").mkdir(parents=True)
    (root / "t-b").mkdir()
    CompletionMarker.stamp(root / "t-a", {})

    config             = BackboneEntryConfig()
    config.seeds       = (0,)
    config.logdir      = tmp_path / "fresh"
    config.resume_from = str(root)

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={"resume_from": str(root)}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root              == root
    assert scheduler.resume_plan.completed  == ["t-a"]
    assert scheduler.resume_plan.unfinished == ["t-b"]
    assert scheduler.resume_plan.missing    == ["t-c"]
    assert scheduler.units()                == [("t-b", {}), ("t-c", {})]
    assert config.resume                    is True
    assert config.training.resume           is True
    assert "resume_from" not in scheduler.forward_overrides

    argv = scheduler.trial_argv("t-b", {}, batched_inference=True)
    assert argv[argv.index("--logdir") + 1]          == str(root)
    assert argv[argv.index("--resume") + 1]          == "true"
    assert argv[argv.index("--training.resume") + 1] == "true"


def test_trials_scheduler_trains_plan_missing_units_from_scratch(tmp_path, monkeypatch):
    """Verifies a unit the resume plan found no run directory for gets an argv without resume flags, so it trains fresh instead of crashing on absent trainer state."""
    _planned_units(monkeypatch, ["t-a", "t-b", "t-c"])

    root = tmp_path / "old_sweep"
    (root / "t-a").mkdir(parents=True)
    (root / "t-b").mkdir()
    CompletionMarker.stamp(root / "t-a", {})

    config             = BackboneEntryConfig()
    config.seeds       = (0,)
    config.logdir      = tmp_path / "fresh"
    config.resume_from = str(root)

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={"resume_from": str(root)}, entry_script=Path("/entry/train_backbone.py"))

    argv = scheduler.trial_argv("t-c", {}, batched_inference=True)
    assert argv[argv.index("--logdir") + 1] == str(root)
    assert "--resume"          not in argv
    assert "--training.resume" not in argv


def test_seed_fanout_forwards_resume_only_to_seeds_with_run_dirs(tmp_path):
    """Verifies a resumed seed sweep drops completed seeds and withholds the resume overrides from a seed the plan found no run directory for."""
    root = tmp_path / "prefix_base"
    (root / "seed0").mkdir(parents=True)
    (root / "seed1").mkdir()
    CompletionMarker.stamp(root / "seed0", {})

    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.resume_from = str(root)

    scheduler   = SeedFanoutScheduler(config, {}, Path("/entry/train.py"), base_name="base", run_dir=root)
    units, plan = scheduler._resume_units([(0, "base/seed0"), (1, "base/seed1"), (2, "base/seed2")])

    assert plan.completed  == ["seed0"]
    assert plan.unfinished == ["seed1"]
    assert plan.missing    == ["seed2"]
    assert units           == [(1, "base/seed1"), (2, "base/seed2")]

    assert scheduler._resumed(1) == {"resume": True, "training.resume": True}
    assert scheduler._resumed(2) == {}
    assert "--training.resume"     in scheduler._job("base/seed1", 1).command
    assert "--training.resume" not in scheduler._job("base/seed2", 2).command


def test_trials_scheduler_resolves_the_root_from_a_single_run(tmp_path, monkeypatch):
    """Verifies pointing resume_from at one run directory resolves the sweep root and marks that unit unfinished."""
    _planned_units(monkeypatch, ["t-a", "t-b"])

    root = tmp_path / "physics"
    (root / "t-a").mkdir(parents=True)

    config             = BackboneEntryConfig()
    config.seeds       = (0,)
    config.logdir      = tmp_path / "fresh"
    config.resume_from = str(root / "t-a")

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root             == root
    assert scheduler.resume_plan.unfinished == ["t-a"]


def test_trials_scheduler_rejects_a_folder_without_planned_runs(tmp_path, monkeypatch):
    """Verifies a resume folder holding no planned run directory exits."""
    _planned_units(monkeypatch, ["t-a", "t-b"])

    (tmp_path / "empty").mkdir()

    config             = BackboneEntryConfig()
    config.seeds       = (0,)
    config.logdir      = tmp_path / "fresh"
    config.resume_from = str(tmp_path / "empty")

    with pytest.raises(SystemExit, match="holds no run directory matching"):
        backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))


def test_ablation_scheduler_houses_runs_in_ablation_dir(tmp_path):
    """Verifies the ablation mode places runs and batch logs under the ablation directory."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ablation"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root == tmp_path / "ablation"
    assert scheduler.log_dir   == tmp_path / "ablation" / "batch_train_logs"

    job = scheduler._job("model_abl-0-full", {"curriculum.enabled": True})

    assert job.command[-2:] == ["--logdir", str(tmp_path / "ablation")]
    assert job.log_path     == tmp_path / "ablation" / "batch_train_logs" / "model_abl-0-full.log"


def test_patch_scheduler_plans_the_grid_under_the_pixel_budget(tmp_path):
    """Verifies the patch mode houses runs under patch/ and plans the rectangular grid with budgeted batches."""
    config                      = BackboneEntryConfig()
    config.logdir               = tmp_path
    config.trials_mode          = "patch"
    config.backbone_name        = "unet"
    config.model_overrides      = {"features": [64, 128, 256]}
    config.patch_trials.maximum = (16, 16)

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root == tmp_path / "patch"

    plans = scheduler.planner().plan()
    assert [name for name, _ in plans] == ["p-008x008", "p-008x016", "p-016x008", "p-016x016"]

    budget = config.training.batch_size * config.training.patch_size[0] * config.training.patch_size[1]
    for _name, overrides in plans:
        size = overrides["training.patch_size"]
        assert overrides["training.batch_size"]      == budget // (size[0] * size[1])
        assert overrides["pretrain.find_batch_size"] is False


def test_context_scheduler_plans_the_receptive_field_ladder(tmp_path):
    """Verifies the context mode plans the pixel-MLP rung and the four constant-depth local CNN rungs with their widths and kernel patterns."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "context"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    plans = scheduler.planner().plan()

    assert plans == [
        ("ctx-mlp",   {"backbone_name": "pixel_mlp", "model_overrides": {"features": [1277] * 20}}),
        ("ctx-cnn01", {"backbone_name": "local_cnn", "model_overrides": {"features": [1277] * 10, "block_kernels": [1] * 10}}),
        ("ctx-cnn09", {"backbone_name": "local_cnn", "model_overrides": {"features": [848]  * 10, "block_kernels": [3] * 2 + [1] * 8}}),
        ("ctx-cnn29", {"backbone_name": "local_cnn", "model_overrides": {"features": [502]  * 10, "block_kernels": [3] * 7 + [1] * 3}}),
        ("ctx-cnn41", {"backbone_name": "local_cnn", "model_overrides": {"features": [426]  * 10, "block_kernels": [3] * 10}}),
    ]


@pytest.mark.dataset
def test_reach_scheduler_houses_runs_and_plans_the_size_matched_arms(tmp_path, test_data_dir, params_dir):
    """Verifies the reach mode plans the size-matched local CNN and unet arms on 32 by 32 patches."""
    config                       = BackboneEntryConfig()
    config.logdir                = tmp_path
    config.trials_mode           = "reach"
    config.paths.dataset_path    = str(test_data_dir)
    config.paths.parameters_path = params_dir / "parameters.npy"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root == tmp_path / "reach"

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["reach-cnn33", "reach-unet"]
    assert plans["reach-cnn33"]["backbone_name"]   == "local_cnn"
    assert plans["reach-cnn33"]["model_overrides"] == {"features": [478] * 8, "dropout": 0.15, "trunk_wd": 1e-4}
    assert plans["reach-unet"]["backbone_name"]    == "unet"
    assert plans["reach-unet"]["model_overrides"]  == {"dropout": 0.15}
    assert all(overrides["training.patch_size"] == (32, 32) for overrides in plans.values())


def test_head_scheduler_plans_the_head_matching_grid(tmp_path):
    """Verifies the head mode plans the head by parameter-matching grid over the unet backbone."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "head"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["hm-conv-sorted_gt", "hm-conv-hungarian", "hm-set_pred-sorted_gt", "hm-set_pred-hungarian"]
    assert all(overrides["backbone_name"] == "unet" for overrides in plans.values())
    assert plans["hm-set_pred-hungarian"]["backbone_head"]                       == "set_pred"
    assert plans["hm-set_pred-hungarian"]["curriculum.warmup.param_matching"]    == "hungarian"
    assert plans["hm-set_pred-hungarian"]["curriculum.complete.param_matching"]  == "hungarian"


def test_augmentation_scheduler_plans_the_cumulative_ladder(tmp_path):
    """Verifies the augmentation mode plans the ladder from no augmentation through flips to the noise rungs."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["aug-0-none", "aug-1-hflip", "aug-2-vflip", "aug-3-noise0.01", "aug-3-noise0.03", "aug-3-noise0.1"]
    assert plans["aug-0-none"]["augmentation.p_flip_h"]        == 0.0
    assert plans["aug-1-hflip"]["augmentation.p_flip_h"]       == 0.5
    assert plans["aug-1-hflip"]["augmentation.p_flip_v"]       == 0.0
    assert plans["aug-2-vflip"]["augmentation.p_flip_v"]       == 0.5
    assert plans["aug-2-vflip"]["augmentation.p_rot90"]        == 0.0
    assert plans["aug-3-noise0.03"]["augmentation.p_noise"]    == 0.5
    assert plans["aug-3-noise0.03"]["augmentation.noise_std"]  == 0.03


def test_scheduler_fans_out_one_gpu_job_per_trial_seed(tmp_path):
    """Verifies multi-seed trials expand into one job per trial and seed, each argv pinned to its own seed."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"
    config.seeds       = [0, 1]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={"seeds": [0, 1]}, entry_script=Path("/entry/train_backbone.py"))

    units = scheduler._seed_units(scheduler.planner().plan(), [0, 1])

    planned = [run_name for run_name, _ in scheduler.planner().plan()]
    assert [name for name, _ in units]                    == [f"{run_name}/seed{seed}" for run_name in planned for seed in (0, 1)]
    assert [overrides["seed"] for _, overrides in units]  == [0, 1] * len(planned)
    assert [overrides["seeds"] for _, overrides in units] == [(0,), (1,)] * len(planned)

    job  = scheduler._job(*units[1])
    argv = job.command

    assert argv[argv.index("--seed") + 1]  == "1"
    assert argv[argv.index("--seeds") + 1] == "[1]"
    assert argv.count("--seeds")           == 1
    assert job.log_path                    == tmp_path / "augmentation" / "batch_train_logs" / "aug-0-none" / "seed1.log"


def test_scheduler_keeps_single_seed_trials_unexpanded(tmp_path):
    """Verifies a single-seed sweep leaves the planned trials untouched and injects no seed overrides."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"
    config.seeds       = [7]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    plans = scheduler.planner().plan()
    units = scheduler._seed_units(plans, [7])

    assert units == plans
    assert all("seed" not in overrides and "seeds" not in overrides for _, overrides in units)


def test_scheduler_run_dispatches_per_seed_jobs(tmp_path, monkeypatch):
    """Verifies the run dispatches every per-seed job and writes the results file."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"
    config.seeds       = [0, 1]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    captured = {}

    def fake_run_queue(jobs):
        """Captures the queued jobs and reports each as done."""
        captured["jobs"] = jobs
        return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    scheduler.run()

    planned = [run_name for run_name, _ in scheduler.planner().plan()]
    assert [job.name for job in captured["jobs"]] == [f"{run_name}/seed{seed}" for run_name in planned for seed in (0, 1)]
    assert scheduler.results_path.is_file()


def test_scheduler_batched_mode_holds_back_trial_inference(tmp_path):
    """Verifies batched inference mode forwards no overrides and disables inference on the training jobs."""
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "augmentation"
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={"infer_at_end": "True"}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.forward_overrides == {}

    argv = scheduler._job("aug-on", {}).command
    assert argv[argv.index("--infer_after") + 1] == "false"


def test_scheduler_inference_jobs_resume_into_inference_only(tmp_path):
    """Verifies batched inference jobs resume the trained unit with inference enabled and log under a dedicated file."""
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "augmentation"
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    job  = scheduler._inference_job("aug-on", {})
    argv = job.command

    assert "--trial" in argv
    assert argv[argv.index("--infer_after") + 1] == "true"
    assert argv[argv.index("--resume") + 1]      == "true"
    assert job.log_path                          == tmp_path / "augmentation" / "batch_train_logs" / "aug-on_infer.log"


def test_scheduler_run_batches_inference_after_all_trials(tmp_path, monkeypatch):
    """Verifies the run trains every trial first, then runs a second inference-only queue, writing both result files."""
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "augmentation"
    config.seeds        = [0]
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    queues = []

    def fake_run_queue(jobs):
        """Captures each queue and reports every job as done."""
        queues.append(jobs)
        return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    scheduler.run()

    planned = [run_name for run_name, _ in scheduler.planner().plan()]
    assert [job.name for job in queues[0]] == planned
    assert [job.name for job in queues[1]] == planned
    assert all(job.command[job.command.index("--infer_after") + 1] == "false" for job in queues[0])
    assert all(job.command[job.command.index("--infer_after") + 1] == "true" for job in queues[1])
    assert scheduler.results_path.is_file()
    assert scheduler.infer_results_path.is_file()


def test_scheduler_batched_inference_skips_failed_trainings(tmp_path, monkeypatch):
    """Verifies a failed training trial exits and is excluded from the batched inference queue."""
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "augmentation"
    config.seeds        = [0]
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    queues = []

    def fake_run_queue(jobs):
        """Captures each queue and fails one trial on the training pass."""
        queues.append(jobs)
        first_pass = len(queues) == 1
        return [{"name": job.name, "gpu": 0, "status": "FAILED" if first_pass and job.name == "aug-0-none" else "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    with pytest.raises(SystemExit, match="1 of 6 training trials failed"):
        scheduler.run()

    trained = [run_name for run_name, _ in scheduler.planner().plan() if run_name != "aug-0-none"]
    assert [job.name for job in queues[1]] == trained


def test_scheduler_without_batched_mode_runs_a_single_queue(tmp_path, monkeypatch):
    """Verifies the default mode runs one queue, passes no inference flag and writes no inference results file."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"
    config.seeds       = [0]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    queues = []

    def fake_run_queue(jobs):
        """Captures the queued jobs and reports each as done."""
        queues.append(jobs)
        return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    scheduler.run()

    assert len(queues) == 1
    assert all("--infer_after" not in job.command for job in queues[0])
    assert not scheduler.infer_results_path.exists()


def test_backbone_launcher_rejects_both_inference_flags(monkeypatch):
    """Verifies setting both infer_after and infer_at_end exits."""
    class FakeCli:
        """Stand-in config CLI returning a config with both inference flags set."""
        def __init__(self, config, description):
            """Exposes empty CLI overrides."""
            self.overrides = {}
        def apply(self, argv):
            """Returns a config with both inference flags enabled."""
            config              = BackboneEntryConfig()
            config.infer_after  = True
            config.infer_at_end = True
            return config

    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    with pytest.raises(SystemExit, match="mutually exclusive"):
        backbone_pipeline.BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py")).run([])


def test_scheduler_houses_each_mode_in_its_own_dir(tmp_path):
    """Verifies every trials mode houses its runs, batch logs and job logdir under a directory named after the mode."""
    for mode in ("curriculum", "warmup", "presence", "pair", "secondary", "patch", "input", "context", "head", "augmentation", "ema", "amp", "ablation"):
        config             = BackboneEntryConfig()
        config.logdir      = tmp_path
        config.trials_mode = mode

        scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

        assert scheduler.runs_root == tmp_path / mode
        assert scheduler.log_dir   == tmp_path / mode / "batch_train_logs"

        job = scheduler._job("model_trial", {})
        assert job.command[-2:] == ["--logdir", str(tmp_path / mode)]


def test_units_expand_trials_by_seeds(tmp_path):
    """Verifies units expand to one uniquely named entry per trial and seed."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ablation"
    config.seeds       = [0, 1]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))
    units     = scheduler.units()
    trials    = scheduler.planner().plan()

    assert len(units) == 2 * len(trials)
    assert all(overrides["seeds"] in ((0,), (1,)) for _, overrides in units)
    assert len({run_name for run_name, _ in units}) == len(units)


def test_trial_argv_matches_job_command(tmp_path):
    """Verifies the job command is the trial argv prefixed by the interpreter, entry script and --trial."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ablation"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))
    overrides = {"curriculum.enabled": True}
    job       = scheduler._job("model_abl-0-full", overrides)

    assert job.command[3:] == scheduler.trial_argv("model_abl-0-full", overrides)
    assert job.command[2]  == "--trial"


def test_trial_argv_refuses_to_build_units_that_lose_their_inference(tmp_path):
    """Verifies building a trial argv under batched inference requires the batched flag, which disables per-trial inference."""
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "ablation"
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    with pytest.raises(ValueError, match="inference_argv"):
        scheduler.trial_argv("model_abl-0-full", {})

    batched = scheduler.trial_argv("model_abl-0-full", {}, batched_inference=True)

    assert batched[batched.index("--infer_after") + 1] == "false"


def test_inference_argv_resumes_the_trained_unit(tmp_path):
    """Verifies the inference argv resumes the named unit in its mode directory with inference enabled, matching the inference job command."""
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "ablation"
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))
    argv      = scheduler.inference_argv("model_abl-0-full", {})

    assert argv[argv.index("--infer_after") + 1] == "true"
    assert argv[argv.index("--resume") + 1]      == "true"
    assert argv[argv.index("--run_name") + 1]    == "model_abl-0-full"
    assert argv[argv.index("--logdir") + 1]      == str(tmp_path / "ablation")

    job = scheduler._inference_job("model_abl-0-full", {})

    assert job.command[3:] == argv


def test_local_device_pins_are_not_forwarded_to_trial_units(tmp_path):
    """Verifies GPU pins are stripped from the overrides forwarded into trial argv."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ablation"

    overrides = {"gpu": 3, "gpus": [0, 1, 3], "gpus_file": "/tmp/pool.json", "curriculum.enabled": True}
    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides=overrides, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.forward_overrides == {"curriculum.enabled": True}

    argv = scheduler.trial_argv("model_abl-0-full", {})

    assert "--gpu" not in argv
    assert "--gpus" not in argv
    assert "--gpus_file" not in argv


def test_building_a_scheduler_does_not_touch_the_batch_log_dir(tmp_path):
    """Verifies constructing a scheduler and building argv neither creates nor overwrites the batch log directory."""
    launcher  = backbone_pipeline.BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py"))
    scheduler = launcher.build_scheduler(["--logdir", str(tmp_path), "--trials_mode", "ablation"])

    assert scheduler.units()
    assert not scheduler.log_dir.exists()

    scheduler.log_dir.mkdir(parents=True)
    marker = scheduler.log_dir / "train_scheduler.log"
    marker.write_text("earlier batched run")

    again = launcher.build_scheduler(["--logdir", str(tmp_path), "--trials_mode", "ablation"])
    again.trial_argv("model_abl-0-full", {})

    assert marker.read_text() == "earlier batched run"


def test_building_a_scheduler_rejects_contradictory_inference_flags(tmp_path):
    """Verifies building a scheduler with both inference flags exits."""
    launcher = backbone_pipeline.BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py"))

    with pytest.raises(SystemExit, match="mutually exclusive"):
        launcher.build_scheduler(["--logdir", str(tmp_path), "--infer_after", "true", "--infer_at_end", "true"])
