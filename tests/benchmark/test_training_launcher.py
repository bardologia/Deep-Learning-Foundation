from __future__ import annotations

from pathlib import Path

import pytest

import pipelines.backbone.training.launcher as backbone_pipeline
from configuration.training    import BackboneEntryConfig, default_curriculum
from pipelines.shared.training import training_launcher as mod


def test_seed_sweep_launcher_runs_runner_over_resolved_config(monkeypatch):
    captured = {}

    class Resolved:
        seed  = 4
        seeds = [4]

    resolved = Resolved()

    class FakeCli:
        def __init__(self, config, description):
            captured["description"] = description
            self.overrides          = {}
        def apply(self, argv):
            captured["argv"] = argv
            return resolved

    class FakeSweep:
        def __init__(self, config, runner_class):
            captured["sweep"] = (config, runner_class)
        def run(self):
            captured["ran"] = True

    runner = object()

    monkeypatch.setattr(mod, "ConfigCli", FakeCli)
    monkeypatch.setattr(mod, "SeedSweepRunner", FakeSweep)

    mod.SeedSweepLauncher(object(), runner, "desc", entry_script=Path("/entry/train.py")).run([])

    assert captured["description"] == "desc"
    assert captured["sweep"]       == (resolved, runner)
    assert captured["ran"]         is True


def test_seed_sweep_launcher_fans_multi_seed_runs_across_the_pool(monkeypatch):
    captured = {}

    class Resolved:
        seed       = 0
        seeds      = [0, 1, 2]
        base_field = "conv2d_ae"

    resolved = Resolved()

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {"training.epochs": "3"}
        def apply(self, argv):
            return resolved

    class FakeScheduler:
        @classmethod
        def for_runner(cls, config, cli_overrides, entry_script, runner_factory, base_label=None):
            captured["for_runner"] = (config, cli_overrides, entry_script, runner_factory, base_label)
            return cls()
        def run(self):
            captured["ran"] = True

    runner = object()

    monkeypatch.setattr(mod, "ConfigCli", FakeCli)
    monkeypatch.setattr(mod, "SeedFanoutScheduler", FakeScheduler)

    mod.SeedSweepLauncher(object(), runner, "desc", entry_script=Path("/entry/train.py"), base_attr="base_field").run([])

    assert captured["for_runner"] == (resolved, {"training.epochs": "3"}, Path("/entry/train.py"), runner, "conv2d_ae")
    assert captured["ran"]        is True


def test_backbone_launcher_trial_runs_single_runner(monkeypatch):
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
        curriculum     = default_curriculum()
        infer_after    = False
        infer_at_end   = False

    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
            return FakeConfig()

    monkeypatch.setattr(backbone_pipeline, "SingleTrainRunner", FakeSingleRunner)
    monkeypatch.setattr(backbone_pipeline, "TrainScheduler", FakeScheduler)
    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    entry = Path("/entry/train_backbone.py")
    backbone_pipeline.BackboneTrainingLauncher(entry_script=entry).run([])

    assert ran.get("scheduler_ran") is True
    assert ran["entry_script"] == entry
    assert "single_ran" not in ran


def test_ablation_scheduler_houses_runs_in_ablation_dir(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ablation"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root == tmp_path / "ablation"
    assert scheduler.log_dir   == tmp_path / "ablation" / "batch_train_logs"

    job = scheduler._job("model_abl-0-full", {"curriculum.enabled": True})

    assert job.command[-2:] == ["--logdir", str(tmp_path / "ablation")]
    assert job.log_path     == tmp_path / "ablation" / "batch_train_logs" / "model_abl-0-full.log"


def test_context_scheduler_plans_the_receptive_field_ladder(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "context"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    plans = scheduler.planner().plan()

    assert plans == [
        ("ctx-cnn01", {"backbone_name": "local_cnn", "model_overrides": {"features": [1277] * 10, "block_kernels": [1] * 10}}),
        ("ctx-cnn09", {"backbone_name": "local_cnn", "model_overrides": {"features": [848]  * 10, "block_kernels": [3] * 2 + [1] * 8}}),
        ("ctx-cnn29", {"backbone_name": "local_cnn", "model_overrides": {"features": [502]  * 10, "block_kernels": [3] * 7 + [1] * 3}}),
        ("ctx-cnn41", {"backbone_name": "local_cnn", "model_overrides": {"features": [426]  * 10, "block_kernels": [3] * 10}}),
    ]


def test_reach_scheduler_houses_runs_and_plans_the_size_matched_arms(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "reach"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.runs_root == tmp_path / "reach"

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["reach-cnn33", "reach-unet"]
    assert plans["reach-cnn33"]["backbone_name"]   == "local_cnn"
    assert plans["reach-cnn33"]["model_overrides"] == {"features": [479] * 8, "dropout": 0.15, "trunk_wd": 1e-4}
    assert plans["reach-unet"]["backbone_name"]    == "unet"
    assert plans["reach-unet"]["model_overrides"]  == {"dropout": 0.15}
    assert all(overrides["training.patch_size"] == (32, 32) for overrides in plans.values())


def test_head_scheduler_plans_the_head_matching_grid(tmp_path):
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


def test_augmentation_scheduler_plans_the_on_off_pair(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["aug-on", "aug-off"]
    assert plans["aug-on"]["augmentation.p_flip_h"]  == 0.5
    assert plans["aug-on"]["augmentation.p_flip_v"]  == 0.5
    assert plans["aug-on"]["augmentation.p_rot90"]   == 0.0
    assert plans["aug-off"]["augmentation.p_flip_h"] == 0.0
    assert plans["aug-off"]["augmentation.p_flip_v"] == 0.0


def test_normalization_scheduler_plans_the_cumulative_ladder(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "normalization"

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    plans = dict(scheduler.planner().plan())

    assert list(plans) == ["nrm-0-initial", "nrm-1-pass_mag", "nrm-2-ifg_phase", "nrm-3-out_amp", "nrm-4-out_sigma"]
    assert plans["nrm-0-initial"]["normalization.pass_mag"]    == "zscore_log1p"
    assert plans["nrm-1-pass_mag"]["normalization.pass_mag"]   == "robust_iqr_log1p"
    assert plans["nrm-3-out_amp"]["normalization.out_amp"]     == "robust_iqr_log1p"
    assert plans["nrm-3-out_amp"]["normalization.out_sigma"]   == "zscore"
    assert plans["nrm-4-out_sigma"]["normalization.out_sigma"] == "robust_iqr_log1p"
    assert all("normalization.out_mu" not in overrides for overrides in plans.values())


def test_scheduler_fans_out_one_gpu_job_per_trial_seed(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"
    config.seeds       = [0, 1]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={"seeds": [0, 1]}, entry_script=Path("/entry/train_backbone.py"))

    units = scheduler._seed_units(scheduler.planner().plan(), [0, 1])

    assert [name for name, _ in units]                    == ["aug-on/seed0", "aug-on/seed1", "aug-off/seed0", "aug-off/seed1"]
    assert [overrides["seed"] for _, overrides in units]  == [0, 1, 0, 1]
    assert [overrides["seeds"] for _, overrides in units] == [(0,), (1,), (0,), (1,)]

    job  = scheduler._job(*units[1])
    argv = job.command

    assert argv[argv.index("--seed") + 1]  == "1"
    assert argv[argv.index("--seeds") + 1] == "[1]"
    assert argv.count("--seeds")           == 1
    assert job.log_path                    == tmp_path / "augmentation" / "batch_train_logs" / "aug-on" / "seed1.log"


def test_scheduler_keeps_single_seed_trials_unexpanded(tmp_path):
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
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"
    config.seeds       = [0, 1]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    captured = {}

    def fake_run_queue(jobs):
        captured["jobs"] = jobs
        return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    scheduler.run()

    assert [job.name for job in captured["jobs"]] == ["aug-on/seed0", "aug-on/seed1", "aug-off/seed0", "aug-off/seed1"]
    assert scheduler.results_path.is_file()


def test_scheduler_batched_mode_holds_back_trial_inference(tmp_path):
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "augmentation"
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={"infer_at_end": "True"}, entry_script=Path("/entry/train_backbone.py"))

    assert scheduler.forward_overrides == {}

    argv = scheduler._job("aug-on", {}).command
    assert argv[argv.index("--infer_after") + 1] == "false"


def test_scheduler_inference_jobs_resume_into_inference_only(tmp_path):
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
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "augmentation"
    config.seeds        = [0]
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    queues = []

    def fake_run_queue(jobs):
        queues.append(jobs)
        return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    scheduler.run()

    assert [job.name for job in queues[0]] == ["aug-on", "aug-off"]
    assert [job.name for job in queues[1]] == ["aug-on", "aug-off"]
    assert all(job.command[job.command.index("--infer_after") + 1] == "false" for job in queues[0])
    assert all(job.command[job.command.index("--infer_after") + 1] == "true" for job in queues[1])
    assert scheduler.results_path.is_file()
    assert scheduler.infer_results_path.is_file()


def test_scheduler_batched_inference_skips_failed_trainings(tmp_path, monkeypatch):
    config              = BackboneEntryConfig()
    config.logdir       = tmp_path
    config.trials_mode  = "augmentation"
    config.seeds        = [0]
    config.infer_at_end = True

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    queues = []

    def fake_run_queue(jobs):
        queues.append(jobs)
        first_pass = len(queues) == 1
        return [{"name": job.name, "gpu": 0, "status": "FAILED" if first_pass and job.name == "aug-off" else "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    with pytest.raises(SystemExit, match="1 of 2 training trials failed"):
        scheduler.run()

    assert [job.name for job in queues[1]] == ["aug-on"]


def test_scheduler_without_batched_mode_runs_a_single_queue(tmp_path, monkeypatch):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "augmentation"
    config.seeds       = [0]

    scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

    queues = []

    def fake_run_queue(jobs):
        queues.append(jobs)
        return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    monkeypatch.setattr(scheduler.stage, "_run_queue", fake_run_queue)

    scheduler.run()

    assert len(queues) == 1
    assert all("--infer_after" not in job.command for job in queues[0])
    assert not scheduler.infer_results_path.exists()


def test_backbone_launcher_rejects_both_inference_flags(monkeypatch):
    class FakeCli:
        def __init__(self, config, description):
            self.overrides = {}
        def apply(self, argv):
            config              = BackboneEntryConfig()
            config.infer_after  = True
            config.infer_at_end = True
            return config

    monkeypatch.setattr(backbone_pipeline, "ConfigCli", FakeCli)

    with pytest.raises(SystemExit, match="mutually exclusive"):
        backbone_pipeline.BackboneTrainingLauncher(entry_script=Path("/entry/train_backbone.py")).run([])


def test_scheduler_houses_each_mode_in_its_own_dir(tmp_path):
    for mode in ("curriculum", "warmup", "presence", "physics", "pair", "secondary", "patch", "input", "context", "head", "augmentation", "normalization", "ablation"):
        config             = BackboneEntryConfig()
        config.logdir      = tmp_path
        config.trials_mode = mode

        scheduler = backbone_pipeline.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))

        assert scheduler.runs_root == tmp_path / mode
        assert scheduler.log_dir   == tmp_path / mode / "batch_train_logs"

        job = scheduler._job("model_trial", {})
        assert job.command[-2:] == ["--logdir", str(tmp_path / mode)]
