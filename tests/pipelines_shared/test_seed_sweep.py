from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import pytest

import pipelines.shared.training.seed_sweep as mod
from pipelines.shared.training.seed_sweep import SeedFanoutScheduler, SeedSet, SeedSweepRunner


@dataclass
class _Config:
    run_name        : str | None = None
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=list)
    logdir          : Path       = Path("/logs")
    gpus            : list[int]  = field(default_factory=lambda: [0, 1])
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0


def _factory(record):
    class _Runner:
        def __init__(self, config):
            self.config = config
        def _resolve_run_name(self):
            return f"label_{self.config.run_name}"
        def run(self):
            record.append((self.config.seed, self.config.run_name))
            return self.config.seed
    return _Runner


def test_empty_seeds_runs_once_with_base_seed_and_no_renaming():
    record = []

    result = SeedSweepRunner(_Config(seed=3), _factory(record)).run()

    assert record == [(3, None)]
    assert result == 3


def test_empty_seeds_preserves_explicit_run_name():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seed=7), _factory(record)).run()

    assert record == [(7, "exp")]


def test_single_explicit_seed_runs_once_with_that_seed():
    record = []

    SeedSweepRunner(_Config(run_name="exp", seed=0, seeds=[5]), _factory(record)).run()

    assert record == [(5, "exp")]


def test_multi_seed_config_is_rejected_loudly():
    with pytest.raises(ValueError, match="exactly one seed"):
        SeedSweepRunner(_Config(seeds=[0, 1, 2]), _factory([])).run()


def test_base_run_name_prefers_the_explicit_run_name():
    assert SeedSweepRunner.base_run_name(_Config(run_name="exp"), base_label="model") == "exp"


def test_base_run_name_stamps_a_label_prefix(monkeypatch):
    monkeypatch.setattr(mod.RunTag, "now", staticmethod(lambda: "STAMP"))

    assert SeedSweepRunner.base_run_name(_Config(), base_label="model") == "model_STAMP"
    assert SeedSweepRunner.base_run_name(_Config())                     == "STAMP"


def _scheduler(tmp_path, config=None, overrides=None):
    config = config or _Config(run_name="exp", seeds=[3, 0, 17], logdir=tmp_path)
    return SeedFanoutScheduler.for_runner(config, overrides or {}, Path("/entry/train.py"), _factory([]))


def test_for_runner_resolves_the_run_dir_through_the_runner(tmp_path):
    scheduler = _scheduler(tmp_path)

    assert scheduler.base_name == "exp"
    assert scheduler.run_dir   == tmp_path / "label_exp"
    assert scheduler.log_dir   == tmp_path / "label_exp" / "batch_train_logs"


def test_for_runner_stamps_a_shared_base_when_run_name_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.RunTag, "now", staticmethod(lambda: "STAMP"))

    config    = _Config(seeds=[0, 1], logdir=tmp_path)
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]), base_label="model")

    assert scheduler.base_name == "model_STAMP"
    assert scheduler.run_dir   == tmp_path / "label_model_STAMP"


def test_jobs_pin_each_seed_and_nest_run_names(tmp_path):
    scheduler = _scheduler(tmp_path)
    job       = scheduler._job(SeedSet.run_name("exp", 17), 17)

    assert job.name     == "exp/seed17"
    assert job.log_path == scheduler.log_dir / "seed17.log"

    argv = job.command
    assert argv[1] == "/entry/train.py"
    assert argv[argv.index("--run_name") + 1] == "exp/seed17"
    assert argv[argv.index("--seed") + 1]     == "17"
    assert argv[argv.index("--seeds") + 1]    == "[17]"
    assert argv[argv.index("--logdir") + 1]   == str(tmp_path)


def test_jobs_forward_overrides_but_hold_back_scheduler_fields(tmp_path):
    overrides = {
        "training.epochs" : "3",
        "gpus"            : "[0, 1]",
        "gpus_file"       : "/pool.json",
        "poll_interval_s" : "1.0",
        "gpu"             : "2",
        "seed"            : "9",
        "seeds"           : "[9, 10]",
        "run_name"        : "cli-name",
    }

    scheduler = _scheduler(tmp_path, overrides=overrides)

    assert scheduler.forward_overrides == {"training.epochs": "3"}

    argv = scheduler._job("exp/seed0", 0).command
    assert "--gpus" not in argv
    assert "--gpus_file" not in argv
    assert "--poll_interval_s" not in argv
    assert "--gpu" not in argv
    assert argv[argv.index("--training.epochs") + 1] == "3"


def test_run_fans_out_one_job_per_seed_and_writes_ordered_results(tmp_path):
    scheduler = _scheduler(tmp_path)
    captured  = {}

    def fake_queue(jobs):
        captured["names"] = [job.name for job in jobs]
        return list(reversed([{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]))

    scheduler.stage._run_queue = fake_queue
    scheduler.run()

    assert captured["names"] == ["exp/seed3", "exp/seed0", "exp/seed17"]
    assert scheduler.results_path.is_file()

    import json
    results = json.loads(scheduler.results_path.read_text())
    assert [r["name"] for r in results] == ["exp/seed3", "exp/seed0", "exp/seed17"]


def test_batched_fanout_holds_back_inline_inference(tmp_path):
    config    = _Config(run_name="exp", seeds=[3, 0, 17], logdir=tmp_path)
    scheduler = SeedFanoutScheduler.for_runner(config, {"infer_at_end": "True"}, Path("/entry/train.py"), _factory([]), infer_at_end=True)

    assert scheduler.forward_overrides == {}

    argv = scheduler._job("exp/seed3", 3).command
    assert argv[argv.index("--infer_after") + 1] == "false"

    infer_job  = scheduler._inference_job("exp/seed3", 3)
    infer_argv = infer_job.command

    assert infer_argv[infer_argv.index("--infer_after") + 1] == "true"
    assert infer_argv[infer_argv.index("--resume") + 1]      == "true"
    assert infer_job.log_path                                == scheduler.log_dir / "seed3_infer.log"


def test_batched_fanout_runs_inference_after_all_seeds(tmp_path):
    config    = _Config(run_name="exp", seeds=[3, 0], logdir=tmp_path)
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]), infer_at_end=True)

    queues = []

    def fake_queue(jobs):
        queues.append(jobs)
        return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    scheduler.stage._run_queue = fake_queue
    scheduler.run()

    assert [job.name for job in queues[0]] == ["exp/seed3", "exp/seed0"]
    assert [job.name for job in queues[1]] == ["exp/seed3", "exp/seed0"]
    assert all(job.command[job.command.index("--infer_after") + 1] == "false" for job in queues[0])
    assert all(job.command[job.command.index("--infer_after") + 1] == "true" for job in queues[1])
    assert scheduler.infer_results_path.is_file()


def test_batched_fanout_skips_inference_for_failed_seeds(tmp_path):
    config    = _Config(run_name="exp", seeds=[3, 0], logdir=tmp_path)
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]), infer_at_end=True)

    queues = []

    def fake_queue(jobs):
        queues.append(jobs)
        first_pass = len(queues) == 1
        return [{"name": job.name, "gpu": 0, "status": "FAILED" if first_pass and job.name == "exp/seed0" else "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    scheduler.stage._run_queue = fake_queue

    with pytest.raises(SystemExit, match="1 of 2 seed runs failed"):
        scheduler.run()

    assert [job.name for job in queues[1]] == ["exp/seed3"]


def test_run_raises_when_a_seed_fails(tmp_path):
    scheduler = _scheduler(tmp_path)

    def fake_queue(jobs):
        return [{"name": job.name, "gpu": 0, "status": "FAILED", "returncode": 1, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    scheduler.stage._run_queue = fake_queue

    with pytest.raises(SystemExit, match="3 of 3 seed runs failed"):
        scheduler.run()
