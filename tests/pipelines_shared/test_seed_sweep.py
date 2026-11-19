
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from types       import SimpleNamespace

import pytest

import pipelines.shared.training.seed_sweep as mod
import pipelines.shared.training.training_launcher as launcher_mod
from pipelines.shared.training.seed_sweep import SeedFanoutScheduler, SeedSet, SeedSweepRunner
from tools.runtime.completion             import CompletionMarker


@dataclass
class _Config:
    run_name        : str | None = None
    resume          : bool       = True
    resume_from     : str        = ""
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=list)
    logdir          : Path       = Path("/logs")
    gpus            : list[int]  = field(default_factory=lambda: [0, 1])
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0
    infer_after     : bool       = False
    training        : SimpleNamespace = field(default_factory=lambda: SimpleNamespace(resume=False))
    inference       : SimpleNamespace = field(default_factory=lambda: SimpleNamespace(split="test"))


def _done(jobs: list) -> list[dict]:
    return [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]


def _complete_seed(base: Path, seed: int, inference: bool = False, split: str = "test") -> Path:
    run = base / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(run, {})

    if inference:
        stamp = run / "inference" / "20260101_000000"
        stamp.mkdir(parents=True)
        CompletionMarker.stamp(stamp, {"split": split})

    return run


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


def test_fanout_resumes_from_the_sweep_base_directory(tmp_path):
    (tmp_path / "label_exp" / "seed3").mkdir(parents=True)

    config    = _Config(resume_from=str(tmp_path / "label_exp"), seeds=[3, 0, 17], logdir=Path("/elsewhere"))
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]))

    assert scheduler.base_name == "label_exp"
    assert scheduler.run_dir   == tmp_path / "label_exp"
    assert scheduler.resuming  is True
    assert config.logdir       == tmp_path
    assert config.resume       is True


def test_fanout_resumes_from_a_single_seed_directory(tmp_path):
    (tmp_path / "label_exp" / "seed3").mkdir(parents=True)

    config    = _Config(resume_from=str(tmp_path / "label_exp" / "seed3"), seeds=[0, 1], logdir=Path("/elsewhere"))
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]))

    assert scheduler.base_name == "label_exp"
    assert scheduler.run_dir   == tmp_path / "label_exp"


def test_fanout_resume_jobs_are_handed_their_seed_directory(tmp_path):
    (tmp_path / "other-label_exp" / "seed0").mkdir(parents=True)

    config    = _Config(resume_from=str(tmp_path / "other-label_exp"), seeds=[0, 1], logdir=Path("/elsewhere"))
    overrides = {"resume_from": str(tmp_path / "other-label_exp"), "training.epochs": "3"}
    scheduler = SeedFanoutScheduler.for_runner(config, overrides, Path("/entry/train.py"), _factory([]))

    for seed in (0, 1):
        argv = scheduler._job(f"other-label_exp/seed{seed}", seed).command

        assert argv[argv.index("--resume_from") + 1]     == str(tmp_path / "other-label_exp" / f"seed{seed}")
        assert argv[argv.index("--training.epochs") + 1] == "3"
        assert argv[argv.index("--seed") + 1]            == str(seed)
        assert "--logdir"   not in argv
        assert "--run_name" not in argv

    infer = scheduler._inference_job("other-label_exp/seed1", 1).command
    assert infer[infer.index("--resume_from") + 1] == str(tmp_path / "other-label_exp" / "seed1")
    assert infer[infer.index("--infer_after") + 1] == "true"


def test_fanout_resume_skips_completed_seeds(tmp_path):
    base = tmp_path / "label_exp"

    for seed in (0, 1, 2):
        (base / f"seed{seed}").mkdir(parents=True)
        CompletionMarker.stamp(base / f"seed{seed}", {})

    config    = _Config(resume_from=str(base), seeds=[0, 1, 2, 3, 4], logdir=Path("/elsewhere"))
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]))

    units, plan = scheduler._resume_units([(seed, SeedSet.run_name("exp", seed)) for seed in (0, 1, 2, 3, 4)])

    assert [seed for seed, _ in units] == [3, 4]
    assert plan.completed              == ["seed0", "seed1", "seed2"]
    assert plan.missing                == ["seed3", "seed4"]


def test_fresh_fanout_jobs_keep_naming_their_runs(tmp_path):
    scheduler = _scheduler(tmp_path)

    argv = scheduler._job("exp/seed3", 3).command

    assert "--resume_from" not in argv
    assert argv[argv.index("--logdir") + 1]   == str(tmp_path)
    assert argv[argv.index("--run_name") + 1] == "exp/seed3"


def test_run_records_the_sweep_configuration(tmp_path):
    import json

    scheduler = _scheduler(tmp_path)
    scheduler.stage._run_queue = lambda jobs: [{"name": job.name, "gpu": 0, "status": "DONE", "returncode": 0, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]
    scheduler.run()

    recorded = json.loads((scheduler.log_dir / "resolved_config.json").read_text())

    assert recorded["seeds"]       == [3, 0, 17]
    assert recorded["resume_from"] == ""


def test_fanout_resume_rejects_a_folder_without_seed_runs(tmp_path):
    config = _Config(resume_from=str(tmp_path), seeds=[0, 1], logdir=Path("/elsewhere"))

    with pytest.raises(SystemExit, match="neither a seed run directory"):
        SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]))


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


def test_resumed_batched_sweep_infers_over_the_complete_seeds_missing_inference(tmp_path):
    base = tmp_path / "label_exp"
    _complete_seed(base, 0, inference=True)
    _complete_seed(base, 1, inference=False)

    config    = _Config(resume_from=str(base), seeds=[0, 1], logdir=Path("/elsewhere"))
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]), infer_at_end=True)
    queues    = []

    def fake_queue(jobs):
        queues.append(jobs)
        return _done(jobs)

    scheduler.stage._run_queue = fake_queue
    scheduler.run()

    assert [[job.name for job in jobs] for jobs in queues] == [["label_exp/seed1"]]
    assert queues[0][0].command[queues[0][0].command.index("--infer_after") + 1] == "true"
    assert scheduler.infer_results_path.is_file()

    written = (scheduler.log_dir / "seed_sweep.log").read_text()

    assert "no training job launched" in written
    assert "1 seed(s) lack 'test' inference -> queued: label_exp/seed1" in written


def test_resumed_inline_inference_sweep_infers_over_the_reused_seeds_missing_it(tmp_path):
    base = tmp_path / "label_exp"
    _complete_seed(base, 0, inference=False)
    _complete_seed(base, 1, inference=True)

    config    = _Config(resume_from=str(base), seeds=[0, 1, 2], logdir=Path("/elsewhere"), infer_after=True)
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]), infer_after=True)
    queues    = []

    def fake_queue(jobs):
        queues.append(jobs)
        if len(queues) == 1:
            _complete_seed(base, 2, inference=True)
        return _done(jobs)

    scheduler.stage._run_queue = fake_queue
    scheduler.run()

    assert [[job.name for job in jobs] for jobs in queues] == [["label_exp/seed2"], ["label_exp/seed0"]]
    assert "--infer_after" not in queues[0][0].command
    assert scheduler.inference_split == "test"


def test_sweep_without_inference_never_runs_an_inference_pass(tmp_path):
    scheduler = _scheduler(tmp_path)
    queues    = []

    def fake_queue(jobs):
        queues.append(jobs)
        return _done(jobs)

    scheduler.stage._run_queue = fake_queue
    scheduler.run()

    assert scheduler.inference_split is None
    assert len(queues) == 1
    assert "none (the sweep infers nothing)" in (scheduler.log_dir / "seed_sweep.log").read_text()


def test_run_raises_when_a_seed_fails(tmp_path):
    scheduler = _scheduler(tmp_path)

    def fake_queue(jobs):
        return [{"name": job.name, "gpu": 0, "status": "FAILED", "returncode": 1, "duration_s": 60.0, "log_file": str(job.log_path)} for job in jobs]

    scheduler.stage._run_queue = fake_queue

    with pytest.raises(SystemExit, match="3 of 3 seed runs failed"):
        scheduler.run()

    written = (scheduler.log_dir / "seed_sweep.log").read_text()

    assert ">>> ABORTED seed_sweep" in written
    assert "[Aborted] SystemExit: 3 of 3 seed runs failed" in written


def test_fresh_launch_into_a_root_holding_seed_runs_warns(tmp_path):
    _complete_seed(tmp_path / "label_exp", 3)

    scheduler = _scheduler(tmp_path)
    scheduler.stage._run_queue = _done
    scheduler.run()

    written = (scheduler.log_dir / "seed_sweep.log").read_text()

    assert f"{scheduler.run_dir} already holds runs of 1 planned unit(s)" in written
    assert "1 completed unit(s) are reused without training" in written


def test_results_file_of_an_earlier_launch_is_merged_not_replaced(tmp_path):
    import json

    base = tmp_path / "label_exp"
    _complete_seed(base, 0)
    (base / "seed1").mkdir(parents=True)

    config    = _Config(resume_from=str(base), seeds=[0, 1], logdir=Path("/elsewhere"))
    scheduler = SeedFanoutScheduler.for_runner(config, {}, Path("/entry/train.py"), _factory([]))

    json.dump([{"name": "label_exp/seed0", "status": "DONE", "duration_s": 60.0}, {"name": "label_exp/seed1", "status": "FAILED", "duration_s": 5.0}], open(scheduler.results_path, "w"))

    scheduler.stage._run_queue = _done
    scheduler.run()

    results = json.loads(scheduler.results_path.read_text())

    assert [record["name"] for record in results] == ["label_exp/seed0", "label_exp/seed1"]
    assert [record["status"] for record in results] == ["DONE", "DONE"]
    assert "seed_sweep_results.json already holds 2 entry(ies)" in (scheduler.log_dir / "seed_sweep.log").read_text()


@dataclass
class _LauncherConfig:
    run_name        : str | None = None
    resume_from     : str        = ""
    seed            : int        = 0
    seeds           : list[int]  = field(default_factory=list)
    logdir          : Path       = Path("/logs")
    gpus            : list[int]  = field(default_factory=lambda: [0, 1])
    ddp             : bool       = False
    gpus_file       : str        = ""
    poll_interval_s : float      = 5.0
    infer_after     : bool       = False
    infer_at_end    : bool       = False


class _FakeScheduler:
    captured = {}

    @classmethod
    def for_runner(cls, config, cli_overrides, entry_script, runner_factory, base_label=None, infer_at_end=False, infer_after=False):
        cls.captured = {"infer_at_end": infer_at_end, "infer_after": infer_after}
        return cls()

    def run(self):
        pass


def _launcher(supports_infer_at_end: bool):
    return launcher_mod.SeedSweepLauncher(_LauncherConfig(seeds=[0, 1]), _factory([]), "test sweep", entry_script=Path("/entry/train.py"), supports_infer_at_end=supports_infer_at_end)


def test_launcher_rejects_infer_after_together_with_infer_at_end():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _launcher(True).run(argv=["--infer_after", "True", "--infer_at_end", "True"])


def test_launcher_forwards_infer_at_end_to_the_fanout(monkeypatch):
    monkeypatch.setattr(launcher_mod, "SeedFanoutScheduler", _FakeScheduler)

    _launcher(True).run(argv=["--infer_at_end", "True"])

    assert _FakeScheduler.captured == {"infer_at_end": True, "infer_after": False}


def test_launcher_forwards_infer_after_to_the_fanout(monkeypatch):
    monkeypatch.setattr(launcher_mod, "SeedFanoutScheduler", _FakeScheduler)

    _launcher(True).run(argv=["--infer_after", "True"])

    assert _FakeScheduler.captured == {"infer_at_end": False, "infer_after": True}


def test_launcher_without_support_never_batches_inference(monkeypatch):
    monkeypatch.setattr(launcher_mod, "SeedFanoutScheduler", _FakeScheduler)

    _launcher(False).run(argv=["--infer_at_end", "True"])

    assert _FakeScheduler.captured == {"infer_at_end": False, "infer_after": False}
