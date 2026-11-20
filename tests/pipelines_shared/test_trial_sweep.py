
from __future__ import annotations

from pathlib import Path

import pytest

from configuration.training.backbone      import BackboneEntryConfig
from pipelines.backbone.training.launcher import BackboneTrainingLauncher, TrainScheduler
from tools.data.io                        import FileIO
from tools.runtime.completion             import CompletionMarker


ENTRY = Path("main/training/train_backbone.py")


class FakeStage:

    def __init__(self, statuses: dict | None = None) -> None:
        self.statuses  = statuses or {}
        self.pool_file = Path("gpus.json")
        self.submitted = []
        self.written   = {}

    def _run_queue(self, jobs: list) -> list[dict]:
        names = [job.name for job in jobs]
        self.submitted.append(names)

        return [{"name": name, "status": self.statuses.get(name, "DONE"), "duration_s": 60.0} for name in names]

    def _order_results(self, results: list[dict], names: list[str]) -> list[dict]:
        return results

    def _write_results(self, results: list[dict], path: Path) -> None:
        self.written[Path(path).name] = results

    def _log_failures(self, failed: list[dict], name_key: str = "name") -> None:
        pass


def _scheduler(logdir: Path, mode: str = "ema", **overrides):
    argv = ["--trials_enabled", "true", "--trials_mode", mode, "--seeds", "[0]", "--logdir", str(logdir)]

    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]

    return BackboneTrainingLauncher(entry_script=ENTRY).build_scheduler(argv)


def _complete_run(root: Path, name: str, inference: bool = False, split: str = "test") -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(run, {})

    if inference:
        stamp = run / "inference" / "20260101_000000"
        stamp.mkdir(parents=True)
        CompletionMarker.stamp(stamp, {"split": split})

    return run


def test_run_directories_stay_under_the_mode_subdirectory(tmp_path):
    assert _scheduler(tmp_path).runs_root == tmp_path / "ema"


def test_planned_units_keep_the_units_a_resume_reuses(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    _complete_run(root, names[0])
    (root / names[1]).mkdir(parents=True)

    scheduler = _scheduler(tmp_path, resume_from=root)

    assert [name for name, _ in scheduler.planned_units()] == names
    assert [name for name, _ in scheduler.units()]         == names[1:]


def test_resumed_sweep_infers_over_the_reused_units_missing_inference(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    _complete_run(root, names[0], inference=True)
    for name in names[1:]:
        _complete_run(root, name)

    scheduler        = _scheduler(tmp_path, resume_from=root, infer_at_end="true")
    scheduler._stage = FakeStage()
    reports          = []

    scheduler._write_mode_report = lambda: reports.append(True)

    scheduler.run()

    assert scheduler.units()          == []
    assert scheduler._stage.submitted == [names[1:]]
    assert reports                    == [True]
    assert "already trained" in (scheduler.log_dir / "train_scheduler.log").read_text()


def test_resumed_sweep_without_pending_inference_launches_nothing(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    for name in names:
        _complete_run(root, name, inference=True)

    scheduler        = _scheduler(tmp_path, resume_from=root, infer_at_end="true")
    scheduler._stage = FakeStage()

    scheduler.run()

    assert scheduler._stage.submitted == []


def test_resumed_inline_inference_sweep_infers_over_the_reused_units_missing_it(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    for name in names:
        _complete_run(root, name, inference=(name != names[0]))

    scheduler        = _scheduler(tmp_path, resume_from=root, infer_after="true")
    scheduler._stage = FakeStage()

    scheduler.run()

    assert scheduler.unit_inference   is True
    assert scheduler.inference_split  == "test"
    assert scheduler._stage.submitted == [[names[0]]]

    written = (scheduler.log_dir / "train_scheduler.log").read_text()

    assert f"1 unit(s) lack 'test' inference -> queued: {names[0]}" in written
    assert "Trainer resume" in written and "per unit: continues from last.pt where it exists" in written


def test_sweep_without_inference_reports_no_split(tmp_path):
    scheduler        = _scheduler(tmp_path)
    scheduler._stage = FakeStage()

    scheduler.run()

    assert scheduler.inference_split is None
    assert len(scheduler._stage.submitted) == 1
    assert "Inference split" in (scheduler.log_dir / "train_scheduler.log").read_text()


def test_inference_is_pending_when_the_stamp_records_another_split(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    for name in names:
        _complete_run(root, name, inference=True, split="val")

    scheduler        = _scheduler(tmp_path, resume_from=root, infer_at_end="true")
    scheduler._stage = FakeStage()

    scheduler.run()

    assert scheduler._stage.submitted == [names]


def test_inference_covers_freshly_trained_units_and_skips_failed_ones(tmp_path):
    names            = [name for name, _ in _scheduler(tmp_path).planned_units()]
    scheduler        = _scheduler(tmp_path, infer_at_end="true")
    scheduler._stage = FakeStage(statuses={names[1]: "FAILED"})

    with pytest.raises(SystemExit, match="training trials failed"):
        scheduler.run()

    assert scheduler._stage.submitted == [names, [name for name in names if name != names[1]]]

    written = (scheduler.log_dir / "train_scheduler.log").read_text()

    assert ">>> ABORTED train_scheduler" in written
    assert "[Aborted] SystemExit: 1 of" in written


def test_fresh_launch_into_a_root_holding_runs_warns_before_launching(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]
    _complete_run(root, names[0])

    scheduler        = _scheduler(tmp_path)
    scheduler._stage = FakeStage()

    scheduler.run()

    written = (scheduler.log_dir / "train_scheduler.log").read_text()

    assert scheduler._stage.submitted == [names]
    assert f"{root} already holds runs of 1 planned unit(s)" in written
    assert "1 completed unit(s) are reused without training" in written


def test_results_file_of_an_earlier_launch_is_merged_not_replaced(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    _complete_run(root, names[0])
    (root / names[1]).mkdir(parents=True)
    FileIO.save_json([{"name": names[0], "status": "DONE", "duration_s": 60.0}, {"name": names[1], "status": "FAILED", "duration_s": 5.0}], root / "batch_train_logs" / "train_scheduler_results.json")

    scheduler        = _scheduler(tmp_path, resume_from=root)
    scheduler._stage = FakeStage()

    scheduler.run()

    written = scheduler._stage.written["train_scheduler_results.json"]

    assert [record["name"] for record in written][:2] == names[:2]
    assert written[0]["status"] == "DONE" and written[0]["duration_s"] == 60.0
    assert written[1]["status"] == "DONE" and written[1]["duration_s"] == 60.0
    assert "train_scheduler_results.json already holds 2 entry(ies)" in (scheduler.log_dir / "train_scheduler.log").read_text()


def test_run_directories_carrying_a_label_prefix_count_as_inferred(tmp_path):
    root  = tmp_path / "ema"
    names = [name for name, _ in _scheduler(tmp_path).planned_units()]

    for name in names:
        _complete_run(root, f"resunet-conv_{name}", inference=True)

    scheduler = _scheduler(tmp_path, resume_from=root, infer_at_end="true")

    assert all(not scheduler._inference_pending(name) for name in names)


def test_trial_argv_pins_the_unit_and_drops_the_scheduler_fields(tmp_path):
    scheduler       = _scheduler(tmp_path, gpus="[0, 1]", poll_interval_s="1.0", **{"training.epochs": "2"})
    run_name, extra = scheduler.units()[0]
    argv            = scheduler.trial_argv(run_name, extra)

    assert argv[argv.index("--run_name") + 1]        == run_name
    assert argv[argv.index("--logdir") + 1]          == str(scheduler.runs_root)
    assert argv[argv.index("--training.epochs") + 1] == "2"

    assert "--trials_enabled"    not in argv
    assert "--trials_mode"       not in argv
    assert "--ema_trials.decays" not in argv
    assert "--gpus"              not in argv
    assert "--poll_interval_s"   not in argv


def test_seed_expanded_units_override_the_forwarded_seed_list(tmp_path):
    scheduler       = _scheduler(tmp_path, seeds="[3, 7]")
    run_name, extra = scheduler.units()[0]
    argv            = scheduler.trial_argv(run_name, extra)

    assert run_name.endswith("/seed3")
    assert argv[argv.index("--seed") + 1]  == "3"
    assert argv[argv.index("--seeds") + 1] == "[3]"


def test_scheduler_fields_name_real_entry_config_fields():
    config  = BackboneEntryConfig()
    unknown = [field for field in TrainScheduler.SCHEDULER_FIELDS if not hasattr(config, field)]

    assert unknown == []


def test_every_trial_mode_claims_its_planner_settings():
    config = BackboneEntryConfig()

    for mode in TrainScheduler.MODE_SUBDIRS:
        field = f"{mode}_trials"

        if hasattr(config, field):
            assert field in TrainScheduler.SCHEDULER_FIELDS
