from __future__ import annotations

from pathlib import Path

from configuration.patch_sweep     import PatchSweepConfig
from pipelines.patch_sweep.planner import PatchSweepPlanner
from pipelines.patch_sweep.stages  import SweepTrainingStage
from tools.monitoring.logger       import Logger
from tools.runtime.completion      import CompletionMarker


def make_logger(tmp_path: Path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="stages_test")


def make_base(tmp_path: Path, names: list[str]) -> Path:
    base = tmp_path / "datasets"
    for name in names:
        (base / name / "data").mkdir(parents=True)
    return base


def make_stage(tmp_path: Path, resume: bool = False, seeds: list[int] | None = None) -> SweepTrainingStage:
    config                    = PatchSweepConfig()
    config.dataset_base_path  = make_base(tmp_path, ["w20_10", "w20_20"])
    config.dataset_filter     = []
    config.patch.maximum      = (32, 16)
    config.paths.log_base_dir = tmp_path
    config.resume             = resume
    config.seeds              = seeds or []

    planner = PatchSweepPlanner(config)

    return SweepTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))


def test_stage_builds_one_item_per_unit(tmp_path):
    stage = make_stage(tmp_path)

    assert len(stage.items) == 16
    assert stage.items[0] == "w20_10-p008x008"


def test_job_carries_the_unit_and_run_context(tmp_path):
    stage = make_stage(tmp_path)
    job   = stage._job("w20_10-p032x016")

    assert "--worker" in job.command and "train" in job.command
    assert "--unit"   in job.command and "w20_10-p032x016" in job.command
    assert "--run-tag" in job.command and "rt" in job.command


def mark_complete(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "test"})


def test_resume_reuses_completed_units(tmp_path):
    stage = make_stage(tmp_path, resume=True)

    mark_complete(stage.stage_dir / "w20_10-p016x008")

    assert stage._is_complete("w20_10-p016x008")
    assert not stage._is_complete("w20_10-p032x008")


def test_resume_ignores_units_with_only_a_checkpoint(tmp_path):
    stage = make_stage(tmp_path, resume=True)

    unit_dir = stage.stage_dir / "w20_10-p016x008"
    unit_dir.mkdir(parents=True)
    (unit_dir / "best_model.pt").write_text("x")

    assert not stage._is_complete("w20_10-p016x008")


def test_resume_off_ignores_completion_markers(tmp_path):
    stage = make_stage(tmp_path, resume=False)

    mark_complete(stage.stage_dir / "w20_10-p016x008")

    assert not stage._is_complete("w20_10-p016x008")


def test_stage_expands_units_by_seed(tmp_path):
    stage = make_stage(tmp_path, seeds=[0, 1])

    assert len(stage.items) == 32
    assert stage.items[0] == "w20_10-p008x008/seed0"


def test_seeded_job_carries_the_unit_base_and_seed(tmp_path):
    stage = make_stage(tmp_path, seeds=[0, 1])
    job   = stage._job("w20_10-p032x016/seed1")

    assert "--unit" in job.command and "w20_10-p032x016" in job.command
    assert "--seed" in job.command and "1" in job.command
    assert "w20_10-p032x016/seed1" not in job.command


def test_resume_sees_seed_run_completion(tmp_path):
    stage = make_stage(tmp_path, resume=True, seeds=[0, 1])

    mark_complete(stage.stage_dir / "w20_10-p016x008" / "seed1")

    assert stage._is_complete("w20_10-p016x008/seed1")
    assert not stage._is_complete("w20_10-p016x008/seed0")
