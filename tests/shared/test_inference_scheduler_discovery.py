from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

from pipelines.shared.inference.inference_scheduler import InferenceScheduler
from pipelines.shared.inference.run_classifier      import RunArtifacts, RunType

from tests.conftest import SilentLogger


def _make_run(directory: Path, config_name: str) -> None:
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / config_name).write_text("{}")


def _scheduler(runs: Path, run_filter, run_type: str, gpus_file: str = "") -> InferenceScheduler:
    config = SimpleNamespace(runs_dir=str(runs), run_filter=run_filter, gpus=[0], gpus_file=gpus_file, poll_interval_s=1)
    return InferenceScheduler(config, Path("main/x.py"), run_type)


def test_no_filter_discovers_runs_at_any_depth_and_respects_type(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs / "run_top", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "group_a" / "run_a1", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "group_a" / "deep" / "run_a2", RunArtifacts.BACKBONE_CONFIG)
    _make_run(runs / "group_b" / "ae_run", RunArtifacts.PROFILE_AE_CONFIG)
    (runs / "empty").mkdir(parents=True)

    scheduler = _scheduler(runs, [], RunType.BACKBONE)
    matched   = sorted(str(directory.relative_to(runs)) for directory in scheduler._run_dirs(SilentLogger()))

    assert matched == ["group_a/deep/run_a2", "group_a/run_a1", "run_top"]


def test_filter_resolves_nested_relative_names(tmp_path):
    runs = tmp_path / "runs"
    _make_run(runs / "group_a" / "run_a1", RunArtifacts.BACKBONE_CONFIG)

    scheduler = _scheduler(runs, ["group_a/run_a1"], RunType.BACKBONE)
    selected  = [str(directory.relative_to(runs)) for directory in scheduler._candidate_dirs(SilentLogger())]

    assert selected == ["group_a/run_a1"]


def test_pool_file_defaults_into_the_work_directory(tmp_path):
    scheduler = _scheduler(tmp_path / "runs", [], RunType.BACKBONE)

    assert scheduler.pool_file == scheduler.work_dir / "gpu_pool.json"


def test_pool_file_honors_configured_gpus_file(tmp_path):
    configured = tmp_path / "pools" / "job42.json"
    scheduler  = _scheduler(tmp_path / "runs", [], RunType.BACKBONE, gpus_file=str(configured))

    assert scheduler.pool_file == configured
