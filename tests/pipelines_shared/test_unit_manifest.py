
from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.shared.orchestration.unit_manifest import UnidentifiedRuns, UnitLedger, UnitManifest, UnitResultsByRun
from tools.runtime.completion                     import CompletionMarker

from tests.conftest import RecordingLogger


def _unit(root: Path, name: str, kind: str = "cross_validation", **identity) -> Path:
    directory = root / name
    UnitManifest.write(directory, kind, identity)
    return directory


def _identity(manifest: dict) -> tuple:
    return (manifest["fold_index"], manifest["seed"])


def test_manifest_round_trips_its_payload_with_kind_and_timestamp(tmp_path):
    path = UnitManifest.write(tmp_path / "unit", "benchmark", {"model": "unet-conv", "seed": 3})

    assert path == tmp_path / "unit" / "meta" / "unit.json"
    assert UnitManifest.exists(tmp_path / "unit")

    manifest = UnitManifest.read(tmp_path / "unit")
    assert manifest["kind"]  == "benchmark"
    assert manifest["model"] == "unet-conv"
    assert manifest["seed"]  == 3
    assert manifest["written_at"]


def test_reading_a_run_without_a_manifest_raises(tmp_path):
    (tmp_path / "run" / "meta").mkdir(parents=True)

    assert not UnitManifest.exists(tmp_path / "run")

    with pytest.raises(FileNotFoundError):
        UnitManifest.read(tmp_path / "run")


def test_ledger_indexes_units_by_identity_whatever_their_directory_name(tmp_path):
    old = _unit(tmp_path, "old-label_fold_1", fold_index=1, seed=None)
    new = _unit(tmp_path, "new-label_fold_2/seed4", fold_index=2, seed=4)

    ledger = UnitLedger(tmp_path, "cross_validation", _identity).build()

    assert ledger.directory((1, None)) == old
    assert ledger.directory((2, 4))    == new
    assert ledger.directory((3, None)) is None
    assert ledger.unidentified         == []


def test_ledger_lists_runs_without_a_manifest_or_of_another_kind_as_unidentified(tmp_path):
    (tmp_path / "stray_run" / "meta").mkdir(parents=True)
    _unit(tmp_path, "bench_unit", kind="benchmark", model="unet", seed=0)
    (tmp_path / "not_a_run").mkdir()

    ledger = UnitLedger(tmp_path, "cross_validation", _identity).build()

    assert ledger.by_identity == {}
    assert sorted(path.name for path in ledger.unidentified) == ["bench_unit", "stray_run"]


def test_ledger_over_a_missing_stage_directory_is_empty(tmp_path):
    ledger = UnitLedger(tmp_path / "absent", "cross_validation", _identity).build()

    assert ledger.by_identity  == {}
    assert ledger.unidentified == []


def test_unidentified_report_says_what_the_stage_does_with_each_directory(tmp_path):
    reused = tmp_path / "unit_a"
    (reused / "meta").mkdir(parents=True)
    CompletionMarker.stamp(reused, {"stage": "training"})
    (tmp_path / "unit_b" / "meta").mkdir(parents=True)
    (tmp_path / "stray_run" / "meta").mkdir(parents=True)

    logger = RecordingLogger()
    report = UnidentifiedRuns(tmp_path, [reused, tmp_path / "unit_b", tmp_path / "stray_run"], "cross_validation", {"unit_a": reused, "unit_b": tmp_path / "unit_b", "unit_c": tmp_path / "unit_c"})

    report.log(logger, "FoldTrainingStage")

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert warnings[0].startswith("FoldTrainingStage: 3 run directory(ies) under")
    assert "no cross_validation manifest" in warnings[0]
    assert warnings[1] == "unit_a: sits at planned unit 'unit_a's own path and holds a completion marker, so the stage uses it as that unit's completed run"
    assert warnings[2] == "unit_b: sits at planned unit 'unit_b's own path without a completion marker, so the stage treats it as that unit's unfinished run"
    assert warnings[3] == "stray_run: claimed by no planned unit, ignored by the stage; the comparison still collects it beside the planned units"


def test_unidentified_report_is_silent_without_directories(tmp_path):
    logger = RecordingLogger()
    UnidentifiedRuns(tmp_path, [], "benchmark", {}).log(logger, "TrainingStage")

    assert logger.messages == []


class _StoredIndex:

    def __init__(self, path: Path, results: dict) -> None:
        self.path    = path
        self.results = results

    def build(self, runs: list) -> dict:
        return self.results


def test_unit_results_follow_their_worker_log_into_a_renamed_directory(tmp_path):
    stage_dir = tmp_path / "moved" / "run" / "training"
    elsewhere = Path("/elsewhere/scratch/run/training")
    results   = {
        "unet__mse/seed3" : {"name": "unet__mse/seed3", "status": "DONE", "duration_s": 7.0, "log_file": str(elsewhere / "old_name__mse" / "seed3" / "worker.log")},
        "unet__mse/seed4" : {"name": "unet__mse/seed4", "status": "FAILED", "duration_s": 2.0, "log_file": str(elsewhere / "unet__mse" / "seed4" / "worker.log")},
    }
    runs = [("old_name__mse/seed3", stage_dir / "old_name__mse" / "seed3"), ("unet__mse/seed4", stage_dir / "unet__mse" / "seed4")]

    logger = RecordingLogger()
    keyed  = UnitResultsByRun(stage_dir, logger).rekey(results, runs)

    assert keyed["old_name__mse/seed3"]["duration_s"] == 7.0
    assert keyed["unet__mse/seed4"]["status"]         == "FAILED"
    assert logger.messages == [("info", "1 unit result(s) matched to the renamed run directory holding their worker log: ['unet__mse/seed3 -> old_name__mse/seed3']")]


def test_unit_results_name_the_runs_and_results_that_match_nothing(tmp_path):
    stage_dir = tmp_path / "folds"
    results   = {"fold_0": {"name": "fold_0", "status": "DONE", "duration_s": 1.0, "log_file": str(stage_dir / "fold_9" / "worker.log")}}
    runs      = [("fold_0", stage_dir / "fold_0")]

    logger = RecordingLogger()
    keyed  = UnitResultsByRun(stage_dir, logger).rekey(results, runs)

    assert keyed == {}
    assert ("warning", "1 unit result(s) name a worker log outside every collected run directory, so their training status and duration reach no report: ['fold_0']") in logger.messages
    assert ("warning", "1 collected run(s) have no stage result, so the reports show their training status and duration as unknown: ['fold_0']") in logger.messages


def test_unit_results_rebuilt_from_markers_pass_through_unchanged(tmp_path):
    rebuilt = {"fold_0": {"name": "fold_0", "status": "DONE", "duration_s": None}}
    logger  = RecordingLogger()

    keyed = UnitResultsByRun(tmp_path / "folds", logger).build(_StoredIndex(tmp_path / "pipeline" / "training_results.json", rebuilt), [("fold_0", tmp_path / "folds" / "fold_0")])

    assert keyed           == rebuilt
    assert logger.messages == []
