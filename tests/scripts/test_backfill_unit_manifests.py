"""Tests the backfill that writes unit manifests onto pre-manifest experiment runs.

Covers the classification of a run as cross-validation or benchmark, the payload
written for each planned unit, the units left alone because they already carry a
manifest or never reached disk, the directories no planned unit accounts for, and
the dry run that writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT   = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_unit_manifests

from configuration.benchmark                      import BenchmarkConfig
from configuration.cross_validation               import CrossValidationConfig
from configuration.training.general.run           import DEFAULT_PARAMS_RUN
from pipelines.benchmark                          import stages as benchmark_stages
from pipelines.cross_validation                   import folds
from pipelines.shared.orchestration.unit_manifest import UnitManifest
from tools.runtime.config_cli                     import ConfigCli

from tests.conftest import RecordingLogger


@pytest.fixture(autouse=True)
def fixed_gaussians(monkeypatch):
    """Pins both unit namings to two Gaussians so the composed names stay stable without a dataset."""
    monkeypatch.setattr(folds.FoldRunNaming, "_n_gaussians", lambda self: 2)
    monkeypatch.setattr(benchmark_stages.SeedExpandedStage, "n_gaussians", staticmethod(lambda config: 2))


def _cross_validation_run(root: Path, test_data_dir: Path, run_tag: str = "cv_run") -> Path:
    """Writes a cross-validation run recording three folds over two seeds against the test dataset crop."""
    config                       = CrossValidationConfig()
    config.run_tag               = run_tag
    config.paths.log_base_dir    = root
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    config.folds.n_folds         = 3
    config.folds.row_start       = 1000
    config.folds.row_end         = 1512
    config.seeds                 = [1, 2]

    run_dir = root / run_tag
    ConfigCli.save_resolved(config, run_dir / "pipeline" / "resolved_config.json")
    (run_dir / "folds").mkdir(parents=True)

    return run_dir


def _benchmark_run(root: Path, run_tag: str = "bench_run") -> Path:
    """Writes a benchmark run recording one seed and returns its directory."""
    config                    = BenchmarkConfig()
    config.run_tag            = run_tag
    config.paths.log_base_dir = root
    config.seeds              = [0]

    run_dir = root / run_tag
    ConfigCli.save_resolved(config, run_dir / "pipeline" / "resolved_config.json")
    (run_dir / "training").mkdir(parents=True)

    return run_dir


def _plant(stage_dir: Path, name: str) -> Path:
    """Creates a run directory under a stage directory without a manifest and returns it."""
    unit = stage_dir / name
    (unit / "meta").mkdir(parents=True)
    return unit


def _backfiller(run_dir: Path, dry_run: bool = False):
    """Returns a backfiller bound to a run directory and a recording logger."""
    return backfill_unit_manifests.UnitManifestBackfiller(run_dir, RecordingLogger(), dry_run=dry_run)


def test_a_fold_unit_gets_the_identity_and_regions_of_its_plan(tmp_path, test_data_dir):
    """A fold directory named the way the recording composes it receives its fold index, seed and scored regions."""
    run_dir = _cross_validation_run(tmp_path, test_data_dir)
    plan    = backfill_unit_manifests.FoldManifestPlan(run_dir)

    name, _ = plan.units()[0]
    unit    = _plant(run_dir / "folds", name)

    assert _backfiller(run_dir).run() == 0

    manifest = UnitManifest.read(unit)

    assert manifest["kind"]          == "cross_validation"
    assert manifest["fold_index"]    == 0
    assert manifest["seed"]          == 1
    assert manifest["run_tag"]       == run_dir.name
    assert sorted(manifest["inference_regions"]) == ["test", "val"]


def test_a_backfilled_fold_is_found_again_under_any_directory_name(tmp_path, test_data_dir):
    """After the backfill the unit is identified by its manifest, so renaming its directory keeps it."""
    from pipelines.shared.orchestration.unit_manifest import UnitLedger

    run_dir = _cross_validation_run(tmp_path, test_data_dir)
    plan    = backfill_unit_manifests.FoldManifestPlan(run_dir)
    name, _ = plan.units()[0]
    unit    = _plant(run_dir / "folds", name)

    _backfiller(run_dir).run()

    renamed = unit.parent.parent / "renamed" / unit.name
    renamed.parent.mkdir()
    unit.rename(renamed)

    ledger = UnitLedger(run_dir / "folds", folds.FoldUnitManifest.KIND, folds.FoldUnitManifest.identity).build()

    assert ledger.directory((0, 1)) == renamed


def test_a_benchmark_unit_gets_its_model_component_and_seed(tmp_path):
    """A benchmark unit directory receives the model key, loss component and seed of its plan."""
    run_dir = _benchmark_run(tmp_path)
    plan    = backfill_unit_manifests.BenchmarkManifestPlan(run_dir)

    name, payload = plan.units()[0]
    unit          = _plant(run_dir / "training", name)

    assert _backfiller(run_dir).run() == 0

    manifest = UnitManifest.read(unit)

    assert manifest["kind"]           == "benchmark"
    assert manifest["model"]          == payload["model"]
    assert manifest["loss_component"] == payload["loss_component"]
    assert manifest["seed"]           == payload["seed"]


def test_units_already_carrying_a_manifest_are_left_alone(tmp_path, test_data_dir):
    """A unit written by a worker of this version is counted as kept and never rewritten."""
    run_dir = _cross_validation_run(tmp_path, test_data_dir)
    plan    = backfill_unit_manifests.FoldManifestPlan(run_dir)

    name, payload = plan.units()[0]
    unit          = _plant(run_dir / "folds", name)
    plan.write(unit, payload)

    written_at  = UnitManifest.read(unit)["written_at"]
    backfiller  = _backfiller(run_dir)
    backfiller.run()

    assert backfiller.written                       == []
    assert backfiller.kept                          == [name]
    assert UnitManifest.read(unit)["written_at"]    == written_at


def test_a_directory_no_planned_unit_accounts_for_is_reported(tmp_path, test_data_dir):
    """A renamed or foreign run directory is left without a manifest and makes the backfill exit non-zero."""
    run_dir = _cross_validation_run(tmp_path, test_data_dir)
    stranger = _plant(run_dir / "folds", "left_over_from_august")

    backfiller = _backfiller(run_dir)

    assert backfiller.run()             == 1
    assert not UnitManifest.exists(stranger)


def test_planned_units_without_a_directory_are_counted_not_created(tmp_path, test_data_dir):
    """Units the run never reached are reported as absent and no directory is made for them."""
    run_dir    = _cross_validation_run(tmp_path, test_data_dir)
    backfiller = _backfiller(run_dir)

    backfiller.run()

    assert len(backfiller.absent)                  == 6
    assert list((run_dir / "folds").iterdir())     == []


def test_a_dry_run_writes_nothing(tmp_path, test_data_dir):
    """A dry run reports the unit it would write without touching the run directory."""
    run_dir = _cross_validation_run(tmp_path, test_data_dir)
    plan    = backfill_unit_manifests.FoldManifestPlan(run_dir)
    name, _ = plan.units()[0]
    unit    = _plant(run_dir / "folds", name)

    backfiller = _backfiller(run_dir, dry_run=True)
    backfiller.run()

    assert backfiller.written == [unit]
    assert not UnitManifest.exists(unit)


def test_the_path_may_be_anywhere_inside_the_run(tmp_path, test_data_dir):
    """A path inside the run resolves to the run directory the recording sits in."""
    run_dir = _cross_validation_run(tmp_path, test_data_dir)
    plan    = backfill_unit_manifests.FoldManifestPlan(run_dir)
    name, _ = plan.units()[0]
    unit    = _plant(run_dir / "folds", name)

    assert _backfiller(unit).run() == 0
    assert UnitManifest.exists(unit)


def test_a_run_that_is_neither_kind_is_refused(tmp_path):
    """A staged run holding neither a folds nor a training directory exits with a message naming both."""
    config                    = CrossValidationConfig()
    config.paths.log_base_dir = tmp_path
    run_dir                   = tmp_path / "odd_run"
    ConfigCli.save_resolved(config, run_dir / "pipeline" / "resolved_config.json")

    with pytest.raises(SystemExit, match="neither a cross-validation nor a benchmark run"):
        _backfiller(run_dir).run()
