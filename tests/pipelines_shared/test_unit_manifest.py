"""Tests for the unit manifest a fan-out worker writes and the ledger that indexes manifests by identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.shared.orchestration.unit_manifest import UnitLedger, UnitManifest


def _unit(root: Path, name: str, kind: str = "cross_validation", **identity) -> Path:
    """Writes a unit run directory carrying a manifest of the given kind and identity fields."""
    directory = root / name
    UnitManifest.write(directory, kind, identity)
    return directory


def _identity(manifest: dict) -> tuple:
    """Returns the (fold index, seed) identity of a manifest."""
    return (manifest["fold_index"], manifest["seed"])


def test_manifest_round_trips_its_payload_with_kind_and_timestamp(tmp_path):
    """The manifest keeps every payload field, records the kind and stamps the writing time."""
    path = UnitManifest.write(tmp_path / "unit", "benchmark", {"model": "unet-conv", "seed": 3})

    assert path == tmp_path / "unit" / "meta" / "unit.json"
    assert UnitManifest.exists(tmp_path / "unit")

    manifest = UnitManifest.read(tmp_path / "unit")
    assert manifest["kind"]  == "benchmark"
    assert manifest["model"] == "unet-conv"
    assert manifest["seed"]  == 3
    assert manifest["written_at"]


def test_reading_a_run_without_a_manifest_raises(tmp_path):
    """A run directory without a manifest raises instead of yielding an empty identity."""
    (tmp_path / "run" / "meta").mkdir(parents=True)

    assert not UnitManifest.exists(tmp_path / "run")

    with pytest.raises(FileNotFoundError, match="No unit manifest"):
        UnitManifest.read(tmp_path / "run")


def test_ledger_indexes_units_by_identity_whatever_their_directory_name(tmp_path):
    """The ledger finds a unit through its manifest even when the directory name follows another scheme."""
    old = _unit(tmp_path, "old-label_fold_1", fold_index=1, seed=None)
    new = _unit(tmp_path, "new-label_fold_2/seed4", fold_index=2, seed=4)

    ledger = UnitLedger(tmp_path, "cross_validation", _identity).build()

    assert ledger.directory((1, None)) == old
    assert ledger.directory((2, 4))    == new
    assert ledger.directory((3, None)) is None
    assert ledger.unidentified         == []


def test_ledger_lists_runs_without_a_manifest_or_of_another_kind_as_unidentified(tmp_path):
    """Run directories lacking a manifest, or carrying one of another experiment, are reported and never matched."""
    (tmp_path / "stray_run" / "meta").mkdir(parents=True)
    _unit(tmp_path, "bench_unit", kind="benchmark", model="unet", seed=0)
    (tmp_path / "not_a_run").mkdir()

    ledger = UnitLedger(tmp_path, "cross_validation", _identity).build()

    assert ledger.by_identity == {}
    assert sorted(path.name for path in ledger.unidentified) == ["bench_unit", "stray_run"]


def test_ledger_refuses_two_directories_claiming_one_identity(tmp_path):
    """Two run directories recording the same identity make the unit ambiguous, so the ledger raises."""
    _unit(tmp_path, "first_fold_0", fold_index=0, seed=None)
    _unit(tmp_path, "second_fold_0", fold_index=0, seed=None)

    with pytest.raises(ValueError, match="same unit identity"):
        UnitLedger(tmp_path, "cross_validation", _identity).build()


def test_ledger_over_a_missing_stage_directory_is_empty(tmp_path):
    """A stage directory that does not exist yet yields an empty ledger instead of an error."""
    ledger = UnitLedger(tmp_path / "absent", "cross_validation", _identity).build()

    assert ledger.by_identity  == {}
    assert ledger.unidentified == []
