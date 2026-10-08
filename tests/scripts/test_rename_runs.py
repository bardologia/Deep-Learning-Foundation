"""Tests the bookkeeping of the run-renaming migration.

Covers the outcome totals logged per root and for the whole migration, the renamed
and patched counts of an applied group, and the abort record left behind when a
later root fails after earlier groups were processed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT   = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rename_runs

from rename_runs import RunRenamer

from tools.monitoring.logger import Logger


@pytest.fixture
def renamer(tmp_path, monkeypatch):
    """Returns a factory building a RunRenamer whose log file lands under tmp_path instead of the repository logs."""
    monkeypatch.setattr(rename_runs, "Logger", lambda log_dir, name: Logger(log_dir=str(tmp_path / "logs"), name=name))

    def build(roots: list[Path], apply: bool = False) -> RunRenamer:
        """Builds a renamer over the given roots.

        Args:
            roots : Directories to migrate.
            apply : Whether renames are performed rather than planned.

        Returns:
            The renamer, logging under tmp_path/logs/rename_runs.log.
        """
        return RunRenamer(roots, apply)

    return build


def _foreign_run(root: Path, name: str) -> Path:
    """Writes a run directory whose summary names a model the renamer does not migrate.

    Args:
        root : Directory holding the run.
        name : Run directory name.

    Returns:
        The run directory.
    """
    run = root / name
    (run / "meta").mkdir(parents=True)
    (run / "meta" / "run_summary.json").write_text(json.dumps({"model_name": "profile_ae", "run_name": name}))

    return run


def _log(tmp_path: Path) -> str:
    """Returns the text of the migration log written under tmp_path."""
    return (tmp_path / "logs" / "rename_runs.log").read_text(encoding="utf-8")


def test_totals_are_logged_per_root_and_for_the_whole_migration(tmp_path, renamer):
    """Verifies every root ends with its outcome table and the run ends with the overall totals and an END banner."""
    first  = tmp_path / "first"
    second = tmp_path / "second"
    _foreign_run(first, "pae_a")
    _foreign_run(first, "pae_b")
    _foreign_run(second, "pae_c")

    migration = renamer([first, second])
    migration.run()

    log = _log(tmp_path)

    assert migration.counts[RunRenamer.FOREIGN] == 3
    assert f"Totals for {first}" in log and f"Totals for {second}" in log
    assert log.count("foreign model") >= 3
    assert "> Totals\n" in log
    assert "planned (dry run)         : 0" in log
    assert "END rename_runs" in log


def test_an_applied_group_counts_renamed_runs_and_patched_files(tmp_path, renamer):
    """Verifies a performed rename is counted, and so is each metadata file whose content changed."""
    parent = tmp_path / "parent"
    (parent / "old_name" / "meta").mkdir(parents=True)
    (parent / "old_name" / "meta" / "run_summary.json").write_text(json.dumps({"run_name": "old_name"}))

    migration = renamer([parent], apply=True)
    migration._rename_runs(parent, {"old_name": "new_name"})
    migration.logger.close()

    assert (parent / "new_name").is_dir()
    assert json.loads((parent / "new_name" / "meta" / "run_summary.json").read_text()) == {"run_name": "new_name"}
    assert migration.counts[RunRenamer.RENAMED] == 1
    assert migration.counts[RunRenamer.PATCHED] == 1


def test_a_dry_run_counts_planned_renames_without_touching_anything(tmp_path, renamer):
    """Verifies a dry run counts the planned renames and leaves the directories and files alone."""
    parent = tmp_path / "parent"
    (parent / "old_name" / "meta").mkdir(parents=True)
    (parent / "old_name" / "meta" / "run_summary.json").write_text(json.dumps({"run_name": "old_name"}))

    migration = renamer([parent], apply=False)
    migration._rename_runs(parent, {"old_name": "new_name"})
    migration.logger.close()

    assert (parent / "old_name").is_dir()
    assert migration.counts[RunRenamer.PLANNED] == 1
    assert migration.counts[RunRenamer.RENAMED] == 0


def test_a_failing_root_leaves_the_totals_and_an_abort_record_in_the_log(tmp_path, renamer):
    """Verifies a root that raises after earlier roots were processed ends the log with the totals so far, the applied group count and an ABORTED banner."""
    first = tmp_path / "first"
    _foreign_run(first, "pae_a")

    with pytest.raises(FileNotFoundError, match="is not a directory"):
        renamer([first, tmp_path / "missing"]).run()

    log = _log(tmp_path)

    assert "Totals before the abort" in log
    assert "aborted after 0 group(s)" in log
    assert "ABORTED rename_runs" in log
    assert "END rename_runs" not in log
