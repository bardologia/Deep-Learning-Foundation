from __future__ import annotations

from pathlib import Path

import pytest

from catalog_roots import CatalogRoots, RunScanner


@pytest.fixture
def roots():
    return CatalogRoots()


def test_an_empty_path_reports_the_caller_message(roots):
    target, error = roots.resolve("   ")

    assert target is None
    assert error  == CatalogRoots.NOT_SET

    target, error = roots.resolve("", "not set")

    assert target is None
    assert error  == "not set"


def test_a_relative_path_is_refused(roots, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()

    target, error = roots.resolve("runs")

    assert target is None
    assert error  == "an absolute path is required"


def test_a_file_or_a_missing_directory_is_refused(roots, tmp_path):
    (tmp_path / "note.md").write_text("x")

    assert roots.resolve(str(tmp_path / "note.md"))[1].startswith("not a directory")
    assert roots.resolve(str(tmp_path / "nowhere"))[1].startswith("not a directory")


def test_resolution_follows_symlinks_and_expands_the_home_marker(roots, tmp_path, monkeypatch):
    real = tmp_path / "real_runs"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)

    target, error = roots.resolve(str(tmp_path / "link"))

    assert error == ""
    assert target == real.resolve()

    monkeypatch.setenv("HOME", str(tmp_path))
    assert roots.resolve("~/real_runs")[0] == real.resolve()


def test_a_refused_path_is_never_recorded(roots, tmp_path):
    roots.open(str(tmp_path / "nowhere"))

    assert roots.snapshot() == ()


def test_open_records_the_resolved_root(roots, tmp_path):
    target, error = roots.open(str(tmp_path) + "/")

    assert error == ""
    assert roots.snapshot() == (str(target),)
    assert roots.known(str(target)) is True
    assert roots.known(str(tmp_path) + "/") is False


def test_containment_covers_nested_paths_only(roots, tmp_path):
    runs = tmp_path / "runs"
    (runs / "group" / "run_a").mkdir(parents=True)
    (tmp_path / "runs_backup").mkdir()

    roots.open(str(runs))

    assert roots.contains(runs / "group" / "run_a") is True
    assert roots.contains(runs)                     is True
    assert roots.contains(tmp_path / "runs_backup") is False
    assert roots.contains(tmp_path)                 is False


def test_containment_rejects_a_shared_name_prefix(roots, tmp_path):
    (tmp_path / "runs").mkdir()
    roots.open(str(tmp_path / "runs"))

    assert roots.contains(Path(str(tmp_path / "runs") + "_backup") / "leak.png") is False


def test_the_enclosing_root_is_the_deepest_match(roots, tmp_path):
    outer = tmp_path / "runs"
    inner = outer / "group"
    inner.mkdir(parents=True)

    roots.open(str(outer))
    roots.open(str(inner))

    assert roots.enclosing(inner / "run_a") == inner.resolve()
    assert roots.enclosing(outer / "run_b") == outer.resolve()
    assert roots.enclosing(tmp_path / "elsewhere") is None


def test_the_same_root_is_recorded_once(roots, tmp_path):
    roots.open(str(tmp_path))
    roots.open(str(tmp_path) + "/")
    roots.open(str(tmp_path / "." ))

    assert len(roots.snapshot()) == 1


def _fake_stamp(root: Path, group: str, run: str, stamp: str, extras: tuple = ()) -> Path:
    stamp_dir = root / group / run / "inference" / stamp
    cubes     = stamp_dir / "cubes"
    cubes.mkdir(parents=True)

    (cubes / "pred_curves.npy").write_bytes(b"x")
    for rel in extras:
        target = stamp_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    return stamp_dir


def _fake_checkpoint_run(root: Path, group: str, run: str, with_config: bool = True) -> Path:
    run_dir = root / group / run
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(b"x")

    if with_config:
        (run_dir / "meta").mkdir()
        (run_dir / "meta" / "model_config.json").write_text("{}")

    return run_dir


def test_stamp_scan_groups_and_sorts_newest_first(tmp_path):
    old  = _fake_stamp(tmp_path, "backbone", "run_a", "2026-01-01_00-00-00")
    new  = _fake_stamp(tmp_path, "backbone", "run_a", "2026-02-01_00-00-00")
    solo = _fake_stamp(tmp_path, ".", "run_b", "2026-01-15_00-00-00")

    out = RunScanner(CatalogRoots()).stamps(str(tmp_path))

    assert out["ok"] is True
    assert [entry["id"] for entry in out["entries"]] == sorted([str(old), str(new), str(solo)], reverse=True)
    assert {entry["group"] for entry in out["entries"]} == {"backbone", "."}
    assert all(entry["stamp"] for entry in out["entries"])


def test_stamp_scan_drops_entries_missing_required_files(tmp_path):
    complete = _fake_stamp(tmp_path, "g", "full", "s1", extras=("metrics.json", "cubes/pixel_mse.npy"))
    _fake_stamp(tmp_path, "g", "bare", "s2")

    out = RunScanner(CatalogRoots()).stamps(str(tmp_path), required=("metrics.json", "cubes/pixel_mse.npy"))

    assert [entry["id"] for entry in out["entries"]] == [str(complete)]


def test_stamp_scan_reports_root_errors(tmp_path):
    out = RunScanner(CatalogRoots()).stamps(str(tmp_path / "nowhere"))

    assert out["ok"] is False
    assert out["entries"] == []


def test_checkpoint_scan_requires_the_persisted_config(tmp_path):
    good = _fake_checkpoint_run(tmp_path, "backbone", "run_ok")
    _fake_checkpoint_run(tmp_path, "backbone", "run_bare", with_config=False)

    out = RunScanner(CatalogRoots()).checkpoint_runs(str(tmp_path), "best_model.pt", "model_config.json")

    assert out["ok"] is True
    assert [entry["id"] for entry in out["entries"]] == [str(good)]
    assert out["entries"][0]["group"] == "backbone"
    assert out["entries"][0]["stamp"] == ""
