from __future__ import annotations

import json
import math
from pathlib import Path
from types   import SimpleNamespace

import pytest

from pipelines.backbone.inference.analysis.seed_comparison import SeedComparison, SeedComparisonReport, SeedInferenceResolver

from tests.conftest import SilentLogger


def _seed_run(group: Path, seed_name: str, metrics: dict, stamp: str = "stamp") -> Path:
    run_dir = group / seed_name
    out_dir = run_dir / "inference" / stamp
    out_dir.mkdir(parents=True)

    (run_dir / "meta").mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out_dir / "report.md").write_text("# per-seed report", encoding="utf-8")

    return run_dir


def _report(group: Path, runs: list[Path], stamp: str = "stamp") -> SeedComparisonReport:
    return SeedComparisonReport(
        group_dir        = group,
        run_dirs         = runs,
        inference_dirs   = [run / "inference" / stamp for run in runs],
        output_subdir    = stamp,
        metrics_filename = "metrics.json",
        report_filename  = "report.md",
        logger           = SilentLogger(),
    )


def test_resolver_picks_the_explicit_inference_subdir(tmp_path):
    run = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")

    resolved = SeedInferenceResolver("20240101_000000", "metrics.json").resolve(run)

    assert resolved == run / "inference" / "20240101_000000"


def test_resolver_raises_when_the_explicit_subdir_has_no_metrics(tmp_path):
    run = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")

    with pytest.raises(FileNotFoundError):
        SeedInferenceResolver("20240202_000000", "metrics.json").resolve(run)


def test_resolver_picks_the_latest_inference_with_metrics(tmp_path):
    run = _seed_run(tmp_path / "group", "seed0", {"m": 1.0}, stamp="20240101_000000")

    later = run / "inference" / "20240301_000000"
    later.mkdir()
    (later / "metrics.json").write_text("{}", encoding="utf-8")

    empty = run / "inference" / "20240401_000000"
    empty.mkdir()

    resolved = SeedInferenceResolver("", "metrics.json").resolve(run)

    assert resolved == later


def test_resolver_raises_when_a_run_has_no_inference(tmp_path):
    run = tmp_path / "group" / "seed0"
    run.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        SeedInferenceResolver("", "metrics.json").resolve(run)


def test_report_requires_at_least_two_runs(tmp_path):
    group = tmp_path / "group"
    run   = _seed_run(group, "seed0", {"m": 1.0})

    with pytest.raises(ValueError):
        _report(group, [run])


def test_report_aggregates_mean_and_sample_std(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0, "overall_r2_gt": 0.5, "split": "test"}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0, "overall_r2_gt": 0.7, "split": "test"}),
    ]

    _report(group, runs).run()

    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))

    assert payload["n_seeds"]               == 2
    assert list(payload["seed_runs"])       == ["seed0", "seed1"]
    assert payload["seed_inference"]        == {"seed0": "stamp", "seed1": "stamp"}
    assert payload["mean"]["curve_mse_gt"]  == pytest.approx(2.0)
    assert payload["std"]["curve_mse_gt"]   == pytest.approx(math.sqrt(2.0))
    assert payload["mean"]["overall_r2_gt"] == pytest.approx(0.6)
    assert "split" not in payload["mean"]


def test_report_markdown_has_per_seed_columns_stamps_and_links(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0}),
    ]

    report_path = _report(group, runs).run()

    text = report_path.read_text(encoding="utf-8")

    assert report_path == group / "inference" / "stamp" / "report.md"
    assert "seed0" in text and "seed1" in text
    assert "±" in text
    assert "`stamp`" in text
    assert "../../seed0/inference/stamp/report.md" in text
    assert "`curve_mse_gt`" in text


def test_report_excludes_per_slice_metrics_from_the_tables(tmp_path):
    group = tmp_path / "group"
    runs  = [
        _seed_run(group, "seed0", {"curve_mse_gt": 1.0, "ssim_gt_elev_3": 0.9}),
        _seed_run(group, "seed1", {"curve_mse_gt": 3.0, "ssim_gt_elev_3": 0.8}),
    ]

    report_path = _report(group, runs).run()

    payload = json.loads((group / "inference" / "stamp" / "metrics.json").read_text(encoding="utf-8"))
    text    = report_path.read_text(encoding="utf-8")

    assert payload["mean"]["ssim_gt_elev_3"] == pytest.approx(0.85)
    assert "ssim_gt_elev_3" not in text


def _config(runs_dir: Path, group_tags: list[str], inference_subdir: str = "", output_subdir: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        runs_dir             = str(runs_dir),
        group_tags           = group_tags,
        inference_subdir     = inference_subdir,
        output_subdir        = output_subdir,
        metrics_filename     = "metrics.json",
        report_filename      = "report.md",
        compute_disagreement = False,
        cubes_subdir         = "cubes",
    )


def test_comparison_compares_each_selected_group_in_isolation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group_a = tmp_path / "runs" / "group_a"
    _seed_run(group_a, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group_a, "seed1", {"curve_mse_gt": 3.0})

    group_b = tmp_path / "runs" / "group_b"
    _seed_run(group_b, "seed0", {"curve_mse_gt": 10.0})
    _seed_run(group_b, "seed1", {"curve_mse_gt": 30.0})

    reports = SeedComparison(_config(tmp_path / "runs", ["group_a", "group_b"], output_subdir="agg")).run()

    payload_a = json.loads((group_a / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))
    payload_b = json.loads((group_b / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert reports == [group_a / "inference" / "agg" / "report.md", group_b / "inference" / "agg" / "report.md"]
    assert payload_a["mean"]["curve_mse_gt"] == pytest.approx(2.0)
    assert payload_b["mean"]["curve_mse_gt"] == pytest.approx(20.0)


def test_comparison_raises_on_unknown_group_tag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})

    with pytest.raises(FileNotFoundError):
        SeedComparison(_config(tmp_path / "runs", ["group", "missing"])).run()


def test_comparison_raises_when_a_group_holds_fewer_than_two_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})

    with pytest.raises(ValueError):
        SeedComparison(_config(tmp_path / "runs", ["group"])).run()


def test_comparison_treats_runs_dir_as_the_group_when_no_tags_are_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0})

    reports = SeedComparison(_config(group, [], output_subdir="agg")).run()

    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert reports == [group / "inference" / "agg" / "report.md"]
    assert payload["mean"]["curve_mse_gt"] == pytest.approx(2.0)


def test_comparison_output_falls_back_to_the_inference_stamp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0})

    SeedComparison(_config(group, [], inference_subdir="stamp")).run()

    assert (group / "inference" / "stamp" / "metrics.json").is_file()


def test_comparison_output_gets_a_fresh_stamp_when_nothing_is_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", {"curve_mse_gt": 1.0})
    _seed_run(group, "seed1", {"curve_mse_gt": 3.0})

    reports = SeedComparison(_config(group, [])).run()

    assert reports[0].parent.parent == group / "inference"
    assert reports[0].parent.name not in ("stamp", "")
