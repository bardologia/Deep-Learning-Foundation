from __future__ import annotations

import json
import math
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest

from pipelines.backbone.inference.analysis.seed_comparison   import SeedComparison
from pipelines.backbone.inference.analysis.seed_disagreement import RiskCoverage, SeedDisagreementMaps, SeedMaps
from tools.runtime.completion                                import CompletionMarker

from tests.conftest import SilentLogger


N_ROWS, N_COLS = 5, 4
CHANNELS       = ["a", "b"]


def _seed_run(group: Path, name: str, predictions: np.ndarray, targets: np.ndarray, stamp: str = "stamp") -> Path:
    run_dir  = group / name
    out_dir  = run_dir / "inference" / stamp
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True)

    (run_dir / "meta").mkdir(exist_ok=True)
    (run_dir / "meta" / "run_summary.json").write_text(json.dumps({"target_channels": CHANNELS}), encoding="utf-8")

    np.save(maps_dir / "predictions.npy", predictions.astype(np.float32))
    np.save(maps_dir / "targets.npy",     targets.astype(np.float32))

    (out_dir / "metrics.json").write_text(json.dumps({"rmse_mean": 1.0}), encoding="utf-8")
    (out_dir / "report.md").write_text("# per-seed report", encoding="utf-8")
    CompletionMarker.stamp(out_dir, {"stage": "inference", "split": "test"})

    return run_dir


def _targets() -> np.ndarray:
    return np.arange(len(CHANNELS) * N_ROWS * N_COLS, dtype=np.float64).reshape(len(CHANNELS), N_ROWS, N_COLS)


def _maps(group: Path, runs: list[Path]) -> SeedMaps:
    return SeedMaps(runs, [run / "inference" / "stamp" for run in runs], "maps")


def _config(runs_dir: Path, compute_disagreement: bool) -> SimpleNamespace:
    return SimpleNamespace(
        runs_dir             = str(runs_dir),
        group_tags           = [],
        split                = "test",
        inference_subdir     = "stamp",
        output_subdir        = "agg",
        metrics_filename     = "metrics.json",
        report_filename      = "report.md",
        compute_disagreement = compute_disagreement,
        compute_ensemble     = False,
        maps_subdir          = "maps",
    )


def test_identical_seeds_give_zero_disagreement(tmp_path):
    group   = tmp_path / "group"
    targets = _targets()
    runs    = [_seed_run(group, f"seed{i}", targets + 1.0, targets) for i in range(2)]

    result = SeedDisagreementMaps(_maps(group, runs), group / "agg", SilentLogger()).run()

    assert np.allclose(np.load(group / "agg" / "seed_std.npy"), 0.0)
    assert result["summary"]["seed_std_a_mean"] == pytest.approx(0.0)
    assert result["summary"]["seed_std_b_p95"]  == pytest.approx(0.0)


def test_known_offsets_give_the_sample_std_per_channel(tmp_path):
    group   = tmp_path / "group"
    targets = _targets()
    runs    = [_seed_run(group, "seed0", targets, targets), _seed_run(group, "seed1", targets + np.array([1.0, 3.0])[:, None, None], targets)]

    result = SeedDisagreementMaps(_maps(group, runs), group / "agg", SilentLogger()).run()
    std    = np.load(group / "agg" / "seed_std.npy")

    assert std.shape == (len(CHANNELS), N_ROWS, N_COLS)
    assert result["summary"]["seed_std_a_mean"] == pytest.approx(math.sqrt(0.5))
    assert result["summary"]["seed_std_b_mean"] == pytest.approx(3.0 * math.sqrt(0.5))
    assert sorted(result["figures"]) == ["a", "b", "label_suspect", "risk_coverage"]


def test_risk_coverage_keeps_the_least_disputed_pixels_first():
    disagreement = np.array([0.1, 0.2, 0.3, 0.4])
    risk         = np.array([1.0, 2.0, 3.0, 4.0])

    rows, scalars = RiskCoverage(disagreement, risk).run()

    assert rows[-1]["coverage"] == 1.0
    assert scalars["full_coverage_risk"] == pytest.approx(2.5)
    assert scalars["risk_at_half_coverage"] == pytest.approx(1.5)
    assert scalars["disagreement_error_spearman"] == pytest.approx(1.0)


def test_comparison_writes_maps_figures_and_report_section(tmp_path):
    group   = tmp_path / "runs" / "group"
    targets = _targets()
    _seed_run(group, "seed0", targets,       targets)
    _seed_run(group, "seed1", targets + 1.0, targets)

    reports = SeedComparison(_config(group, compute_disagreement=True), SilentLogger()).run()

    text    = reports[0].read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))
    figures = group / "inference" / "agg" / "figures" / "seed_disagreement"

    assert "Seed disagreement maps" in text
    assert "AURC" in text
    assert "`a`" in text
    assert payload["disagreement"]["seed_std_a_mean"] == pytest.approx(math.sqrt(0.5))
    assert payload["disagreement"]["risk_coverage_aurc"] > 0.0
    assert (group / "inference" / "agg" / "seed_std.npy").is_file()
    assert sorted(path.name for path in figures.iterdir()) == ["a.png", "b.png", "label_suspect.png", "risk_coverage.png"]


def test_comparison_skips_disagreement_when_disabled(tmp_path):
    group   = tmp_path / "runs" / "group"
    targets = _targets()
    _seed_run(group, "seed0", targets,       targets)
    _seed_run(group, "seed1", targets + 1.0, targets)

    reports = SeedComparison(_config(group, compute_disagreement=False), SilentLogger()).run()

    text    = reports[0].read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert "Seed disagreement maps" not in text
    assert "disagreement"           not in payload
    assert not (group / "inference" / "agg" / "seed_std.npy").exists()
