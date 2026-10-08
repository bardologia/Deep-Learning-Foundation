"""Tests covering per-pixel seed disagreement maps and their integration into the seed comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest

from pipelines.backbone.inference.analysis.seed_comparison   import SeedComparison
from pipelines.backbone.inference.analysis.seed_disagreement import SeedDisagreementMaps
from tools.runtime.completion                                import CompletionMarker

from tests.backbone_inference._helpers import StageLogger
from tests.conftest                    import SilentLogger


N_Z, N_ROWS, N_COLS = 4, 3, 2


def _params(*slots: tuple) -> np.ndarray:
    """Builds a constant parameter cube from per-slot triples.

    Args:
        *slots : One (amplitude, centre in metres, width in metres) triple per slot.

    Returns:
        Parameter cube of shape (3 * n_slots, N_ROWS, N_COLS).
    """
    cube = np.zeros((len(slots) * 3, N_ROWS, N_COLS), dtype=np.float32)

    for k, (amp, mu, sigma) in enumerate(slots):
        cube[3 * k, :, :]     = amp
        cube[3 * k + 1, :, :] = mu
        cube[3 * k + 2, :, :] = sigma

    return cube


def _seed_run(group: Path, name: str, pred_curves: np.ndarray, params_pred: np.ndarray, params_gt: np.ndarray | None = None, stamp: str = "stamp", activity_threshold: float | None = 1e-4, region_start: tuple[int, int] = (0, 0)) -> Path:
    """Writes a seed run directory with prediction cubes, metrics and a report.

    Args:
        group              : Directory holding the seed runs.
        name               : Name of the seed run directory.
        pred_curves        : Predicted curve cube of shape (N_Z, N_ROWS, N_COLS).
        params_pred        : Predicted parameter cube of shape (3 * n_slots, N_ROWS, N_COLS).
        params_gt          : Ground-truth parameter cube, defaulting to params_pred.
        stamp              : Name of the inference output subdirectory.
        activity_threshold : Activity threshold recorded in the metrics file, or None to leave it out as inference wrote it before the field existed.
        region_start       : Row and column start of the split region recorded in the metrics file.

    Returns:
        Path of the created run directory.
    """
    run_dir   = group / name
    out_dir   = run_dir / "inference" / stamp
    cubes_dir = out_dir / "cubes"
    cubes_dir.mkdir(parents=True)

    (run_dir / "meta").mkdir(exist_ok=True)
    np.save(cubes_dir / "pred_curves.npy", pred_curves.astype(np.float32))
    np.save(cubes_dir / "params_pred.npy", params_pred.astype(np.float32))
    np.save(cubes_dir / "params_gt.npy",   (params_gt if params_gt is not None else params_pred).astype(np.float32))
    np.save(cubes_dir / "pixel_mse.npy",   pred_curves.mean(axis=0).astype(np.float32) + 0.1)

    row_start, col_start = region_start
    metrics              = {"curve_mse_gt": 1.0, "split_region": [row_start, row_start + N_ROWS, col_start, col_start + N_COLS]}
    if activity_threshold is not None:
        metrics["activity_threshold"] = activity_threshold

    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out_dir / "report.md").write_text("# per-seed report", encoding="utf-8")
    CompletionMarker.stamp(out_dir, {"stage": "inference", "split": "test"})

    return run_dir


def _maps(group: Path, runs: list[Path], stamp: str = "stamp", active_amp_thr: float = 1e-4, logger=None) -> SeedDisagreementMaps:
    """Returns a seed disagreement map builder over the given runs, inference stamp and activity threshold."""
    return SeedDisagreementMaps(
        group_dir        = group,
        run_dirs         = runs,
        inference_dirs   = [run / "inference" / stamp for run in runs],
        cubes_subdir     = "cubes",
        metrics_filename = "metrics.json",
        output_dir       = group / "inference" / "agg",
        active_amp_thr   = active_amp_thr,
        logger           = logger if logger is not None else SilentLogger(),
    )


def _config(runs_dir: Path, compute_disagreement: bool) -> SimpleNamespace:
    """Returns a seed comparison config stub with disagreement computation on or off."""
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
        cubes_subdir         = "cubes",
    )


def test_identical_seeds_give_zero_disagreement(tmp_path):
    """Seeds predicting the same cubes give zero standard deviation on every map."""
    group  = tmp_path / "group"
    curves = np.ones((N_Z, N_ROWS, N_COLS))
    params = _params((1.0, 5.0, 2.0))
    runs   = [_seed_run(group, f"seed{i}", curves, params) for i in range(2)]

    result = _maps(group, runs).run()

    for run in runs:
        cubes_dir = run / "inference" / "stamp" / "cubes"
        assert np.allclose(np.load(cubes_dir / "seed_std_profile.npy"), 0.0)
        assert np.allclose(np.load(cubes_dir / "seed_std_amp.npy"),     0.0)
        assert np.allclose(np.load(cubes_dir / "seed_std_mu.npy"),      0.0)
        assert np.allclose(np.load(cubes_dir / "seed_std_sigma.npy"),   0.0)

    assert result["summary"]["n_seeds"]               == 2
    assert result["summary"]["seed_std_profile_mean"] == pytest.approx(0.0)


def test_stages_are_timed_and_the_profile_map_reports_progress(tmp_path):
    """Every heavy stage runs under a timer, the profile map drives one bar over the z bins, and the summary is tabled."""
    group  = tmp_path / "group"
    curves = np.ones((N_Z, N_ROWS, N_COLS))
    params = _params((1.0, 5.0, 2.0))
    runs   = [_seed_run(group, f"seed{i}", curves, params) for i in range(2)]

    logger = StageLogger()
    _maps(group, runs, logger=logger).run()

    assert logger.timers == ["Seed disagreement: profile map", "Seed disagreement: slot matching", "Seed disagreement: maps written", "Seed disagreement: risk-coverage"]
    assert logger.totals == [N_Z]
    assert logger.tables == ["Seed disagreement summary"]
    assert ("info", f"Profile cubes of 2 seeds: {N_Z} z bins x {N_ROWS} rows x {N_COLS} cols") in logger.messages
    assert ("info", "Seed disagreement over 2 seeds at activity threshold 0.0001") in logger.messages


def test_known_offsets_give_the_sample_std(tmp_path):
    """Profile, amplitude, centre and width maps carry the sample standard deviation across seeds."""
    group    = tmp_path / "group"
    curves_a = np.zeros((N_Z, N_ROWS, N_COLS))
    curves_b = np.zeros((N_Z, N_ROWS, N_COLS))

    curves_b[0, 0, 0] = 2.0

    gt   = _params((2.0, 7.0, 2.0))
    runs = [
        _seed_run(group, "seed0", curves_a, _params((1.0, 5.0, 2.0)), params_gt=gt),
        _seed_run(group, "seed1", curves_b, _params((3.0, 9.0, 2.0)), params_gt=gt),
    ]

    _maps(group, runs).run()

    profile = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_profile.npy")
    amp     = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_amp.npy")
    mu      = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_mu.npy")
    sigma   = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_sigma.npy")

    assert profile[0, 0] == pytest.approx(math.sqrt(2.0) / N_Z)
    assert profile[1, 1] == pytest.approx(0.0)
    assert np.allclose(amp,   math.sqrt(2.0))
    assert np.allclose(mu,    math.sqrt(8.0))
    assert np.allclose(sigma, 0.0)


def test_risk_map_and_offsets_are_read_once_per_run(tmp_path, monkeypatch):
    """The mean risk map and the region offsets are each read once per run."""
    group  = tmp_path / "group"
    curves = np.ones((N_Z, N_ROWS, N_COLS))
    params = _params((1.0, 5.0, 2.0))
    runs   = [_seed_run(group, f"seed{i}", curves, params) for i in range(2)]

    maps  = _maps(group, runs)
    calls = {"risk": 0, "offsets": 0}

    mean_risk_map = maps._mean_risk_map
    offsets       = maps._offsets

    monkeypatch.setattr(maps, "_mean_risk_map", lambda: (calls.__setitem__("risk", calls["risk"] + 1), mean_risk_map())[1])
    monkeypatch.setattr(maps, "_offsets",       lambda: (calls.__setitem__("offsets", calls["offsets"] + 1), offsets())[1])

    maps.run()

    assert calls == {"risk": 1, "offsets": 1}


def test_mu_is_nan_where_fewer_than_two_seeds_are_active(tmp_path):
    """Pixels with fewer than two active seeds report NaN parameter disagreement."""
    group = tmp_path / "group"

    curves   = np.zeros((N_Z, N_ROWS, N_COLS))
    active   = _params((1.0, 5.0, 2.0))
    inactive = _params((0.0, 99.0, 99.0))

    runs = [
        _seed_run(group, "seed0", curves, active, params_gt=active),
        _seed_run(group, "seed1", curves, inactive, params_gt=active),
    ]

    logger = StageLogger()
    result = _maps(group, runs, logger=logger).run()

    mu  = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_mu.npy")
    amp = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_amp.npy")

    assert np.all(np.isnan(mu))
    assert np.all(np.isnan(amp))
    assert result["summary"]["seed_std_mu_finite_n"]         == 0
    assert result["summary"]["seed_std_mu_finite_frac"]      == 0.0
    assert result["summary"]["seed_std_mu_mean"]             is None
    assert result["summary"]["seed_std_profile_finite_frac"] == 1.0
    assert any(level == "warning" and message.startswith("seed_std_mu: statistics cover 0.0% of the region") for level, message in logger.messages)


def test_swapped_slot_order_across_seeds_gives_zero_disagreement(tmp_path):
    """Matching slots across seeds removes the disagreement caused by permuted slot order."""
    group  = tmp_path / "group"
    curves = np.ones((N_Z, N_ROWS, N_COLS))

    gt       = _params((1.0, 10.0, 3.0), (2.0, 40.0, 4.0))
    straight = _params((1.0, 10.0, 3.0), (2.0, 40.0, 4.0))
    swapped  = _params((2.0, 40.0, 4.0), (1.0, 10.0, 3.0))

    runs = [
        _seed_run(group, "seed0", curves, straight, params_gt=gt),
        _seed_run(group, "seed1", curves, swapped, params_gt=gt),
    ]

    _maps(group, runs).run()

    cubes_dir = runs[0] / "inference" / "stamp" / "cubes"

    assert np.allclose(np.load(cubes_dir / "seed_std_amp.npy"),   0.0)
    assert np.allclose(np.load(cubes_dir / "seed_std_mu.npy"),    0.0)
    assert np.allclose(np.load(cubes_dir / "seed_std_sigma.npy"), 0.0)


def test_recorded_threshold_above_a_faint_slot_masks_it(tmp_path):
    """Seeds scored at a threshold above a slot's amplitude lose that slot from the matched parameter maps."""
    curves = np.zeros((N_Z, N_ROWS, N_COLS))
    faint  = _params((5e-4, 5.0, 2.0))
    loose  = [_seed_run(tmp_path / "loose",  f"seed{i}", curves, faint) for i in range(2)]
    strict = [_seed_run(tmp_path / "strict", f"seed{i}", curves, faint, activity_threshold=1e-3) for i in range(2)]

    _maps(tmp_path / "loose",  loose,  active_amp_thr=1e-4).run()
    _maps(tmp_path / "strict", strict, active_amp_thr=1e-3).run()

    loose_mu  = np.load(loose [0] / "inference" / "stamp" / "cubes" / "seed_std_mu.npy")
    strict_mu = np.load(strict[0] / "inference" / "stamp" / "cubes" / "seed_std_mu.npy")

    assert np.allclose(loose_mu, 0.0)
    assert np.all(np.isnan(strict_mu))


def test_missing_cubes_raise(tmp_path):
    """A run missing a prediction cube is rejected."""
    group = tmp_path / "group"
    runs  = [_seed_run(group, f"seed{i}", np.zeros((N_Z, N_ROWS, N_COLS)), _params((1.0, 5.0, 2.0))) for i in range(2)]

    (runs[1] / "inference" / "stamp" / "cubes" / "params_pred.npy").unlink()

    with pytest.raises(FileNotFoundError):
        _maps(group, runs).run()


def test_seeds_inferred_on_different_regions_are_refused(tmp_path):
    """Seeds whose inferences cover different regions of the same shape are refused before any map is written."""
    group  = tmp_path / "group"
    curves = np.zeros((N_Z, N_ROWS, N_COLS))
    params = _params((1.0, 5.0, 2.0))
    runs   = [_seed_run(group, "seed0", curves, params), _seed_run(group, "seed1", curves, params, region_start=(100, 0))]

    with pytest.raises(ValueError, match="different regions"):
        _maps(group, runs).run()

    assert not (runs[0] / "inference" / "stamp" / "cubes" / "seed_std_profile.npy").exists()


def test_shape_mismatch_raises(tmp_path):
    """Runs whose cubes differ in spatial shape are rejected."""
    group = tmp_path / "group"
    run_a = _seed_run(group, "seed0", np.zeros((N_Z, N_ROWS, N_COLS)), _params((1.0, 5.0, 2.0)))
    run_b = _seed_run(group, "seed1", np.zeros((N_Z, N_ROWS + 1, N_COLS)), _params((1.0, 5.0, 2.0)))

    np.save(run_b / "inference" / "stamp" / "cubes" / "params_pred.npy", np.zeros((3, N_ROWS + 1, N_COLS), dtype=np.float32))

    with pytest.raises(ValueError):
        _maps(group, [run_a, run_b]).run()


def test_comparison_writes_maps_figures_and_report_section(tmp_path):
    """Enabling disagreement writes per-seed map cubes, aggregate figures and the report section with risk-coverage scalars."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", np.zeros((N_Z, N_ROWS, N_COLS)), _params((1.0, 5.0, 2.0)))
    _seed_run(group, "seed1", np.ones((N_Z, N_ROWS, N_COLS)),  _params((2.0, 6.0, 3.0)))

    reports = SeedComparison(_config(group, compute_disagreement=True), SilentLogger()).run()

    text    = reports[0].read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert payload["activity_threshold"] == 1e-4
    assert payload["disagreement"]["seed_std_amp_finite_frac"] == 1.0
    assert "Seed disagreement maps" in text
    assert f"100.0% ({N_ROWS * N_COLS} px)" in text
    assert "seed_std_profile"       in text
    assert "AURC" in text
    assert payload["disagreement"]["seed_std_amp_mean"] == pytest.approx(math.sqrt(0.5))
    assert payload["disagreement"]["risk_coverage_aurc"] > 0.0
    assert (group / "inference" / "agg" / "figures" / "seed_disagreement" / "risk_coverage.png").is_file()

    for seed in ("seed0", "seed1"):
        for name in SeedDisagreementMaps.MAP_TITLES:
            assert (group / seed / "inference" / "stamp" / "cubes" / f"{name}.npy").is_file()

    for name in SeedDisagreementMaps.MAP_TITLES:
        assert (group / "inference" / "agg" / "figures" / "seed_disagreement" / f"{name}.png").is_file()


def test_comparison_skips_disagreement_when_disabled(tmp_path):
    """Disabling disagreement leaves the report, payload and cube directories untouched."""
    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", np.zeros((N_Z, N_ROWS, N_COLS)), _params((1.0, 5.0, 2.0)))
    _seed_run(group, "seed1", np.ones((N_Z, N_ROWS, N_COLS)),  _params((2.0, 6.0, 3.0)))

    reports = SeedComparison(_config(group, compute_disagreement=False), SilentLogger()).run()

    text    = reports[0].read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert "Seed disagreement maps" not in text
    assert "disagreement"           not in payload
    assert not (group / "seed0" / "inference" / "stamp" / "cubes" / "seed_std_profile.npy").exists()
