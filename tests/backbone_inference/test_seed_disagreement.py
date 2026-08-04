from __future__ import annotations

import json
import math
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest

from pipelines.backbone.inference.analysis.seed_comparison   import SeedComparison
from pipelines.backbone.inference.analysis.seed_disagreement import SeedDisagreementMaps

from tests.conftest import SilentLogger


N_ELEV, N_AZ, N_RG = 4, 3, 2


def _params(*slots: tuple) -> np.ndarray:
    cube = np.zeros((len(slots) * 3, N_AZ, N_RG), dtype=np.float32)

    for k, (amp, mu, sigma) in enumerate(slots):
        cube[3 * k, :, :]     = amp
        cube[3 * k + 1, :, :] = mu
        cube[3 * k + 2, :, :] = sigma

    return cube


def _seed_run(group: Path, name: str, pred_curves: np.ndarray, params_pred: np.ndarray, params_gt: np.ndarray | None = None, stamp: str = "stamp") -> Path:
    run_dir   = group / name
    out_dir   = run_dir / "inference" / stamp
    cubes_dir = out_dir / "cubes"
    cubes_dir.mkdir(parents=True)

    (run_dir / "meta").mkdir(exist_ok=True)
    np.save(cubes_dir / "pred_curves.npy", pred_curves.astype(np.float32))
    np.save(cubes_dir / "params_pred.npy", params_pred.astype(np.float32))
    np.save(cubes_dir / "params_gt.npy",   (params_gt if params_gt is not None else params_pred).astype(np.float32))
    np.save(cubes_dir / "pixel_mse.npy",   pred_curves.mean(axis=0).astype(np.float32) + 0.1)

    (out_dir / "metrics.json").write_text(json.dumps({"curve_mse_gt": 1.0, "split_region": [0, N_AZ, 0, N_RG]}), encoding="utf-8")
    (out_dir / "report.md").write_text("# per-seed report", encoding="utf-8")

    return run_dir


def _maps(group: Path, runs: list[Path], stamp: str = "stamp") -> SeedDisagreementMaps:
    return SeedDisagreementMaps(
        group_dir        = group,
        run_dirs         = runs,
        inference_dirs   = [run / "inference" / stamp for run in runs],
        cubes_subdir     = "cubes",
        metrics_filename = "metrics.json",
        output_dir       = group / "inference" / "agg",
        logger           = SilentLogger(),
    )


def _config(runs_dir: Path, compute_disagreement: bool) -> SimpleNamespace:
    return SimpleNamespace(
        runs_dir             = str(runs_dir),
        group_tags           = [],
        inference_subdir     = "stamp",
        output_subdir        = "agg",
        metrics_filename     = "metrics.json",
        report_filename      = "report.md",
        compute_disagreement = compute_disagreement,
        cubes_subdir         = "cubes",
    )


def test_identical_seeds_give_zero_disagreement(tmp_path):
    group  = tmp_path / "group"
    curves = np.ones((N_ELEV, N_AZ, N_RG))
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


def test_known_offsets_give_the_sample_std(tmp_path):
    group    = tmp_path / "group"
    curves_a = np.zeros((N_ELEV, N_AZ, N_RG))
    curves_b = np.zeros((N_ELEV, N_AZ, N_RG))

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

    assert profile[0, 0] == pytest.approx(math.sqrt(2.0) / N_ELEV)
    assert profile[1, 1] == pytest.approx(0.0)
    assert np.allclose(amp,   math.sqrt(2.0))
    assert np.allclose(mu,    math.sqrt(8.0))
    assert np.allclose(sigma, 0.0)


def test_mu_is_nan_where_fewer_than_two_seeds_are_active(tmp_path):
    group = tmp_path / "group"

    curves   = np.zeros((N_ELEV, N_AZ, N_RG))
    active   = _params((1.0, 5.0, 2.0))
    inactive = _params((0.0, 99.0, 99.0))

    runs = [
        _seed_run(group, "seed0", curves, active, params_gt=active),
        _seed_run(group, "seed1", curves, inactive, params_gt=active),
    ]

    _maps(group, runs).run()

    mu  = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_mu.npy")
    amp = np.load(runs[0] / "inference" / "stamp" / "cubes" / "seed_std_amp.npy")

    assert np.all(np.isnan(mu))
    assert np.all(np.isnan(amp))


def test_swapped_slot_order_across_seeds_gives_zero_disagreement(tmp_path):
    group  = tmp_path / "group"
    curves = np.ones((N_ELEV, N_AZ, N_RG))

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


def test_missing_cubes_raise(tmp_path):
    group = tmp_path / "group"
    runs  = [_seed_run(group, f"seed{i}", np.zeros((N_ELEV, N_AZ, N_RG)), _params((1.0, 5.0, 2.0))) for i in range(2)]

    (runs[1] / "inference" / "stamp" / "cubes" / "params_pred.npy").unlink()

    with pytest.raises(FileNotFoundError):
        _maps(group, runs).run()


def test_shape_mismatch_raises(tmp_path):
    group = tmp_path / "group"
    run_a = _seed_run(group, "seed0", np.zeros((N_ELEV, N_AZ, N_RG)), _params((1.0, 5.0, 2.0)))
    run_b = _seed_run(group, "seed1", np.zeros((N_ELEV, N_AZ + 1, N_RG)), _params((1.0, 5.0, 2.0)))

    np.save(run_b / "inference" / "stamp" / "cubes" / "params_pred.npy", np.zeros((3, N_AZ + 1, N_RG), dtype=np.float32))

    with pytest.raises(ValueError):
        _maps(group, [run_a, run_b]).run()


def test_comparison_writes_maps_figures_and_report_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", np.zeros((N_ELEV, N_AZ, N_RG)), _params((1.0, 5.0, 2.0)))
    _seed_run(group, "seed1", np.ones((N_ELEV, N_AZ, N_RG)),  _params((2.0, 6.0, 3.0)))

    reports = SeedComparison(_config(group, compute_disagreement=True)).run()

    text    = reports[0].read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert "Seed disagreement maps" in text
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


def test_comparison_skips_disagreement_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    group = tmp_path / "runs" / "group"
    _seed_run(group, "seed0", np.zeros((N_ELEV, N_AZ, N_RG)), _params((1.0, 5.0, 2.0)))
    _seed_run(group, "seed1", np.ones((N_ELEV, N_AZ, N_RG)),  _params((2.0, 6.0, 3.0)))

    reports = SeedComparison(_config(group, compute_disagreement=False)).run()

    text    = reports[0].read_text(encoding="utf-8")
    payload = json.loads((group / "inference" / "agg" / "metrics.json").read_text(encoding="utf-8"))

    assert "Seed disagreement maps" not in text
    assert "disagreement"           not in payload
    assert not (group / "seed0" / "inference" / "stamp" / "cubes" / "seed_std_profile.npy").exists()
