from __future__ import annotations

import json
import types

import numpy as np
import pytest

from configuration.inference                       import InferenceConfig
from pipelines.backbone.inference.data_consistency import DataConsistencyEvaluator
from tools.data.regions                            import CropRegion
from tools.sar                                     import GeometryField

from tests.conftest import SilentLogger


def _geometry_field(n_tracks: int, H: int, W: int) -> GeometryField:
    slant   = np.linspace(5000.0, 5200.0, W)
    heights = 3000.0

    return GeometryField(
        labels        = ["REF"] + [f"S{i}" for i in range(1, n_tracks)],
        reference     = "REF",
        wavelength    = 0.23,
        azimuth_start = 0,
        range_start   = 0,
        look_angle    = np.arccos(np.clip(heights / slant, -1.0, 1.0)),
        slant_range   = slant,
        baseline_h    = np.cumsum(np.full((n_tracks, H), 8.0), axis=0) - 8.0,
        baseline_v    = np.zeros((n_tracks, H)),
    )


def _gt_curves(x_axis: np.ndarray, H: int, W: int) -> np.ndarray:
    L      = x_axis.size
    curves = np.zeros((L, H, W), dtype=np.float32)

    for a in range(H):
        for r in range(W):
            mu           = -5.0 + 3.0 * a + 2.5 * r
            curves[:, a, r] = np.exp(-((x_axis - mu) ** 2) / (2.0 * 6.0 ** 2))

    return curves


def _synth_gamma(curves: np.ndarray, kz_track: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
    dx    = float(x_axis[1] - x_axis[0])
    phase = kz_track[None, :, :] * x_axis[:, None, None]

    return (np.exp(1j * phase) * curves).sum(axis=0) * dx


def _write_trainer_config(run_dir, convention: str) -> None:
    docs = run_dir / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "trainer_config.json").write_text(json.dumps({"geometry": {"height_axis_convention": convention}}))


def _build_case(tmp_path, sign: float = 1.0, convention: str = "height"):
    n_tracks, H, W = 3, 6, 5
    x_axis         = np.linspace(-20.0, 80.0, 24, dtype=np.float32)

    field = _geometry_field(n_tracks, H, W)
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    field.save(tmp_path / "meta" / GeometryField.FILENAME)
    _write_trainer_config(tmp_path, convention)

    kz     = field.kz(convention).astype(np.float32)
    curves = _gt_curves(x_axis, H, W)

    inputs = np.zeros((1 + 2 * (n_tracks - 1), H, W), dtype=np.complex64)
    inputs[:n_tracks] = 1.0 + 0.0j
    for track in range(1, n_tracks):
        gamma = _synth_gamma(curves, sign * kz[track], x_axis)
        inputs[n_tracks - 1 + track] = np.exp(1j * np.angle(gamma)).astype(np.complex64)

    region = CropRegion(azimuth_start=0, azimuth_end=H, range_start=0, range_end=W)

    run = types.SimpleNamespace(
        dataset_config   = types.SimpleNamespace(preprocessing_run_directory=tmp_path, secondary_labels=("S1", "S2")),
        split_region     = region,
        global_crop      = region,
        complex_inputs   = inputs,
        n_secondaries    = n_tracks - 1,
    )

    cfg = InferenceConfig(
        run_directory   = tmp_path,
        device          = "cpu",
        phase_multilook = 1,
        physics_floor   = 1e-4,
    )

    return run, cfg, curves, x_axis


def test_identical_curves_have_zero_physics_error(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path)

    consistency = DataConsistencyEvaluator(run, cfg, SilentLogger()).run(curves, curves, x_axis)

    assert consistency.metrics["physics_coherence_error_mean"]  == pytest.approx(0.0, abs=1e-9)
    assert consistency.metrics["physics_covariance_error_mean"] == pytest.approx(0.0, abs=1e-9)
    assert consistency.metrics["physics_valid_fraction"]        == pytest.approx(1.0)
    assert consistency.valid_mask.all()


def test_phase_agreement_detects_correct_sign(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path, sign=1.0)

    consistency = DataConsistencyEvaluator(run, cfg, SilentLogger()).run(curves, curves, x_axis)

    assert consistency.metrics["phase_agreement_gt_mean"] == pytest.approx(1.0, abs=1e-4)
    assert consistency.metrics["phase_agreement_gt_mean"] > consistency.metrics["phase_agreement_gt_flipped_mean"] + 0.05


def test_phase_agreement_detects_flipped_sign(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path, sign=-1.0)

    consistency = DataConsistencyEvaluator(run, cfg, SilentLogger()).run(curves, curves, x_axis)

    assert consistency.metrics["phase_agreement_gt_flipped_mean"] > consistency.metrics["phase_agreement_gt_mean"] + 0.05


def test_wrong_prediction_scores_worse_than_gt(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path)

    shifted = np.roll(curves, 6, axis=0)

    consistency = DataConsistencyEvaluator(run, cfg, SilentLogger()).run(shifted, curves, x_axis)

    assert consistency.metrics["physics_coherence_error_mean"] > 1e-3
    assert consistency.metrics["phase_agreement_gt_mean"] > consistency.metrics["phase_agreement_pred_mean"]


def test_missing_geometry_field_raises(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path)
    (tmp_path / "meta" / GeometryField.FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="geometry field"):
        DataConsistencyEvaluator(run, cfg, SilentLogger()).run(curves, curves, x_axis)


def test_missing_trainer_config_raises(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path)
    (tmp_path / "docs" / "trainer_config.json").unlink()

    with pytest.raises(FileNotFoundError, match="trainer_config"):
        DataConsistencyEvaluator(run, cfg, SilentLogger()).run(curves, curves, x_axis)


def test_height_convention_read_from_training_run(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path, convention="slant")

    consistency = DataConsistencyEvaluator(run, cfg, SilentLogger()).run(curves, curves, x_axis)

    assert consistency.metrics["physics_coherence_error_mean"] == pytest.approx(0.0, abs=1e-9)
    assert consistency.metrics["phase_agreement_gt_mean"]      == pytest.approx(1.0, abs=1e-4)


def test_per_track_metrics_present_and_labelled(tmp_path):
    run, cfg, curves, x_axis = _build_case(tmp_path)

    consistency = DataConsistencyEvaluator(run, cfg, SilentLogger()).run(curves, curves, x_axis)

    assert consistency.track_labels == ["REF", "S1", "S2"]
    for label in ("REF", "S1", "S2"):
        assert f"physics_coherence_error_track_{label}" in consistency.metrics
    for label in ("S1", "S2"):
        assert f"phase_agreement_gt_track_{label}" in consistency.metrics
        assert f"phase_agreement_pred_flipped_track_{label}" in consistency.metrics


def test_phase_residual_stats_offset_and_spread():
    from pipelines.backbone.inference.data_consistency import PhaseResidualStats

    assert PhaseResidualStats.offset(1.0 + 0.0j) == pytest.approx(0.0)
    assert PhaseResidualStats.offset(0.0 + 1.0j) == pytest.approx(np.pi / 2.0)

    assert PhaseResidualStats.spread(1.0)  == pytest.approx(0.0)
    assert PhaseResidualStats.spread(0.5)  > PhaseResidualStats.spread(0.9)
    assert np.isfinite(PhaseResidualStats.spread(0.0))
