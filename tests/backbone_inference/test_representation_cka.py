"""Tests covering linear CKA similarity, layer cross-matrices and grid-alignment validation."""

from __future__ import annotations

import numpy as np
import pytest

from configuration.diagnostics                                import CkaConfig
from pipelines.backbone.inference.analysis.representation_cka import CkaComparison, CkaComputation

from tests.backbone_inference._helpers import StageLogger, build_run_directory
from tests.conftest                    import SilentLogger


def _features(n: int = 200, d: int = 8, seed: int = 0) -> np.ndarray:
    """Returns a standard normal feature matrix of shape (n, d) drawn from the given seed."""
    return np.random.default_rng(seed).normal(size=(n, d))


def _cka(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Returns the debiased linear CKA of one feature pair via the production cross-matrix path."""
    return float(CkaComputation.cross_matrix({"a": features_a}, {"b": features_b})[0, 0])


def test_cka_of_identical_features_is_one():
    """CKA of a feature matrix with itself is one."""
    X = _features()

    assert _cka(X, X) == pytest.approx(1.0)


def test_cka_is_invariant_to_rotation_and_scale():
    """CKA is unchanged by orthogonal rotation and isotropic scaling."""
    rng = np.random.default_rng(1)
    X   = _features(seed=1)
    Q, _ = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))

    assert _cka(X, 3.0 * (X @ Q)) == pytest.approx(1.0, abs=1e-8)


def test_cka_of_independent_features_is_low():
    """Independent features score a low CKA."""
    value = _cka(_features(seed=2), _features(seed=3))

    assert value < 0.2


def test_cka_rejects_mismatched_samples():
    """Feature matrices with different sample counts are rejected."""
    with pytest.raises(ValueError):
        _cka(_features(n=100), _features(n=50))


def test_debiasing_stays_low_for_independent_high_dimensional_features():
    """Debiasing keeps independent high-dimensional features near zero when samples are few."""
    value = _cka(_features(n=64, d=32, seed=7), _features(n=64, d=32, seed=8))

    assert value < 0.15


def test_constant_features_score_zero():
    """Constant features have no variance to align and score zero."""
    assert _cka(np.ones((50, 4)), _features(n=50)) == 0.0


def test_best_match_profiles_and_divergence_depth():
    """Best-match profiles take row and column maxima, and divergence depth is the first drop below threshold."""
    matrix = np.array([
        [0.9, 0.2, 0.1],
        [0.3, 0.8, 0.2],
        [0.1, 0.2, 0.3],
    ])

    forward, backward = CkaComputation.best_match_profiles(matrix)

    assert np.allclose(forward, [0.9, 0.8, 0.3])
    assert np.allclose(backward, [0.9, 0.8, 0.3])
    assert CkaComputation.divergence_depth(forward)              == pytest.approx(1.0)
    assert CkaComputation.divergence_depth(np.array([0.9, 0.8])) is None
    assert CkaComputation.divergence_depth(np.array([0.2, 0.8])) == pytest.approx(0.0)


def test_cross_matrix_and_alignment_score():
    """The cross matrix pairs every layer of both models and the alignment score rewards a matched diagonal."""
    shared = _features(seed=4)

    layers_a = {"early": shared, "late": _features(seed=5)}
    layers_b = {"first": shared, "second": _features(seed=6)}

    matrix = CkaComputation.cross_matrix(layers_a, layers_b)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] < 0.2

    score = CkaComputation.alignment_score(matrix)
    assert 0.5 < score <= 1.0


def test_self_cross_matrix_is_symmetric_and_matches_the_pairwise_values():
    """A model compared with itself gives a symmetric matrix with a unit diagonal and pairwise-consistent entries."""
    layers = {"early": _features(seed=11), "mid": _features(seed=12), "late": _features(seed=13)}

    matrix = CkaComputation.cross_matrix(layers, layers)

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix[0, 2] == pytest.approx(_cka(layers["early"], layers["late"]))


def test_alignment_score_of_identity_matrix_is_one():
    """A perfectly diagonal cross matrix scores an alignment of one."""
    assert CkaComputation.alignment_score(np.eye(3)) == pytest.approx(1.0)


def test_alignment_is_validated_from_sample_grids(tmp_path):
    """Sample grids must share crop, patch size and stride across the compared runs."""
    comparison = CkaComparison(CkaConfig(output_dir=tmp_path), SilentLogger())

    grid    = ((1000, 2000, 500, 900), (64, 32), (32, 16))
    strided = ((1000, 2000, 500, 900), (64, 32), (64, 32))

    comparison._validate_alignment([grid, grid])

    with pytest.raises(ValueError):
        comparison._validate_alignment([grid, strided])


def _aligned_runs(tmp_path, test_data_dir, params_dir) -> CkaConfig:
    """Builds two identical real runs under one runs root and returns a small CKA configuration over both."""
    runs = tmp_path / "runs"
    for name in ("a", "b"):
        build_run_directory(runs / name, test_data_dir, params_dir)

    return CkaConfig(runs_dir=runs, run_filter=["a/run_unet_e2e", "b/run_unet_e2e"], output_dir=tmp_path / "cka", batch_size=2, max_batches=1, max_layers=4, samples_per_batch=64)


@pytest.mark.dataset
@pytest.mark.slow
def test_comparison_announces_and_tracks_its_matrices(tmp_path, test_data_dir, params_dir):
    """The matrix stage announces the pair count, drives one progress task over every matrix and times each stage."""
    logger = StageLogger()

    CkaComparison(_aligned_runs(tmp_path, test_data_dir, params_dir), logger).run()

    infos = [message for level, message in logger.messages if level == "info"]

    assert "2 runs -> 2 self matrices and 1 pair matrices" in infos
    assert any("probed layers usable, 64 sampled pixels" in message for message in infos)
    assert logger.totals == [3] and logger.advances == 3
    assert logger.timers == ["CKA features [1/2] a/run_unet_e2e", "CKA features [2/2] b/run_unet_e2e", "CKA matrices (3)"]
