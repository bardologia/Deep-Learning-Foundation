from __future__ import annotations

import numpy as np
import pytest

from configuration.diagnostics                       import CkaConfig
from pipelines.backbone.inference.analysis.representation_cka import CkaComparison, CkaComputation

from tests.conftest import SilentLogger


def _features(n: int = 200, d: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d))


def test_cka_of_identical_features_is_one():
    X = _features()

    assert CkaComputation.linear_cka(X, X) == pytest.approx(1.0)


def test_cka_is_invariant_to_rotation_and_scale():
    rng = np.random.default_rng(1)
    X   = _features(seed=1)
    Q, _ = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))

    assert CkaComputation.linear_cka(X, 3.0 * (X @ Q)) == pytest.approx(1.0, abs=1e-8)


def test_cka_of_independent_features_is_low():
    value = CkaComputation.linear_cka(_features(seed=2), _features(seed=3))

    assert value < 0.2


def test_cka_rejects_mismatched_samples():
    with pytest.raises(ValueError):
        CkaComputation.linear_cka(_features(n=100), _features(n=50))


def test_cross_matrix_and_alignment_score():
    shared = _features(seed=4)

    layers_a = {"early": shared, "late": _features(seed=5)}
    layers_b = {"first": shared, "second": _features(seed=6)}

    matrix = CkaComputation.cross_matrix(layers_a, layers_b)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] < 0.2

    score = CkaComputation.alignment_score(matrix)
    assert 0.5 < score <= 1.0


def test_alignment_score_of_identity_matrix_is_one():
    assert CkaComputation.alignment_score(np.eye(3)) == pytest.approx(1.0)


def test_alignment_is_validated_from_sample_grids(tmp_path):
    comparison = CkaComparison(CkaConfig(output_dir=tmp_path), SilentLogger())

    grid    = ((1000, 2000, 500, 900), (64, 32), (32, 16))
    strided = ((1000, 2000, 500, 900), (64, 32), (64, 32))

    comparison._validate_alignment([grid, grid])

    with pytest.raises(ValueError):
        comparison._validate_alignment([grid, strided])
