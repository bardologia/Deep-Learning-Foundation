"""Tests for GaussianMatcher, which pairs predicted Gaussian slots with ground-truth ones.

Covers the identity assignment on a well-separated exact match, the invariance of the aligned
prediction to the order of the predicted slots, and the NaN marking of ground-truth slots that no
prediction was matched to."""

from __future__ import annotations

import numpy as np

from tools.metrics.gaussian_matching import GaussianMatcher


N_K = 5


def _separated_params(H=3, W=3):
    """Returns a parameter cube of shape (3 * N_K, H, W) whose slots sit at means 0, 10, 20, 30 and 40 metres."""
    params = np.zeros((N_K * 3, H, W), dtype=np.float32)
    for k in range(N_K):
        params[3 * k]     = 1.0
        params[3 * k + 1] = float(k * 10)
        params[3 * k + 2] = 3.0
    return params


def _reorder(params, order):
    """Returns the parameter cube with its Gaussian slots permuted by `order`."""
    grouped = params.reshape(len(order), 3, *params.shape[1:])
    return grouped[order].reshape(len(order) * 3, *params.shape[1:])


def test_assignment_is_identity_for_separated_match():
    """Matching well-separated identical parameters assigns each slot to itself."""
    p        = _separated_params()
    sel      = GaussianMatcher(amp_threshold=1e-4).assignment(p, p.copy(), N_K)
    expected = np.tile(np.arange(N_K), (sel.shape[0], 1))
    assert np.array_equal(sel, expected)


def test_aligned_prediction_invariant_to_pred_slot_order():
    """Permuting the predicted slots leaves the aligned prediction unchanged."""
    rng  = np.random.default_rng(0)
    gt   = _separated_params(4, 4)
    pred = gt.copy()
    for k in range(N_K):
        pred[3 * k + 1] += 0.5 * rng.standard_normal((4, 4)).astype(np.float32)
        pred[3 * k + 2] += 0.2 * rng.standard_normal((4, 4)).astype(np.float32)

    matcher  = GaussianMatcher(amp_threshold=1e-4)
    base     = matcher.aligned_prediction(pred,                          gt, N_K)
    shuffled = matcher.aligned_prediction(_reorder(pred, [2, 0, 4, 1, 3]), gt, N_K)

    assert np.allclose(base, shuffled, atol=1e-5, equal_nan=True)


def test_aligned_prediction_marks_missed_gt_as_nan():
    """A ground-truth slot with no matching prediction comes back as NaN."""
    H, W = 2, 2
    gt   = np.zeros((N_K * 3, H, W), dtype=np.float32)
    pred = np.zeros((N_K * 3, H, W), dtype=np.float32)

    gt[0], gt[1], gt[2] = 1.0, 0.0,  3.0
    gt[3], gt[4], gt[5] = 1.0, 40.0, 3.0

    pred[0], pred[1], pred[2] = 1.0, 0.0, 3.0

    aligned = GaussianMatcher(amp_threshold=1e-4).aligned_prediction(pred, gt, N_K)

    assert np.isfinite(aligned[1, 0, 0])
    assert aligned[1, 0, 0] == 0.0
    assert np.isnan(aligned[4, 0, 0])
