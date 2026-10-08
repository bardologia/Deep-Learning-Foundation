"""Tests for the image and profile autoencoder inference metric computations.

Covers the shared metric base, the key sets each family reports, the refusal of a profile
split with no active curve, the pass-through of normalized errors measured before
denormalization, and that the metric objects reuse the result arrays instead of copying them
into float64.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.autoencoder_common.inference.metrics  import AeMetricsBase
from pipelines.image_autoencoder.inference.metrics   import ImageAeMetrics
from pipelines.profile_autoencoder.inference.metrics import ProfileAeMetrics


NORMALIZED = {"mse_mean_normalized": 0.25, "mae_mean_normalized": 0.5}


def test_metrics_share_base():
    """Both autoencoder metric classes derive from AeMetricsBase."""
    assert issubclass(ImageAeMetrics, AeMetricsBase)
    assert issubclass(ProfileAeMetrics, AeMetricsBase)


def test_image_metrics_compute_keys():
    """Image metrics report patch and channel counts plus the reconstruction and embedding keys."""
    rng = np.random.default_rng(0)
    res = SimpleNamespace(
        gt                = rng.random((6, 2, 4, 4)).astype(np.float32),
        pred              = rng.random((6, 2, 4, 4)).astype(np.float32),
        embeddings        = rng.random((6, 8)).astype(np.float32),
        normalized_errors = dict(NORMALIZED),
    )

    out = ImageAeMetrics(res).compute()

    assert out["n_patches"] == 6
    assert out["n_channels"] == 2
    for key in ("mse_mean", "psnr", "channel_mse", "mse_mean_normalized", "embedding_norm_mean"):
        assert key in out


def test_profile_metrics_compute_keys():
    """Profile metrics report curve count and profile length plus the curve-shape and embedding keys."""
    rng = np.random.default_rng(1)
    res = SimpleNamespace(
        gt                = rng.random((10, 20)).astype(np.float32),
        pred              = rng.random((10, 20)).astype(np.float32),
        embeddings        = rng.random((10, 4)).astype(np.float32),
        normalized_errors = dict(NORMALIZED),
    )

    out = ProfileAeMetrics(res, np.linspace(0.0, 1.0, 20), 1e-3).compute()

    assert out["n_curves"] == 10
    assert out["profile_length"] == 20
    for key in ("mse_mean", "pearson_mean", "power_rel_error_mean", "peak_location_mae", "embedding_norm_mean"):
        assert key in out


def test_profile_metrics_refuse_a_split_with_no_active_curve():
    """Curves all below the activity threshold raise instead of yielding NaN shape and peak metrics."""
    res = SimpleNamespace(
        gt                = np.full((10, 20), 1e-6, dtype=np.float32),
        pred              = np.zeros((10, 20), dtype=np.float32),
        embeddings        = np.zeros((10, 4), dtype=np.float32),
        normalized_errors = dict(NORMALIZED),
    )

    with pytest.raises(ValueError, match="None of the 10 reconstructed curves"):
        ProfileAeMetrics(res, np.linspace(0.0, 1.0, 20), 1e-3).compute()


def test_normalized_errors_are_the_measured_ones_not_a_renormalized_round_trip():
    """The normalized errors are copied from the predictor rather than recomputed after denormalization."""
    rng = np.random.default_rng(2)
    res = SimpleNamespace(
        gt                = rng.random((4, 6)).astype(np.float32),
        pred              = rng.random((4, 6)).astype(np.float32),
        embeddings        = rng.random((4, 2)).astype(np.float32),
        normalized_errors = dict(NORMALIZED),
    )

    out = ProfileAeMetrics(res, np.linspace(0.0, 1.0, 6), 1e-3).compute()

    assert out["mse_mean_normalized"] == NORMALIZED["mse_mean_normalized"]
    assert out["mae_mean_normalized"] == NORMALIZED["mae_mean_normalized"]


def test_metrics_do_not_duplicate_the_result_arrays_in_float64():
    """The metric object holds the predictor's float32 arrays by reference, without upcasting."""
    rng = np.random.default_rng(3)
    res = SimpleNamespace(
        gt                = rng.random((4, 6)).astype(np.float32),
        pred              = rng.random((4, 6)).astype(np.float32),
        embeddings        = rng.random((4, 2)).astype(np.float32),
        normalized_errors = dict(NORMALIZED),
    )

    metrics = ProfileAeMetrics(res, np.linspace(0.0, 1.0, 6), 1e-3)

    assert metrics.gt is res.gt
    assert metrics.pred is res.pred
    assert metrics.emb is res.embeddings


def test_write_json_roundtrip(tmp_path):
    """write_json creates the metrics file at the requested path."""
    path = AeMetricsBase.write_json({"a": 1, "b": 2.5}, tmp_path / "m.json")
    assert path.exists()


def _result(shape, embedding_dim, seed):
    """Returns a stub predictor result with random ground truth, prediction and embeddings of the given shape."""
    rng = np.random.default_rng(seed)

    return SimpleNamespace(
        gt                = rng.random(shape).astype(np.float32),
        pred              = rng.random(shape).astype(np.float32),
        embeddings        = rng.random((shape[0], embedding_dim)).astype(np.float32),
        normalized_errors = dict(NORMALIZED),
    )


def test_per_sample_mse_is_computed_once_and_reused():
    """Repeated per_sample_mse calls return the cached array, equal to the manual per-curve MSE."""
    result  = _result((10, 20), embedding_dim=4, seed=3)
    metrics = ProfileAeMetrics(result, np.linspace(0.0, 1.0, 20), 1e-3)

    first  = metrics.per_sample_mse()
    second = metrics.per_sample_mse()

    assert second is first
    np.testing.assert_allclose(first, np.mean((result.pred - result.gt) ** 2, axis=1), rtol=1e-6)


def test_per_sample_mse_reduces_every_axis_but_the_sample_axis():
    """For image patches of shape (patches, channels, height, width) the MSE reduces all but the patch axis."""
    result  = _result((6, 2, 4, 4), embedding_dim=8, seed=4)
    metrics = ImageAeMetrics(result)

    np.testing.assert_allclose(metrics.per_sample_mse(), np.mean((result.pred - result.gt) ** 2, axis=(1, 2, 3)), rtol=1e-6)
