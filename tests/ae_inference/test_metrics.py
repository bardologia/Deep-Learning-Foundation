from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipelines.autoencoder_common.inference.metrics  import AeMetricsBase
from pipelines.image_autoencoder.inference.metrics   import ImageAeMetrics
from pipelines.profile_autoencoder.inference.metrics import ProfileAeMetrics


NORMALIZED = {"mse_mean_normalized": 0.25, "mae_mean_normalized": 0.5}


def test_metrics_share_base():
    assert issubclass(ImageAeMetrics, AeMetricsBase)
    assert issubclass(ProfileAeMetrics, AeMetricsBase)


def test_image_metrics_compute_keys():
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


def test_normalized_errors_are_the_measured_ones_not_a_renormalized_round_trip():
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
    path = AeMetricsBase.write_json({"a": 1, "b": 2.5}, tmp_path / "m.json")
    assert path.exists()
