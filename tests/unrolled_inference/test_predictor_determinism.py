from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from configuration.inference.unrolled       import UnrolledInferenceConfig
from models.unrolled                        import get_unrolled
from pipelines.unrolled.inference.predictor import UnrolledPredictor

from tools.monitoring.logger import Logger


HEIGHT = 9
WIDTH  = 5
TRACKS = 3
X_LEN  = 32


def _stub_run(noise_std: float) -> SimpleNamespace:
    torch.manual_seed(0)
    model = get_unrolled("gamma_net", n_iterations=2, prox_hidden=4)[0].eval()

    rng = np.random.default_rng(0)
    gt  = np.zeros((6, HEIGHT, WIDTH), dtype=np.float32)

    gt[0::3] = rng.uniform(1.0, 3.0,  size=(2, HEIGHT, WIDTH))
    gt[1::3] = rng.uniform(0.0, 40.0, size=(2, HEIGHT, WIDTH))
    gt[2::3] = rng.uniform(2.0, 6.0,  size=(2, HEIGHT, WIDTH))

    kz = np.linspace(-0.15, 0.15, TRACKS, dtype=np.float32).reshape(TRACKS, 1, 1) * np.ones((TRACKS, HEIGHT, WIDTH), dtype=np.float32)

    return SimpleNamespace(
        model         = model,
        gt_parameters = gt,
        kz_field      = kz,
        x_axis        = np.linspace(-20.0, 60.0, X_LEN, dtype=np.float32),
        ppg           = 3,
        power_floor   = 1e-6,
        noise_std     = noise_std,
    )


def _predictor(tmp_path, noise_std: float) -> UnrolledPredictor:
    config = UnrolledInferenceConfig(run_directory=tmp_path, device="cpu", chunk_cells=X_LEN * WIDTH * 4)
    logger = Logger(log_dir=str(tmp_path / "logs"), name="predictor", level="ERROR")

    return UnrolledPredictor(_stub_run(noise_std), config, logger)


def test_profile_pair_matches_error_maps_under_noise(tmp_path):
    predictor  = _predictor(tmp_path, noise_std=0.5)
    prediction = predictor.run_inference()

    for azimuth, range_index in ((0, 0), (5, 2), (8, 4)):
        pair = predictor.profile_pair(azimuth, range_index)
        l1   = np.abs(pair["pred"] - pair["gt"]).mean()

        assert np.isclose(l1, prediction.curve_l1_map[azimuth, range_index], rtol=1e-5, atol=1e-6)


def test_run_inference_is_repeatable_under_noise(tmp_path):
    predictor = _predictor(tmp_path, noise_std=0.5)

    first  = predictor.run_inference()
    second = predictor.run_inference()

    assert np.array_equal(first.curve_l1_map, second.curve_l1_map)
    assert np.array_equal(first.peak_error_map, second.peak_error_map)


def test_profile_pair_reuses_the_chunk_it_already_predicted(tmp_path):
    predictor = _predictor(tmp_path, noise_std=0.0)
    calls     = []

    original = predictor._predict_rows
    predictor._predict_rows = lambda start, end: (calls.append((start, end)), original(start, end))[1]

    for range_index in range(WIDTH):
        predictor.profile_pair(0, range_index)

    assert calls == [(0, 4)]


def test_noise_free_profile_pair_matches_error_maps(tmp_path):
    predictor  = _predictor(tmp_path, noise_std=0.0)
    prediction = predictor.run_inference()

    pair = predictor.profile_pair(6, 1)
    l1   = np.abs(pair["pred"] - pair["gt"]).mean()

    assert np.isclose(l1, prediction.curve_l1_map[6, 1], rtol=1e-5, atol=1e-6)
