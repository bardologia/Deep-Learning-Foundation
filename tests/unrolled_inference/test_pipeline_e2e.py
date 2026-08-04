from __future__ import annotations

import json

import numpy as np
import pytest

from configuration.inference.unrolled          import UnrolledInferenceConfig
from configuration.training                    import UnrolledEntryConfig
from pipelines.shared.inference.run_classifier import RunClassifier, RunType
from pipelines.unrolled.inference.pipeline     import UnrolledInferencePipeline
from pipelines.unrolled.training.pipeline      import UnrolledTrainingPipeline


pytestmark = [pytest.mark.real_data, pytest.mark.slow, pytest.mark.usefixtures("force_cpu")]


def _entry_config(test_data_dir, params_dir, tmp_path) -> UnrolledEntryConfig:
    entry = UnrolledEntryConfig(run_name="e2e_unrolled", logdir=tmp_path)

    entry.paths.dataset_path    = test_data_dir
    entry.paths.parameters_path = params_dir / "parameters.npy"

    entry.training.epochs        = 1
    entry.training.batch_size    = 8
    entry.training.num_workers   = 0
    entry.training.warmup_steps  = 2
    entry.training.train_azimuth = (1000, 1400)
    entry.training.val_azimuth   = (1400, 1500)
    entry.training.test_azimuth  = (1500, 1600)
    entry.training.patch_size    = (64, 64)
    entry.training.patch_stride  = (64, 64)

    entry.model_overrides = {"n_iterations": 2, "prox_hidden": 4}

    return entry


def test_unrolled_train_then_infer_end_to_end(test_data_dir, params_dir, tmp_path):
    entry = _entry_config(test_data_dir, params_dir, tmp_path)

    results       = UnrolledTrainingPipeline(entry).run()
    run_directory = tmp_path / "e2e_unrolled"

    assert np.isfinite(results["test"]["loss"])
    assert np.isfinite(results["test"]["peak_mae_m"])
    assert (run_directory / "best_model.pt").is_file()
    assert (run_directory / "training_summary.json").is_file()
    assert (run_directory / "meta" / "unrolled_model_config.json").is_file()
    assert (run_directory / "meta" / "dataset_creation_config.json").is_file()
    assert RunClassifier.classify(run_directory) == RunType.UNROLLED

    config = UnrolledInferenceConfig(
        run_directory      = run_directory,
        output_subdir      = "e2e",
        device             = "cpu",
        n_example_profiles = 1,
        save_profile_cube  = True,
    )

    report_path = UnrolledInferencePipeline(config).run()
    output_dir  = run_directory / "inference" / "unrolled" / "e2e"

    assert report_path.is_file()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["split"] == "test"
    assert metrics["split_region"] == [1500, 1600, 500, 1000]
    assert metrics["model_name"] == "gamma_net"
    assert metrics["n_iterations"] == 2
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["curve_rmse"])
    assert np.isfinite(metrics["peak_mae_m"])
    assert 0.0 < metrics["valid_fraction"] <= 1.0
    assert metrics["n_pixels"] == 100 * 500

    cube = np.load(output_dir / "profile_cube.npy")
    assert cube.shape[1:] == (100, 500)
    assert np.isfinite(cube).all()
    assert (cube >= 0.0).all()

    figures_dir = output_dir / "figures"
    assert (figures_dir / "curve_l1_map.png").is_file()
    assert (figures_dir / "peak_error_map.png").is_file()
    assert (figures_dir / "gt_peak_height.png").is_file()
    assert (figures_dir / "pred_peak_height.png").is_file()
    assert (figures_dir / "error_histogram.png").is_file()
    assert (figures_dir / "profiles" / "best_01.png").is_file()
    assert (figures_dir / "profiles" / "median_01.png").is_file()
    assert (figures_dir / "profiles" / "worst_01.png").is_file()
