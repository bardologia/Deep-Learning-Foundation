from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from configuration.inference.image_autoencoder        import ImageAeInferenceConfig
from configuration.inference.profile_autoencoder      import ProfileAeInferenceConfig
from configuration.training                           import ImageAeEntryConfig
from configuration.training.profile_autoencoder       import ProfileAeEntryConfig
from pipelines.backbone.inference.loader              import RunLoader
from pipelines.image_autoencoder.inference.pipeline   import ImageAeInferencePipeline
from pipelines.image_autoencoder.training.pipeline    import TrainingPipeline
from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferencePipeline
from pipelines.profile_autoencoder.training.pipeline  import TrainingPipeline as ProfileTrainingPipeline


pytestmark = [pytest.mark.real_data, pytest.mark.slow, pytest.mark.usefixtures("force_cpu")]


def _entry_config(test_data_dir, params_dir, tmp_path) -> ImageAeEntryConfig:
    entry = ImageAeEntryConfig(run_name="e2e_image_ae", logdir=tmp_path)

    entry.paths.dataset_path    = test_data_dir
    entry.paths.parameters_path = params_dir / "parameters.npy"

    entry.training.epochs               = 1
    entry.training.validation_frequency = 1
    entry.training.batch_size           = 8
    entry.training.num_workers          = 0
    entry.training.train_azimuth        = (1000, 1400)
    entry.training.val_azimuth          = (1400, 1500)
    entry.training.test_azimuth         = (1500, 1600)
    entry.training.patch_size           = (64, 64)
    entry.training.patch_stride         = (64, 64)

    entry.model_overrides = {"base_channels": 8, "depth": 1, "embedding_dim": 8}

    return entry


def test_build_dataset_config_reads_n_gaussians(tmp_path):
    loader  = RunLoader(tmp_path, logger=SimpleNamespace())
    payload = {
        "preprocessing_run_directory" : str(tmp_path),
        "parameters_path"             : str(tmp_path / "parameters.npy"),
        "split_regions"               : {
            "train" : {"azimuth_start": 0, "azimuth_end": 10, "range_start": 0, "range_end": 10},
            "val"   : {"azimuth_start": 10, "azimuth_end": 20, "range_start": 0, "range_end": 10},
            "test"  : {"azimuth_start": 20, "azimuth_end": 30, "range_start": 0, "range_end": 10},
        },
        "patch"            : {"size": [8, 8], "stride": [8, 8], "use_symmetric_padding": True},
        "secondary_labels" : None,
        "input_config"     : {
            "use_primary": True,  "primary_representation": "mag_only",
            "use_secondaries": False, "secondaries_representation": "mag_only",
            "use_interferograms": False, "interferograms_representation": "angle_only",
            "use_dem": False,
        },
        "output_config" : {
            "use_amplitude": True, "use_mu": True, "use_sigma": True,
            "output_strategies": {},
        },
        "batch_size"  : 4,
        "pin_memory"  : False,
        "n_gaussians" : 5,
    }

    config = loader._build_dataset_config(payload=payload, batch_size=None, num_workers=0)

    assert config.n_gaussians == 5


def test_image_ae_train_then_infer_end_to_end(test_data_dir, params_dir, tmp_path):
    entry = _entry_config(test_data_dir, params_dir, tmp_path)

    (_train_losses, _val_losses, best_val_loss), run_directory = TrainingPipeline(entry).run()

    assert np.isfinite(best_val_loss)
    assert (run_directory / "best_model.pt").is_file()
    assert (run_directory / "meta" / "dataset_creation_config.json").is_file()

    payload = json.loads((run_directory / "meta" / "dataset_creation_config.json").read_text())
    assert payload["n_gaussians"] == 5

    config = ImageAeInferenceConfig(
        run_directory = run_directory,
        output_subdir = "e2e",
        device        = "cpu",
        num_workers   = 0,
        save_plots    = False,
    )

    report_path = ImageAeInferencePipeline(config).run()
    output_dir  = run_directory / "inference" / "image_ae" / "e2e"

    assert report_path.is_file()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["split"] == "test"
    assert metrics["n_channels"] == 9
    assert np.isfinite(metrics["mse_mean"])
    assert np.isfinite(metrics["mse_mean_normalized"])
    assert len(metrics["channel_mse"]) == 9

    embeddings = np.load(output_dir / "embeddings.npy")
    assert embeddings.shape[1] == 8
    assert np.isfinite(embeddings).all()


def _profile_entry_config(test_data_dir, params_dir, tmp_path) -> ProfileAeEntryConfig:
    entry = ProfileAeEntryConfig(run_name="e2e_profile_ae", logdir=tmp_path)

    entry.paths.dataset_path    = test_data_dir
    entry.paths.parameters_path = params_dir / "parameters.npy"

    entry.training.epochs               = 1
    entry.training.validation_frequency = 1
    entry.training.batch_size           = 256
    entry.training.num_workers          = 0
    entry.training.scale_lr_with_batch  = False
    entry.training.train_azimuth        = (1000, 1400)
    entry.training.val_azimuth          = (1400, 1500)
    entry.training.test_azimuth         = (1500, 1600)

    entry.pixel_subsample = 0.005
    entry.model_overrides = {"hidden_dim": 64, "depth": 2, "embedding_dim": 8}

    return entry


def test_profile_ae_train_then_infer_end_to_end(test_data_dir, params_dir, tmp_path):
    entry = _profile_entry_config(test_data_dir, params_dir, tmp_path)

    (_train_losses, _val_losses, best_val_loss), run_directory = ProfileTrainingPipeline(entry).run()

    assert np.isfinite(best_val_loss)
    assert (run_directory / "best_model.pt").is_file()
    assert (run_directory / "meta" / "profile_dataset_config.json").is_file()

    config = ProfileAeInferenceConfig(
        run_directory = run_directory,
        output_subdir = "e2e",
        device        = "cpu",
        num_workers   = 0,
        save_plots    = False,
    )

    report_path = ProfileAeInferencePipeline(config).run()
    output_dir  = run_directory / "inference" / "profile_ae" / "e2e"

    assert report_path.is_file()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["split"] == "test"
    assert metrics["split_regions"] == [[1500, 1600, 500, 1000]]
    assert np.isfinite(metrics["mse_mean"])
    assert np.isfinite(metrics["mse_mean_normalized"])

    embeddings    = np.load(output_dir / "embeddings.npy")
    pixel_indices = np.load(output_dir / "pixel_indices.npy")

    assert embeddings.shape == (metrics["n_curves"], 8)
    assert np.isfinite(embeddings).all()

    assert pixel_indices.shape == (embeddings.shape[0],)
    assert pixel_indices.dtype == np.int64
    assert len(np.unique(pixel_indices)) == pixel_indices.shape[0]
    assert pixel_indices.min() >= 0
    assert pixel_indices.max() < 100 * 500
