from __future__ import annotations

import json

import numpy as np
import pytest

from configuration.dataset                     import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.inference                   import InferenceConfig
from configuration.sar.gaussian_config         import GaussianConfig
from configuration.training                    import OverfitCheckConfig
from configuration.training.backbone           import BackboneTrainerConfig
from models.dual                               import DUAL_CONFIG_REGISTRY
from pipelines.backbone.inference.pipeline     import InferencePipeline
from pipelines.dual.inference.pipeline         import DUAL_INFERENCE_COMPONENTS
from pipelines.dual.training.pipeline          import DualTrainingPipeline
from pipelines.shared.inference.run_classifier import RunClassifier, RunType
from tools.data.regions                        import CropRegion

from tests.backbone_training._helpers import geometry_config


pytestmark = [pytest.mark.real_data, pytest.mark.slow, pytest.mark.usefixtures("force_cpu")]

N_GAUSSIANS      = 5
SECONDARY_LABELS = ("FL01_PS04", "FL01_PS06")


def _dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    input_config = InputConfig(
        use_primary        = True, primary_representation        = Representation.MAG_ONLY,
        use_secondaries    = True, secondaries_representation    = Representation.MAG_ONLY,
        use_interferograms = True, interferograms_representation = Representation.ANGLE_ONLY,
    )

    splits = SplitRegions(
        train = CropRegion(1000, 1064, 500, 564),
        val   = CropRegion(1064, 1128, 500, 564),
        test  = CropRegion(1128, 1192, 500, 564),
    )

    return DatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = splits,
        secondary_labels            = SECONDARY_LABELS,
        patch                       = PatchConfig(size=(32, 32), stride=(32, 32)),
        input_config                = input_config,
        batch_size                  = 4,
        num_workers                 = 0,
        n_gaussians                 = N_GAUSSIANS,
    )


def _trainer_config(test_data_dir, params_dir, tmp_path) -> BackboneTrainerConfig:
    gaussian = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy")
    config   = BackboneTrainerConfig(gaussian=gaussian)

    config.io.logdir                     = str(tmp_path)
    config.io.writer                     = None
    config.training.epochs               = 1
    config.training.validation_frequency = 1
    config.resources.enabled             = False
    config.geometry                      = geometry_config()

    config.curriculum.complete.use_param_l1    = True
    config.curriculum.complete.weight_param_l1 = 1.0

    return config


def _model_config():
    config = DUAL_CONFIG_REGISTRY["dual_resunet"]()

    config.params_features     = [8, 16]
    config.existence_features  = [8]
    config.existence_input     = ("ifg",)
    config.params_overrides    = {"bottleneck_factor": 1, "dropout": 0.0}
    config.existence_overrides = {"bottleneck_factor": 1, "dropout": 0.0}

    return config


def test_dual_train_then_infer_end_to_end(test_data_dir, params_dir, tmp_path):
    pipeline = DualTrainingPipeline(
        trainer_config = _trainer_config(test_data_dir, params_dir, tmp_path),
        dataset_config = _dataset_config(test_data_dir, params_dir),
        backbone_name  = "dual_resunet",
        model_config   = _model_config(),
        seed           = 0,
        run_name       = "e2e_dual",
        overfit_check  = OverfitCheckConfig(enabled=False),
    )

    train_losses, val_losses, best_val = pipeline.run(probe_config=None)
    run_directory = pipeline.run_metadata.run_directory

    assert len(train_losses) == 1
    assert (run_directory / "best_model.pt").is_file()
    assert not (run_directory / "meta" / "model_config.json").exists()
    assert RunClassifier.classify(run_directory) == RunType.DUAL

    model_payload = json.loads((run_directory / "meta" / "dual_model_config.json").read_text())
    assert model_payload["model_name"]                   == "dual_resunet"
    assert model_payload["config"]["in_channels"]        == 5
    assert model_payload["config"]["params_channels"]    == [0, 1, 2, 3, 4]
    assert model_payload["config"]["existence_channels"] == [3, 4]

    run_summary = json.loads((run_directory / "meta" / "run_summary.json").read_text())
    assert run_summary["model_name"]   == "dual_resunet"
    assert run_summary["in_channels"]  == 5
    assert run_summary["out_channels"] == 3 * N_GAUSSIANS

    config = InferenceConfig(
        run_directory            = run_directory,
        output_subdir            = "e2e",
        device                   = "cpu",
        split                    = "test",
        num_workers              = 0,
        cpu_workers              = 2,
        save_plots               = False,
        save_animations          = False,
        save_cubes               = True,
        compute_reduced          = False,
        compute_data_consistency = True,
    )

    report_path = InferencePipeline(config, components=DUAL_INFERENCE_COMPONENTS).run()

    output_dir = run_directory / "inference" / "e2e"
    assert report_path == output_dir / "report.md"
    assert report_path.is_file()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["split"] == "test"
    for key in ("curve_mse_gt", "overall_r2_gt", "pixel_mse_gt_mean", "active_frac_gt", "matched_recall"):
        assert np.isfinite(metrics[key]), key

    cubes = output_dir / "cubes"
    assert np.load(cubes / "params_pred.npy").shape == (3 * N_GAUSSIANS, 64, 64)
    assert np.load(cubes / "params_gt.npy").shape   == (3 * N_GAUSSIANS, 64, 64)

    assert "dual_resunet" in report_path.read_text()


def test_dual_training_rejects_a_dem_bearing_input_stack(test_data_dir, params_dir, tmp_path):
    dataset_config                      = _dataset_config(test_data_dir, params_dir)
    dataset_config.input_config.use_dem = True

    with pytest.raises(ValueError, match="never uses the DEM"):
        DualTrainingPipeline(
            trainer_config = _trainer_config(test_data_dir, params_dir, tmp_path),
            dataset_config = dataset_config,
            backbone_name  = "dual_resunet",
            model_config   = _model_config(),
            seed           = 0,
            run_name       = "e2e_dual_dem",
            overfit_check  = OverfitCheckConfig(enabled=False),
        )
