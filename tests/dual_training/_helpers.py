"""Shared configuration builders for the dual training and dual inference tests."""

from __future__ import annotations

from configuration.dataset             import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.sar.gaussian import GaussianConfig
from configuration.training.backbone   import BackboneTrainerConfig
from models.dual                       import DUAL_CONFIG_REGISTRY
from tools.data.regions                import CropRegion

from tests.backbone_training._helpers import geometry_config


N_GAUSSIANS      = 5
SECONDARY_LABELS = ("CH01", "CH02")


def dual_dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    """Returns a dataset config over three 64x64 crop regions with magnitude and phase inputs."""
    input_config = InputConfig(
        use_primary        = True, primary_representation        = Representation.MAG_ONLY,
        use_secondaries    = True, secondaries_representation    = Representation.MAG_ONLY,
        use_derived = True, derived_representation = Representation.ANGLE_ONLY,
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


def dual_trainer_config(test_data_dir, params_dir, tmp_path) -> BackboneTrainerConfig:
    """Returns a one-epoch trainer config writing under the temporary path with the parameter L1 term enabled."""
    gaussian = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy")
    config   = BackboneTrainerConfig(gaussian=gaussian)

    config.io.logdir                     = str(tmp_path)
    config.training.epochs               = 1
    config.training.validation_frequency = 1
    config.resources.enabled             = False
    config.geometry                      = geometry_config()

    config.curriculum.complete.use_param_l1    = True
    config.curriculum.complete.weight_param_l1 = 1.0

    return config


def dual_model_config():
    """Returns a small dual_resunet config with narrow trunks and no dropout."""
    config = DUAL_CONFIG_REGISTRY["dual_resunet"]()

    config.params_features     = [8, 16]
    config.existence_features  = [8]
    config.existence_input     = ("derived",)
    config.params_overrides    = {"bottleneck_factor": 1, "dropout": 0.0}
    config.existence_overrides = {"bottleneck_factor": 1, "dropout": 0.0}

    return config
