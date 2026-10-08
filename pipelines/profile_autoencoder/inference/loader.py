"""Loading of a trained profile-autoencoder run for inference.

Rebuilds the dataset, normaliser and model from the persisted run metadata and
checks that the profile geometry still matches what the run was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import numpy as np
from torch.utils.data import DataLoader

from configuration.inference.profile_autoencoder         import ProfileAeInferenceConfig
from models.profile_autoencoder                          import get_profile_autoencoder
from pipelines.shared.dataset.dataset_spatial            import Layout
from pipelines.profile_autoencoder.dataset.datasets      import ProfileDataset
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from tools.data.gaussians                                import GaussianAxis
from pipelines.profile_autoencoder.dataset.splitting     import ParameterCropper
from pipelines.shared.config.config_persistence          import ProfileAutoencoderConfigIO, ProfileDatasetConfigIO
from pipelines.shared.inference.run_artifacts            import RunArtifactLoader
from tools.data.regions                                  import CropRegion


@dataclass
class ProfileAeRun:
    """Everything inference needs from a trained profile-autoencoder run.

    Attributes:
        model                       : The loaded autoencoder in eval mode.
        ae_name                     : Registry name of the autoencoder architecture.
        embedding_dim               : Latent dimensionality.
        x_axis                      : Z axis of shape (H,), in metres.
        normalizer                  : Log1p standardisation fitted during training.
        dataset                     : Dataset of the requested split.
        loader                      : Non-shuffling loader over that dataset.
        split_name                  : Split being inferred.
        checkpoint_meta             : Best epoch and validation loss of the checkpoint.
        preprocessing_run_directory : Preprocessing run the parameters came from.
        split_regions               : Crop regions covered by the split.
        activity_threshold          : Amplitude above which a pixel counts as active, as recorded by the parameter run.
    """

    model                       : object
    ae_name                     : str
    embedding_dim               : int
    z_axis                      : np.ndarray
    normalizer                  : ProfileNormalizer
    dataset                     : ProfileDataset
    loader                      : DataLoader
    split_name                  : str
    checkpoint_meta             : dict
    preprocessing_run_directory : Path
    split_regions               : list[CropRegion]
    activity_threshold          : float


class ProfileAeRunLoader(RunArtifactLoader):
    """Reconstructs a `ProfileAeRun` from a run directory on disk."""

    def _read_run_summary(self) -> dict:
        """Reads run_summary.json and rejects runs of any other family.

        Returns:
            The parsed run summary.

        Raises:
            ValueError : If the run was not produced by profile-AE training.
        """
        summary = self._read_json("run_summary.json")

        if str(summary.get("model_name")) != "profile_ae":
            raise ValueError(f"Run '{self.run_directory}' is not a standalone profile-autoencoder run (model_name='{summary.get('model_name')}'). Use 'infer' for backbone and JEPA runs.")

        return summary

    def _build_dataset(self, config: ProfileAeInferenceConfig, dataset_config, normalizer: ProfileNormalizer):
        """Rebuilds the split dataset from the persisted dataset configuration.

        Args:
            config         : Inference configuration selecting the split and subsampling.
            dataset_config : Dataset configuration persisted at training time.
            normalizer     : Normalisation applied to each curve.

        Returns:
            Tuple of the dataset, the z axis of shape (H,) in metres and
            its length.
        """
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = ParameterCropper(layout, dataset_config.split_regions, logger=self.logger)

        param_arrays = cropper.load_split(config.split)
        z_len        = cropper.profile_length()
        z_axis       = GaussianAxis.build(dataset_config.x_min, dataset_config.x_max, z_len)

        dataset = ProfileDataset(
            param_arrays       = param_arrays,
            z_axis             = z_axis,
            n_gaussians        = dataset_config.n_gaussians,
            split_name         = config.split,
            activity_threshold = dataset_config.activity_threshold,
            pixel_subsample    = config.pixel_subsample,
            keep_empty_frac    = config.keep_empty_frac,
            seed               = config.seed,
            normalizer         = normalizer,
            augmenter          = None,
            logger             = self.logger,
        )

        return dataset, z_axis, z_len

    def _build_model(self, z_len: int, device: str):
        """Instantiates the persisted autoencoder architecture on a device.

        Args:
            z_len  : Profile length in z bins.
            device : Torch device string.

        Returns:
            Tuple of the model on the device and its registry name.
        """
        ae_cfg, ae_name       = ProfileAutoencoderConfigIO.load(self.meta_directory)
        ae_cfg.profile_length = z_len

        model, _ = get_profile_autoencoder(ae_name, ae_cfg)

        return model.to(device), ae_name

    @staticmethod
    def _describe(ckpt_path: Path, embedding_dim: int, z_len: int, dataset: ProfileDataset, config: ProfileAeInferenceConfig, batch_size: int, activity_threshold: float) -> dict:
        """Returns the model table rows: checkpoint, geometry, scored curve set and the settings that selected it.

        Args:
            ckpt_path          : Checkpoint the weights were restored from.
            embedding_dim      : Latent dimensionality.
            z_len              : Profile length in z bins.
            dataset            : Rebuilt split dataset, exposing the kept active and empty pixel counts.
            config             : Inference configuration with the sampling settings and the optional batch-size override.
            batch_size         : Effective batch size of the loader.
            activity_threshold : Amplitude above which a pixel counts as active, as recorded by the training run.

        Returns:
            Ordered mapping of row label to value.
        """
        return {
            "Checkpoint"         : ckpt_path,
            "Embedding dim"      : embedding_dim,
            "Profile length"     : z_len,
            "Split"              : config.split,
            "Curves"             : f"{len(dataset):,} ({dataset.n_kept_active:,} active + {dataset.n_kept_empty:,} empty)",
            "Activity threshold" : f"{activity_threshold:g} (from the training run's {ProfileDatasetConfigIO.FILENAME})",
            "Pixel subsample"    : f"{config.pixel_subsample:g} of the active pixels",
            "Keep empty frac"    : f"{config.keep_empty_frac:g} of the empty pixels",
            "Selection seed"     : config.seed,
            "Batch size"         : f"{batch_size} ({'inference override' if config.batch_size is not None else 'training batch size'})",
        }

    def load(self, *, config: ProfileAeInferenceConfig, device: str) -> ProfileAeRun:
        """Loads dataset, model and checkpoint and assembles the run bundle.

        Args:
            config : Inference configuration.
            device : Torch device string.

        Returns:
            The assembled `ProfileAeRun`.

        Raises:
            ValueError : If the rebuilt profile length differs from the length the run was trained on.
        """
        self.logger.section("[Profile AE Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        summary        = self._read_run_summary()
        embedding_dim  = int(summary["out_channels"])
        dataset_config = ProfileDatasetConfigIO.load(self.meta_directory)
        normalizer     = ProfileNormalizer(ProfileStats.load(self.meta_directory, self.logger))

        dataset, z_axis, z_len = self._build_dataset(config, dataset_config, normalizer)

        if z_len != int(summary["x_axis_length"]):
            raise ValueError(f"Profile length mismatch: dataset gives {z_len} bins but the run was trained on {summary['x_axis_length']}. The persisted dataset no longer matches the training dataset.")

        model, ae_name = self._build_model(z_len, device)
        ckpt_path      = self.run_directory / config.checkpoint_name

        ckpt, ckpt_x_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)

        model.load_state_dict(ckpt["params"])
        model.eval()

        self._validate_x_axis(ckpt_x_axis, np.asarray(z_axis, dtype=np.float32), ckpt_path)

        batch_size = config.batch_size if config.batch_size is not None else dataset_config.batch_size
        loader     = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        regions = dataset_config.split_regions.regions(config.split)

        self.logger.section(f"[Profile AE Model]  : '{ae_name}'")
        self.logger.kv_table(self._describe(ckpt_path, embedding_dim, z_len, dataset, config, batch_size, dataset_config.activity_threshold))

        return ProfileAeRun(
            model                       = model,
            ae_name                     = ae_name,
            embedding_dim               = embedding_dim,
            z_axis                      = z_axis,
            normalizer                  = normalizer,
            dataset                     = dataset,
            loader                      = loader,
            split_name                  = config.split,
            checkpoint_meta             = ckpt_meta,
            preprocessing_run_directory = Path(dataset_config.preprocessing_run_directory),
            split_regions               = regions,
            activity_threshold          = dataset_config.activity_threshold,
        )
