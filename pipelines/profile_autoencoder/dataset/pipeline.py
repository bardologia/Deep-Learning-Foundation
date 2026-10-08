"""Assembly of the profile-autoencoder train, validation and test loaders.

Crops the extracted parameter maps per split region, fits the normalisation
statistics on the training split alone, and wraps the datasets into loaders.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing  import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from configuration.dataset                               import ProfileDatasetConfig
from pipelines.profile_autoencoder.dataset.augmentation  import ProfileAugmenter
from pipelines.profile_autoencoder.dataset.datasets      import ProfileDataset
from pipelines.shared.dataset.loaders                    import Loader
from tools.data.gaussians                                import GaussianAxis
from tools.data.io                                       import FileIO
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats, ProfileStatsComputer
from pipelines.profile_autoencoder.dataset.splitting     import ParameterCropper
from pipelines.shared.dataset.dataset_spatial            import Layout
from tools.monitoring.logger                             import Logger


class ProfileDatasetPipeline:
    """Builds the profile datasets and loaders for one training run.

    Attributes:
        config                 : Dataset configuration driving cropping and loader settings.
        training_run_directory : Run directory logs and statistics are written under.
        seed                   : Seed shared by augmentation, pixel selection and loader shuffling.
        resume                 : Whether this run resumes an existing one, in which case the stored profile statistics must match instead of being overwritten.
        layout                 : Spatial layout of the preprocessing run.
        cropper                : Splits the parameter maps into per-split regions.
        augmenter              : Augmentation applied to the training split.
    """

    def __init__(self, config: ProfileDatasetConfig, training_run_directory: Path, logger: Logger | None = None, seed: int = 0, resume: bool = False) -> None:
        """Prepares the layout, cropper and augmenter for a training run.

        Args:
            config                 : Dataset configuration.
            training_run_directory : Directory receiving logs and statistics.
            logger                 : Logger to reuse; a run-scoped one is created when omitted.
            seed                   : Seed shared by augmentation, selection and shuffling.
            resume                 : Whether this run resumes an existing one, in which case the stored profile normalization statistics must match instead of being overwritten.
        """
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)
        self.seed                   = int(seed)
        self.resume                 = bool(resume)

        log_dir = self.training_run_directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger(log_dir=str(log_dir), name="profile_dataset_pipeline", level="INFO")

        self.layout    = Layout(config.preprocessing_run_directory, logger=self.logger, parameters_path=config.parameters_path)
        self.cropper   = ParameterCropper(self.layout, config.split_regions, logger=self.logger)
        self.augmenter = ProfileAugmenter(config.augmentation, logger=self.logger, seed=self.seed)

        self.logger.section("[ProfileDatasetPipeline Initialized]")
        self.logger.kv_table({
            "Pre-processing Run" : str(config.preprocessing_run_directory),
            "Training Run"       : str(self.training_run_directory),
            "Gaussians"          : config.n_gaussians,
            "Pixel subsample"    : config.pixel_subsample,
            "Keep empty frac"    : config.keep_empty_frac,
        })

    def _build_axis(self) -> Tuple[np.ndarray, int]:
        """Returns the z axis of shape (H,) in metres and its length."""
        z_len  = self.cropper.profile_length()
        z_axis = GaussianAxis.build(self.config.x_min, self.config.x_max, z_len)

        return z_axis, z_len

    def _build_dataset(self, split_name: str, z_axis: np.ndarray, normalizer: Optional[ProfileNormalizer] = None, augmenter: Optional[ProfileAugmenter] = None) -> ProfileDataset:
        """Loads one split's parameter regions and wraps them in a dataset.

        Args:
            split_name : "train", "val" or "test".
            z_axis     : Z axis of shape (H,), in metres.
            normalizer : Normalisation applied to each curve.
            augmenter  : Augmentation applied when the split is "train".

        Returns:
            The dataset for the requested split.
        """
        param_arrays = self.cropper.load_split(split_name)

        return ProfileDataset(
            param_arrays       = param_arrays,
            z_axis             = z_axis,
            n_gaussians        = self.config.n_gaussians,
            split_name         = split_name,
            activity_threshold = self.config.activity_threshold,
            pixel_subsample    = self.config.pixel_subsample,
            keep_empty_frac    = self.config.keep_empty_frac,
            seed               = self.seed,
            normalizer         = normalizer,
            augmenter          = augmenter,
            logger             = self.logger,
        )

    def _persist_stats(self, stats: ProfileStats) -> None:
        """Saves the fitted profile statistics, or checks them against a resumed run.

        Args:
            stats : Statistics fitted on the train split.

        Raises:
            ValueError : If the run resumes and the fitted statistics differ from those the restored checkpoint was trained under.
        """
        meta_directory = self.training_run_directory / "meta"
        stats_path     = meta_directory / ProfileStats.FILENAME

        if self.resume and stats_path.is_file():
            stored = FileIO.load_json(stats_path)
            fresh  = json.loads(json.dumps(stats.payload()))

            if stored != fresh:
                raise ValueError(f"Resuming this run would overwrite {stats_path} with different profile normalization statistics; the restored checkpoint was trained under the stored ones. Start a fresh run or drop the overrides that changed the fitted statistics.")

            self.logger.info(f"Profile normalization stats match the resumed run: {stats_path}")
            return

        stats.save(meta_directory)
        self.logger.ok(f"Profile normalization stats saved: {meta_directory}")

    def _fit_normalizer(self, train_ds: ProfileDataset) -> ProfileNormalizer:
        """Fits the normalisation statistics on the training split and persists them.

        Args:
            train_ds : Training dataset, sampled to estimate location and scale.

        Returns:
            A normaliser built from the fitted statistics.

        Raises:
            ValueError : If the run resumes and the fitted statistics differ from those the restored checkpoint was trained under.
        """
        stats = ProfileStatsComputer.compute(train_ds, self.logger, max_samples=self.config.stats_max_samples)
        self._persist_stats(stats)

        return ProfileNormalizer(stats)

    def run(self) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], dict[str, ProfileDataset], np.ndarray, int, ProfileNormalizer]:
        """Builds every split, fits the normaliser and returns the loaders.

        Returns:
            Tuple of the train, validation and test loaders, the datasets keyed
            by split name, the z axis of shape (H,) in metres, its
            length, and the fitted normaliser.
        """
        z_axis, z_len = self._build_axis()

        train_ds   = self._build_dataset("train", z_axis)
        normalizer = self._fit_normalizer(train_ds)

        train_ds.normalizer = normalizer
        train_ds.augmenter  = self.augmenter

        val_ds  = self._build_dataset("val",  z_axis, normalizer=normalizer)
        test_ds = self._build_dataset("test", z_axis, normalizer=normalizer)

        train_loader, val_loader, test_loader = Loader.build(
            train_dataset   = train_ds,
            val_dataset     = val_ds,
            test_dataset    = test_ds,
            batch_size      = self.config.batch_size,
            num_workers     = self.config.num_workers,
            prefetch_factor = self.config.prefetch_factor,
            pin_memory      = self.config.pin_memory,
            shuffle_train   = self.config.shuffle_train,
            seed            = self.seed,
            logger          = self.logger,
            section_label   = "[Profile Loaders]",
        )

        datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

        self.logger.kv_table({
            "Train Samples" : len(train_ds),
            "Val Samples"   : len(val_ds),
            "Test Samples"  : len(test_ds),
            "Axis Length"   : z_len,
        })

        return (train_loader, val_loader, test_loader), datasets, z_axis, z_len, normalizer
