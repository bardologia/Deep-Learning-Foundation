from __future__ import annotations

from pathlib import Path
from typing  import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from configuration.dataset                               import ProfileDatasetConfig
from pipelines.profile_autoencoder.dataset.augmentation  import ProfileAugmenter
from pipelines.profile_autoencoder.dataset.datasets      import ProfileDataset
from pipelines.shared.dataset.loaders                    import Loader
from tools.data.gaussians                                import GaussianAxis
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStatsComputer
from pipelines.profile_autoencoder.dataset.splitting     import ParameterCropper
from pipelines.shared.dataset.dataset_spatial            import Layout
from tools.monitoring.logger                             import Logger


class ProfileDatasetPipeline:
    def __init__(self, config: ProfileDatasetConfig, training_run_directory: Path, logger: Logger | None = None, seed: int = 0) -> None:
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)
        self.seed                   = int(seed)

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
        x_len  = self.cropper.profile_length()
        x_axis = GaussianAxis.build(self.config.x_min, self.config.x_max, x_len)

        return x_axis, x_len

    def _build_dataset(self, split_name: str, x_axis: np.ndarray, normalizer: Optional[ProfileNormalizer] = None, augmenter: Optional[ProfileAugmenter] = None) -> ProfileDataset:
        param_arrays = self.cropper.load_split(split_name)

        return ProfileDataset(
            param_arrays    = param_arrays,
            x_axis          = x_axis,
            n_gaussians     = self.config.n_gaussians,
            split_name      = split_name,
            amp_zero_thr    = self.config.amp_zero_thr,
            pixel_subsample = self.config.pixel_subsample,
            keep_empty_frac = self.config.keep_empty_frac,
            seed            = self.seed,
            normalizer      = normalizer,
            augmenter       = augmenter,
            logger          = self.logger,
        )

    def _fit_normalizer(self, train_ds: ProfileDataset) -> ProfileNormalizer:
        stats = ProfileStatsComputer.compute(train_ds, self.logger, max_samples=self.config.stats_max_samples)
        stats.save(self.training_run_directory / "meta")

        return ProfileNormalizer(stats)

    def run(self) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], dict[str, ProfileDataset], np.ndarray, int, ProfileNormalizer]:
        x_axis, x_len = self._build_axis()

        train_ds   = self._build_dataset("train", x_axis)
        normalizer = self._fit_normalizer(train_ds)

        train_ds.normalizer = normalizer
        train_ds.augmenter  = self.augmenter

        val_ds  = self._build_dataset("val",  x_axis, normalizer=normalizer)
        test_ds = self._build_dataset("test", x_axis, normalizer=normalizer)

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

        return (train_loader, val_loader, test_loader), datasets, x_axis, x_len, normalizer
