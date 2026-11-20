from __future__ import annotations

from pathlib import Path
from typing  import Tuple

import numpy as np
from torch.utils.data import DataLoader

from configuration.dataset                      import DatasetConfig
from pipelines.backbone.dataset.augmentation    import SpatialAugmenter
from pipelines.backbone.dataset.datasets        import MultiRegionDataset, PatchDataset
from pipelines.backbone.dataset.metadata_writer import MetadataWriter
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.spatial         import Patcher
from pipelines.backbone.dataset.stats           import Stats
from pipelines.backbone.dataset.stats_computer  import StatsComputer
from pipelines.shared.dataset.dataset_files     import DatasetFiles
from pipelines.shared.dataset.loaders           import Loader
from tools.monitoring.logger                    import Logger


class DatasetPipeline:

    def __init__(self, config: DatasetConfig, training_run_directory: Path, logger: Logger, seed: int = 0, resume: bool = False) -> None:
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)
        self.logger                 = logger
        self.seed                   = int(seed)
        self.resume                 = bool(resume)

        self.files           = DatasetFiles.read(config.dataset_directory)
        self.augmenter       = SpatialAugmenter(config.augmentation, logger=logger, seed=self.seed)
        self.metadata_writer = MetadataWriter(self.training_run_directory, logger=logger)

        self.logger.section("[Dataset Pipeline]")
        self.logger.kv_table({
            "Dataset"           : str(config.dataset_directory),
            "Grid"              : f"{self.files.rows} rows x {self.files.cols} cols",
            "Input channels"    : ", ".join(self.files.input_channels),
            "Target channels"   : ", ".join(self.files.target_channels),
            "Training run"      : str(self.training_run_directory),
            "Patch size"        : tuple(config.patch.size),
            "Patch stride"      : tuple(config.patch.stride),
            "Symmetric padding" : config.patch.use_symmetric_padding,
        })
        self.logger.metrics_table(config.split_regions.region_rows(), ["Split", "Crop", "Rows", "Cols"], title="Split regions")

    def _crop_split(self, split_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
        inputs  = self.files.inputs()
        targets = self.files.targets()

        return [(DatasetFiles.crop(inputs, region), DatasetFiles.crop(targets, region)) for region in self.config.split_regions.regions(split_name)]

    def _persist_stats(self, stats: Stats) -> None:
        stats_path = self.metadata_writer.metadata_directory / Stats.FILENAME

        if self.resume and stats_path.is_file():
            self.logger.ok(f"Normalization stats kept from the resumed run: {stats_path}")
            return

        stats.save(self.metadata_writer.metadata_directory)
        self.logger.ok(f"Normalization stats saved: {stats_path}")

    def _build_dataset(self, split_name: str, crops: list[tuple[np.ndarray, np.ndarray]], normalizer: Normalizer) -> MultiRegionDataset:
        augmenter = self.augmenter if split_name == "train" else None
        parts     = []

        for inputs, targets in crops:
            patcher = Patcher.build(
                spatial_size          = tuple(inputs.shape[-2:]),
                patch_size            = self.config.patch.size,
                stride                = self.config.patch.stride,
                use_symmetric_padding = self.config.patch.use_symmetric_padding,
            )

            parts.append(PatchDataset(inputs, targets, patcher, normalizer, count_once=split_name != "train", augmenter=augmenter))

        return MultiRegionDataset(parts)

    def _log_splits(self, datasets: dict[str, MultiRegionDataset]) -> None:
        rows = []

        for name, dataset in datasets.items():
            rows.append({
                "Split"        : name,
                "Regions"      : len(dataset.parts),
                "Grids"        : ", ".join(f"{part.patcher.grid.n_v} x {part.patcher.grid.n_h}" for part in dataset.parts),
                "Patches"      : len(dataset),
                "Pixel weight" : "1 / coverage, once per pixel" if dataset.parts[0].count_once else "1 per covering patch",
                "Memory"       : f"{sum(part.nbytes for part in dataset.parts) / 1e6:.1f} MB",
            })

        self.logger.section("[Dataset Splits]")
        self.logger.metrics_table(rows, ["Split", "Regions", "Grids", "Patches", "Pixel weight", "Memory"], title="Padding weighs 0 in every split")

    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, MultiRegionDataset], Normalizer]:
        train_crops = self._crop_split("train")
        stats       = StatsComputer.compute(train_crops, self.files.input_channels, self.files.target_channels, self.config.normalization, self.seed, self.logger)

        self._persist_stats(stats)

        normalizer = Normalizer(stats)
        datasets   = {
            "train" : self._build_dataset("train", train_crops, normalizer),
            "val"   : self._build_dataset("val",   self._crop_split("val"),  normalizer),
            "test"  : self._build_dataset("test",  self._crop_split("test"), normalizer),
        }

        train_loader, val_loader, test_loader = Loader.build(
            train_dataset   = datasets["train"],
            val_dataset     = datasets["val"],
            test_dataset    = datasets["test"],
            batch_size      = self.config.batch_size,
            num_workers     = self.config.num_workers,
            prefetch_factor = self.config.prefetch_factor,
            seed            = self.seed,
            logger          = self.logger,
        )

        self.metadata_writer.save_dataset_configuration(self.config)
        self.metadata_writer.save_patch_metadata({name: [part.patcher.grid for part in dataset.parts] for name, dataset in datasets.items()})
        self._log_splits(datasets)

        return train_loader, val_loader, test_loader, datasets, normalizer
