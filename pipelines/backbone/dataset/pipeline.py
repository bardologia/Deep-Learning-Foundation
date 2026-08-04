from __future__ import annotations

from pathlib import Path
from typing  import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from configuration.dataset                      import DatasetConfig
from tools.data.regions                         import CropRegion
from pipelines.backbone.dataset.augmentation    import SpatialAugmenter
from pipelines.backbone.dataset.datasets        import MultiRegionDataset, PatchDataset
from pipelines.shared.dataset.loaders           import Loader
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.stats_computer  import StatsComputer
from pipelines.backbone.dataset.spatial         import Cropper, Patcher
from pipelines.backbone.dataset.metadata_writer import MetadataWriter
from pipelines.shared.dataset.dataset_spatial   import Layout
from tools.data.gaussians                       import GaussianAxis
from tools.monitoring.logger                    import Logger
from tools.sar                                  import GeometryField


class DatasetPipeline:
    def __init__(self, config : DatasetConfig, training_run_directory : Path, logger : Logger | None = None, seed : int = 0, height_axis_convention : str = "height", build_geometry_field : bool = False) -> None:
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)
        self.seed                   = int(seed)
        self.height_axis_convention = height_axis_convention

        log_dir = self.training_run_directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger(log_dir=str(log_dir), name="dataset_pipeline", level="INFO")

        config.split_regions.validate_disjoint()

        self.layout          = Layout(config.preprocessing_run_directory,  logger=self.logger, parameters_path=config.parameters_path)
        self.cropper         = Cropper(self.layout, config.split_regions,  logger=self.logger, secondary_labels=config.secondary_labels, input_config=config.input_config)
        self.metadata_writer = MetadataWriter(self.training_run_directory, logger=self.logger)
        self.geometry_field  = self._load_geometry_field() if build_geometry_field else None
        self.augmenter       = SpatialAugmenter(config.augmentation,       logger=self.logger, seed=self.seed, rot90_inactive=self.geometry_field is not None)

        if self.geometry_field is not None and config.augmentation.p_rot90 > 0.0:
            self.logger.warning("augmentation.p_rot90 > 0 has no effect: 90-degree rotations are skipped while a per-pixel geometry field is active.")

        if config.patch.size[0] != config.patch.size[1] and config.augmentation.p_rot90 > 0.0:
            raise ValueError(f"augmentation.p_rot90 > 0 is incompatible with a rectangular patch {tuple(config.patch.size)}: a 90-degree rotation swaps the patch axes and the rotated patch no longer matches the batch shape. Disable rot90 or use a square patch.")

        ic = config.input_config
        oc = config.output_config

        self.logger.section("[DatasetPipeline Initialized]")
        self.logger.kv_table(
            {
                "Pre-processing Run" : str(config.preprocessing_run_directory),
                "Training Run"       : str(self.training_run_directory),
                "Primary"            : f"use={ic.use_primary} rep={ic.primary_representation.value}",
                "Secondaries"        : f"use={ic.use_secondaries} rep={ic.secondaries_representation.value}",
                "Interferograms"     : f"use={ic.use_interferograms} rep={ic.interferograms_representation.value}",
                "DEM"                : f"use={ic.use_dem}",
                "Output Parameters"  : ",".join(oc.role_names),
                "Patch Size"         : config.patch.size,
                "Patch Stride"       : config.patch.stride,
                "Symmetric Padding"  : config.patch.use_symmetric_padding,
            },
            title="Dataset Creation",
        )

    @property
    def profile_length(self) -> int:
        return self.layout.profile_length

    def _load_geometry_field(self) -> GeometryField:
        path = Path(self.config.preprocessing_run_directory) / "meta" / GeometryField.FILENAME

        if not path.is_file():
            raise FileNotFoundError(f"A per-pixel physics-geometry loss term is active but {path} is missing; this dataset predates automatic geometry-field generation, re-run preprocessing for it.")

        field = GeometryField.load(path).subset(self.config.secondary_labels)
        field.validate_extent(self.layout.global_crop)

        self.logger.section("[Per-Pixel Geometry Field Loaded]")
        self.logger.kv_table(field.describe(), title="Geometry Field")
        self.logger.subsection(f"Height-axis convention : {self.height_axis_convention}")

        return field

    def _region_kz_field(self, region: CropRegion) -> Optional[np.ndarray]:
        if self.geometry_field is None:
            return None

        azimuth_slice, range_slice = region.local_slices(self.layout.global_crop)
        region_field               = self.geometry_field.slice(azimuth_slice, range_slice)

        return region_field.kz(self.height_axis_convention).astype(np.float32)

    def _build_dataset(self, split_name : str, normalizer : Optional[Normalizer] = None):
        regions  = self.config.split_regions.regions(split_name)
        parts    = []
        patchers = []

        for region in regions:
            arrays  = self.cropper.load_split(region)
            spatial = (region.azimuth_size, region.range_size)

            patcher = Patcher.build(
                spatial_size          = spatial,
                patch_size            = self.config.patch.size,
                stride                = self.config.patch.stride,
                use_symmetric_padding = self.config.patch.use_symmetric_padding,
            )

            part = PatchDataset(
                inputs           = arrays["inputs"],
                gt_parameters    = arrays["parameters"],
                grid             = patcher,
                input_config     = self.config.input_config,
                output_config    = self.config.output_config,
                split_name       = split_name,
                n_secondaries    = arrays["n_secondaries"],
                n_interferograms = arrays["n_interferograms"],
                n_primary        = arrays["n_primary"],
                normalizer       = normalizer,
                n_gaussians      = self.config.n_gaussians,
                augmenter        = self.augmenter,
                dem              = arrays["dem"] if self.config.input_config.use_dem else None,
                kz_field         = self._region_kz_field(region),
            )

            parts.append(part)
            patchers.append(patcher)

        self.logger.subsection(f"Split '{split_name}': {len(parts)} disjoint region(s), {sum(len(p) for p in parts)} patches total")

        if len(parts) == 1:
            return parts[0], patchers[0]

        return MultiRegionDataset(parts), patchers

    def _patch_grids(self, patcher):
        if isinstance(patcher, list):
            return [p.grid for p in patcher]
        return patcher.grid

    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, PatchDataset]]:

        train_ds, train_patcher = self._build_dataset("train")

        norm_stats = StatsComputer.compute(
            dataset          = train_ds,
            logger           = self.logger,
            input_config     = self.config.input_config,
            output_config    = self.config.output_config,
            n_secondaries    = train_ds.n_secondaries,
            n_interferograms = train_ds.n_interferograms,
            n_gaussians      = self.config.n_gaussians,
            normalization    = self.config.normalization,
        )

        norm_stats.save(self.training_run_directory / "meta")
        self.logger.info(f"Normalization stats saved: {self.training_run_directory / 'meta'}")

        normalizer          = Normalizer(norm_stats)
        train_ds.normalizer = normalizer

        val_ds,   val_patcher  = self._build_dataset("val",  normalizer=normalizer)
        test_ds,  test_patcher = self._build_dataset("test", normalizer=normalizer)

        train_loader, val_loader, test_loader = Loader.build(
            train_dataset   = train_ds,
            val_dataset     = val_ds,
            test_dataset    = test_ds,
            batch_size      = self.config.batch_size,
            num_workers     = self.config.num_workers,
            pin_memory      = self.config.pin_memory,
            shuffle_train   = self.config.shuffle_train,
            prefetch_factor = self.config.prefetch_factor,
            seed            = self.seed,
            logger          = self.logger,
        )

        self.metadata_writer.save_dataset_configuration(self.config)

        self.metadata_writer.save_crop_metadata(
            global_crop = self.layout.global_crop,
            splits      = dict(self.config.split_regions.items()),
        )

        self.metadata_writer.save_patch_metadata({
            "train" : self._patch_grids(train_patcher),
            "val"   : self._patch_grids(val_patcher),
            "test"  : self._patch_grids(test_patcher),
        })

        datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

        return train_loader, val_loader, test_loader, datasets


class BackboneDatasetPreparation:
    def __init__(self, dataset_config, trainer_config, run_directory, logger, seed, build_geometry_field: bool = False, height_axis_convention: str = "height") -> None:
        self.dataset_config         = dataset_config
        self.trainer_config         = trainer_config
        self.run_directory          = Path(run_directory)
        self.logger                 = logger
        self.seed                   = seed
        self.build_geometry_field   = build_geometry_field
        self.height_axis_convention = height_axis_convention

    def run(self):
        gaussian_cfg                    = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(self.dataset_config, self.run_directory, logger=self.logger, seed=self.seed, height_axis_convention=self.height_axis_convention, build_geometry_field=self.build_geometry_field)
        x_len            = dataset_pipeline.profile_length
        x_axis           = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, x_len)

        self.dataset_config.x_axis = x_axis

        train_loader, val_loader, test_loader, datasets = dataset_pipeline.run()

        return (train_loader, val_loader, test_loader), datasets, x_axis, x_len
