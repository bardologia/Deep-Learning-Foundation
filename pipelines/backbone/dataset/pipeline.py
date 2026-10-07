"""Dataset construction for backbone training runs.

DatasetPipeline crops the preprocessing artifacts per split, cuts them into patches,
fits the normalization on the train split, builds the dataloaders and writes the
dataset metadata. BackboneDatasetPreparation wraps it for the training entry points
and resolves the Gaussian profile axis.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing  import Optional, Tuple

from torch.utils.data import DataLoader

from configuration.dataset                      import DatasetConfig
from tools.data.io                              import FileIO
from pipelines.backbone.dataset.augmentation    import SpatialAugmenter
from pipelines.backbone.dataset.datasets        import MultiRegionDataset, PatchDataset
from pipelines.shared.dataset.loaders           import Loader
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.stats           import Stats
from pipelines.backbone.dataset.stats_computer  import StatsComputer
from pipelines.backbone.dataset.spatial         import Cropper, Patcher
from pipelines.backbone.dataset.metadata_writer import MetadataWriter
from pipelines.shared.dataset.dataset_spatial   import Layout
from tools.data.gaussians                       import GaussianAxis
from tools.monitoring.logger                    import Logger


class DatasetPipeline:
    """Builds the train, validation and test dataloaders of a backbone training run.

    Attributes:
        config                 : Dataset configuration driving crops, patching, channels and normalization.
        training_run_directory : Run directory receiving the stats and metadata files.
        activity_threshold     : Amplitude above which a Gaussian slot counts as active when the output statistics are fitted, as recorded by the label run.
        layout                 : Layout of the preprocessing artifacts this dataset is cut from.
        cropper                : Loader of the per-split crops out of those artifacts.
        metadata_writer        : Writer of the dataset, crop and patch metadata.
        augmenter              : Augmenter applied to the train split.
    """

    def __init__(self, config : DatasetConfig, training_run_directory : Path, activity_threshold : float, logger : Logger | None = None, seed : int = 0, resume : bool = False) -> None:
        """Validates the split geometry and wires the crop, metadata and augmentation components.

        Args:
            config                 : Dataset configuration of the run.
            training_run_directory : Directory the run writes its metadata into.
            activity_threshold     : Amplitude above which a Gaussian slot counts as active when the output statistics are fitted, as recorded by the label run.
            logger                 : Logger to use; a run-local dataset_pipeline logger is created when omitted.
            seed                   : Seed of the augmentation generator and of the dataloader shuffling.
            resume                 : Whether this run resumes an existing one, in which case the stored normalization statistics must match instead of being overwritten.

        Raises:
            ValueError : If 90-degree rotations are enabled together with a rectangular patch.
        """
        self.config                 = config
        self.training_run_directory = Path(training_run_directory)
        self.activity_threshold     = float(activity_threshold)
        self.seed                   = int(seed)
        self.resume                 = bool(resume)

        log_dir = self.training_run_directory / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger(log_dir=str(log_dir), name="dataset_pipeline", level="INFO")

        config.split_regions.validate_disjoint()

        self.layout          = Layout(config.preprocessing_run_directory,  logger=self.logger, parameters_path=config.parameters_path)
        self.cropper         = Cropper(self.layout, config.split_regions,  logger=self.logger, secondary_labels=config.secondary_labels, input_config=config.input_config)
        self.metadata_writer = MetadataWriter(self.training_run_directory, logger=self.logger)
        self.augmenter       = SpatialAugmenter(config.augmentation,       logger=self.logger, seed=self.seed)

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
                "Derived"            : f"use={ic.use_derived} rep={ic.derived_representation.value}",
                "Aux"                : f"use={ic.use_aux}",
                "Output Parameters"  : ",".join(oc.role_names),
                "Activity Threshold" : f"{self.activity_threshold:g}",
                "Patch Size"         : config.patch.size,
                "Patch Stride"       : config.patch.stride,
                "Symmetric Padding"  : config.patch.use_symmetric_padding,
            },
            title="Dataset Creation",
        )

    @property
    def profile_length(self) -> int:
        """Returns the number of z bins of the preprocessed profiles."""
        return self.layout.profile_length

    def _build_dataset(self, split_name : str, normalizer : Optional[Normalizer] = None):
        """Builds the dataset of one split over its disjoint crop regions.

        Args:
            split_name : Split to build, one of "train", "val" or "test".
            normalizer : Normalizer to attach; left unset for the train split, which is built before the statistics are fitted.

        Returns:
            A (dataset, patcher) pair. With a single region the dataset is a PatchDataset
            and the second element its Patcher; with several regions it is a
            MultiRegionDataset and a list of Patchers, one per region.
        """
        regions  = self.config.split_regions.regions(split_name)
        parts    = []
        patchers = []

        for region in regions:
            arrays  = self.cropper.load_split(region)
            spatial = (region.row_size, region.col_size)

            patcher = Patcher.build(
                spatial_size          = spatial,
                patch_size            = self.config.patch.size,
                stride                = self.config.patch.stride,
                use_symmetric_padding = self.config.patch.use_symmetric_padding,
            )

            part = PatchDataset(
                inputs        = arrays["inputs"],
                gt_parameters = arrays["parameters"],
                grid          = patcher,
                input_config  = self.config.input_config,
                output_config = self.config.output_config,
                split_name    = split_name,
                n_secondaries = arrays["n_secondaries"],
                n_derived     = arrays["n_derived"],
                n_primary     = arrays["n_primary"],
                normalizer    = normalizer,
                n_gaussians   = self.config.n_gaussians,
                augmenter     = self.augmenter,
                aux           = arrays["aux"] if self.config.input_config.use_aux else None,
            )

            parts.append(part)
            patchers.append(patcher)

        self.logger.subsection(f"Split '{split_name}': {len(parts)} disjoint region(s), {sum(len(p) for p in parts)} patches total")

        if len(parts) == 1:
            return parts[0], patchers[0]

        return MultiRegionDataset(parts), patchers

    def _persist_stats(self, norm_stats) -> None:
        """Saves the fitted normalization statistics, or checks them against a resumed run.

        Args:
            norm_stats : Statistics fitted on the train split.

        Raises:
            ValueError : If the run resumes and the fitted statistics differ from those the restored checkpoint was trained under.
        """
        meta_directory = self.training_run_directory / "meta"
        stats_path     = meta_directory / Stats.FILENAME

        if self.resume and stats_path.is_file():
            stored = FileIO.load_json(stats_path)
            fresh  = json.loads(json.dumps(norm_stats.payload()))

            if stored != fresh:
                raise ValueError(f"Resuming this run would overwrite {stats_path} with different normalization statistics; the restored checkpoint was trained under the stored ones. Start a fresh run or drop the overrides that changed the fitted statistics.")

            self.logger.info(f"Normalization stats match the resumed run: {stats_path}")
            return

        norm_stats.save(meta_directory)
        self.logger.info(f"Normalization stats saved: {meta_directory}")

    def _patch_grids(self, patcher):
        """Returns the GridInfo of one Patcher, or the list of grids of several."""
        if isinstance(patcher, list):
            return [p.grid for p in patcher]
        return patcher.grid

    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, PatchDataset]]:
        """Builds the splits, fits the normalization and writes the dataset metadata.

        The train split is built first and its statistics are fitted on it alone, then
        shared with the validation and test splits.

        Returns:
            A tuple of the train, validation and test dataloaders and a dictionary of the
            three datasets keyed by split name.
        """

        train_ds, train_patcher = self._build_dataset("train")

        norm_stats = StatsComputer.compute(
            dataset            = train_ds,
            logger             = self.logger,
            input_config       = self.config.input_config,
            output_config      = self.config.output_config,
            n_secondaries      = train_ds.n_secondaries,
            n_derived          = train_ds.n_derived,
            n_gaussians        = self.config.n_gaussians,
            activity_threshold = self.activity_threshold,
            normalization      = self.config.normalization,
        )

        self._persist_stats(norm_stats)

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
    """Runs the dataset pipeline for a training entry point and resolves the Gaussian axis.

    Attributes:
        dataset_config : Dataset configuration, mutated with the Gaussian count and z-axis.
        trainer_config : Trainer configuration supplying the Gaussian bounds, the slot count and the training resume flag.
        run_directory  : Training run directory.
    """

    def __init__(self, dataset_config, trainer_config, run_directory, logger, seed) -> None:
        """Stores the configurations and run context for the preparation step.

        Args:
            dataset_config : Dataset configuration of the run.
            trainer_config : Trainer configuration carrying the Gaussian settings and the training resume flag.
            run_directory  : Directory of the training run.
            logger         : Logger passed through to the dataset pipeline.
            seed           : Seed for augmentation and dataloader shuffling.
        """
        self.dataset_config = dataset_config
        self.trainer_config = trainer_config
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.seed           = seed

    def run(self):
        """Builds the dataloaders and the Gaussian height axis of the run.

        Returns:
            A tuple of the (train, val, test) dataloader triple, the datasets keyed by
            split name, the Gaussian x-axis in metres and its length in bins.
        """
        gaussian_cfg                    = self.trainer_config.gaussian
        self.dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(self.dataset_config, self.run_directory, activity_threshold=gaussian_cfg.activity_threshold, logger=self.logger, seed=self.seed, resume=self.trainer_config.training.resume)
        z_len            = dataset_pipeline.profile_length
        z_axis           = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, z_len)

        self.dataset_config.z_axis = z_axis

        train_loader, val_loader, test_loader, datasets = dataset_pipeline.run()

        return (train_loader, val_loader, test_loader), datasets, z_axis, z_len
