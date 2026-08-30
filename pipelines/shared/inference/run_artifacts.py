"""Reconstruction of a trained run's dataset and checkpoint for inference.

Rebuilds the persisted dataset configuration, loads the checkpoint scalars, and
assembles the patch dataset and stitching grid for the split being scored.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Optional, Tuple

import numpy as np
import torch

from configuration.dataset                    import DatasetConfig, InputConfig, OutputConfig, PatchConfig, SplitRegions
from pipelines.backbone.dataset.datasets      import PatchDataset
from pipelines.backbone.dataset.normalizer    import Normalizer
from pipelines.backbone.dataset.spatial       import Cropper, GridInfo, Patcher
from pipelines.backbone.dataset.stats         import Stats
from pipelines.shared.dataset.dataset_spatial import Layout
from tools.data.io                            import FileIO
from tools.data.regions                       import CropRegion
from tools.monitoring.logger                  import Logger


class RunArtifactLoader:
    """Rebuilds the dataset, grid and checkpoint of a trained run.

    Attributes:
        run_directory  : Root of the training run being scored.
        logger         : Logger passed on to the layout and cropper.
        meta_directory : Run's meta/ directory holding the persisted configs.
    """

    def __init__(self, run_directory: Path, logger: Logger) -> None:
        """Points the loader at a training run and its meta directory."""
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_json(self, name: str) -> dict:
        """Returns the JSON payload of a file in the run's meta directory."""
        return FileIO.load_json(self.meta_directory / name)

    def _parse_split_payload(self, value):
        """Returns one persisted split as a CropRegion, or a list of them when disjoint."""
        if isinstance(value, list):
            return [CropRegion(**region) for region in value]
        return CropRegion(**value)

    def _build_dataset_config(self, payload: dict, batch_size: Optional[int], num_workers: int) -> DatasetConfig:
        """Rebuilds the training-time dataset configuration for inference use.

        Args:
            payload     : Contents of dataset_creation_config.json.
            batch_size  : Batch size override; the stored value is kept when None.
            num_workers : Loader worker count to use at inference time.

        Returns:
            Dataset configuration with shuffling disabled.

        Raises:
            ValueError : The stored patch stride predates per-axis strides.
        """
        splits = payload["split_regions"]

        split_regions = SplitRegions(
            train = self._parse_split_payload(splits["train"]),
            val   = self._parse_split_payload(splits["val"]),
            test  = self._parse_split_payload(splits["test"]),
        )

        if not isinstance(payload["patch"]["stride"], list):
            raise ValueError(f"dataset_creation_config.json stores patch stride {payload['patch']['stride']!r} from before per-axis strides; patch it to a [vertical, horizontal] list (e.g. [32, 32]) to run inference on this run.")

        patch = PatchConfig(
            size                  = tuple(payload["patch"]["size"]),
            stride                = tuple(payload["patch"]["stride"]),
            use_symmetric_padding = bool(payload["patch"]["use_symmetric_padding"]),
        )

        secondary_labels = payload["secondary_labels"]

        return DatasetConfig(
            preprocessing_run_directory = Path(payload["preprocessing_run_directory"]),
            parameters_path             = Path(payload["parameters_path"]),
            split_regions               = split_regions,
            secondary_labels            = tuple(secondary_labels) if secondary_labels is not None else None,
            patch                       = patch,
            input_config                = InputConfig.from_dict(payload["input_config"]),
            output_config               = OutputConfig.from_dict(payload["output_config"]),
            batch_size                  = batch_size if batch_size is not None else int(payload["batch_size"]),
            num_workers                 = int(num_workers),
            shuffle_train               = False,
            pin_memory                  = bool(payload["pin_memory"]),
            n_gaussians                 = int(payload["n_gaussians"]),
        )

    def _load_checkpoint(self, ckpt_path: Path, device: str) -> tuple[dict, np.ndarray, dict]:
        """Loads a checkpoint together with its elevation axis and best-epoch scalars.

        Args:
            ckpt_path : Path of the checkpoint file.
            device    : Torch device the tensors are mapped to.

        Returns:
            Triple of the raw checkpoint dict, the elevation axis of shape
            (x_axis_length,) in metres, and a metadata dict with the last epoch,
            the best validation loss and the best epoch.

        Raises:
            FileNotFoundError : No checkpoint exists at that path.
        """
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for inference: {ckpt_path}")

        ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        z_axis = np.asarray(ckpt["z_axis"], dtype=np.float32)

        meta = {
            "epoch"         : int(ckpt["epoch"]),
            "best_val_loss" : float(ckpt["best_val_loss"]),
            "best_epoch"    : int(ckpt["best_epoch"]),
        }

        return ckpt, z_axis, meta

    @staticmethod
    def _validate_x_axis(ckpt_x_axis: np.ndarray, z_axis: np.ndarray, ckpt_path: Path) -> None:
        """Checks that the checkpoint's z axis matches the rebuilt dataset axis.

        Args:
            ckpt_x_axis : Z axis stored in the checkpoint, shape (x_axis_length,) in metres.
            z_axis      : Z axis produced by the persisted dataset config, shape (x_axis_length,) in metres.
            ckpt_path   : Checkpoint path, quoted in the error message.

        Raises:
            ValueError : The axes differ in length or values, which would report every z metric in the wrong units.
        """
        if ckpt_x_axis.shape != z_axis.shape or not np.allclose(ckpt_x_axis, z_axis, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Checkpoint '{ckpt_path}' was trained on z axis [{ckpt_x_axis[0]:.3f}, {ckpt_x_axis[-1]:.3f}] with {ckpt_x_axis.size} bins, but the persisted dataset config produces axis [{z_axis[0]:.3f}, {z_axis[-1]:.3f}] with {z_axis.size} bins; every z metric would be reported in the wrong units.")

    def _build_dataset(self, dataset_config : DatasetConfig, split_name : str, n_gaussians : int, norm_stats : Stats, load_cube : bool, region_override : Optional[CropRegion] = None) -> Tuple[PatchDataset, GridInfo, CropRegion, CropRegion, dict]:
        """Builds the patch dataset and stitching grid of one split.

        Args:
            dataset_config  : Rebuilt dataset configuration of the run.
            split_name      : Split to load ("train", "val", "test").
            n_gaussians     : Number of Gaussian components per pixel in the targets.
            norm_stats      : Normalization statistics fitted at training time.
            load_cube   : Whether the full cube cube is loaded alongside the inputs, needed for curve-space metrics.
            region_override : Wider crop replacing the stored split region, used to extend a cross-validation fold over its guard bands.

        Returns:
            Tuple of the patch dataset, the patch grid describing the tiling of
            the region, the region actually loaded, the dataset's global crop,
            and the raw split arrays keyed by array name.

        Raises:
            ValueError : The split is not a single contiguous region, so stitching is undefined, or the override does not contain the stored region.
        """
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = Cropper(layout, dataset_config.split_regions, logger=self.logger, secondary_labels=dataset_config.secondary_labels)

        regions = dataset_config.split_regions.regions(split_name)
        if len(regions) != 1:
            raise ValueError(f"Inference requires a single contiguous region for split '{split_name}'; found {len(regions)} disjoint regions. Stitching is only defined over one rectangular crop.")

        region = regions[0]
        if region_override is not None:
            if not region_override.contains(region):
                raise ValueError(f"region_override {region_override.as_tuple()} must contain the stored '{split_name}' region {region.as_tuple()}; the override only widens the inferred crop, it never relocates it.")
            region = region_override

        arrays = cropper.load_split(region, load_cube=load_cube)

        grid = Patcher.build(
            spatial_size          = (region.row_size, region.col_size),
            patch_size            = dataset_config.patch.size,
            stride                = dataset_config.patch.stride,
            use_symmetric_padding = dataset_config.patch.use_symmetric_padding,
        )

        inputs        = arrays["inputs"]
        gt_parameters = arrays["parameters"]

        dataset = PatchDataset(
            inputs        = inputs,
            gt_parameters = gt_parameters,
            grid          = grid,
            input_config  = dataset_config.input_config,
            output_config = dataset_config.output_config,
            split_name    = split_name,
            n_secondaries = arrays["n_secondaries"],
            n_derived     = arrays["n_derived"],
            normalizer    = Normalizer(norm_stats),
            n_gaussians   = n_gaussians,
            aux           = arrays["aux"] if dataset_config.input_config.use_aux else None,
        )

        return dataset, grid, region, layout.global_crop, arrays
