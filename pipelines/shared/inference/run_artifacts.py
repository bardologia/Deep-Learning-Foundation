
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

    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_json(self, name: str) -> dict:
        return FileIO.load_json(self.meta_directory / name)

    def _build_dataset_config(self, payload: dict, batch_size: Optional[int], num_workers: int) -> DatasetConfig:
        split_regions = SplitRegions.from_payload(payload["split_regions"])

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
    def _validate_x_axis(ckpt_x_axis: np.ndarray, z_axis: np.ndarray, ckpt_path: Path, dataset_dir: Path) -> None:
        if ckpt_x_axis.shape != z_axis.shape or not np.allclose(ckpt_x_axis, z_axis, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Checkpoint '{ckpt_path}' was trained on z axis [{ckpt_x_axis[0]:.3f}, {ckpt_x_axis[-1]:.3f}] with {ckpt_x_axis.size} bins, but the dataset it is scored on ({dataset_dir}) produces axis [{z_axis[0]:.3f}, {z_axis[-1]:.3f}] with {z_axis.size} bins; every z metric would be reported in the wrong units.")

    @staticmethod
    def _axis_label(axis: np.ndarray) -> str:
        return f"[{axis[0]:.3f}, {axis[-1]:.3f}] m, {axis.size} bins"

    def _build_dataset(self, dataset_config : DatasetConfig, split_name : str, n_gaussians : int, norm_stats : Stats, load_cube : bool, region_override : Optional[CropRegion] = None) -> Tuple[PatchDataset, GridInfo, CropRegion, Layout, dict]:
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = Cropper(layout, dataset_config.split_regions, logger=self.logger, secondary_labels=dataset_config.secondary_labels)

        regions = dataset_config.split_regions.regions(split_name)

        if region_override is None:
            if len(regions) != 1:
                raise ValueError(f"Inference requires a single contiguous region for split '{split_name}'; found {len(regions)} disjoint regions. Stitching is only defined over one rectangular crop.")

            region = regions[0]
        else:
            if not any(region_override.contains(stored) for stored in regions):
                raise ValueError(f"region_override {region_override.as_tuple()} must contain the stored '{split_name}' region {[stored.as_tuple() for stored in regions]}; the override only widens the inferred crop, it never relocates it.")

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

        return dataset, grid, region, layout, arrays
