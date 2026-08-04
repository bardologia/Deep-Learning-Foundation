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

    def _parse_split_payload(self, value):
        if isinstance(value, list):
            return [CropRegion(**region) for region in value]
        return CropRegion(**value)

    def _build_dataset_config(self, payload: dict, batch_size: Optional[int], num_workers: int) -> DatasetConfig:
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
        ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        x_axis = np.asarray(ckpt["x_axis"], dtype=np.float32)

        meta = {
            "epoch"         : int(ckpt["epoch"]),
            "best_val_loss" : float(ckpt["best_val_loss"]),
            "best_epoch"    : int(ckpt["best_epoch"]),
        }

        return ckpt, x_axis, meta

    def _build_dataset(self, dataset_config : DatasetConfig, split_name : str, x_axis : np.ndarray, n_gaussians : int, norm_stats : Stats, load_tomogram : bool) -> Tuple[PatchDataset, GridInfo, CropRegion, CropRegion, dict]:
        layout  = Layout(dataset_config.preprocessing_run_directory, logger=self.logger, parameters_path=dataset_config.parameters_path)
        cropper = Cropper(layout, dataset_config.split_regions, logger=self.logger, secondary_labels=dataset_config.secondary_labels)

        regions = dataset_config.split_regions.regions(split_name)
        if len(regions) != 1:
            raise ValueError(f"Inference requires a single contiguous region for split '{split_name}'; found {len(regions)} disjoint regions. Stitching is only defined over one rectangular crop.")

        region = regions[0]
        arrays = cropper.load_split(region, load_tomogram=load_tomogram)

        grid = Patcher.build(
            spatial_size          = (region.azimuth_size, region.range_size),
            patch_size            = dataset_config.patch.size,
            stride                = dataset_config.patch.stride,
            use_symmetric_padding = dataset_config.patch.use_symmetric_padding,
        )

        inputs        = arrays["inputs"]
        gt_parameters = arrays["parameters"]

        dataset = PatchDataset(
            inputs           = inputs,
            gt_parameters    = gt_parameters,
            grid             = grid,
            input_config     = dataset_config.input_config,
            output_config    = dataset_config.output_config,
            split_name       = split_name,
            n_secondaries    = arrays["n_secondaries"],
            n_interferograms = arrays["n_interferograms"],
            normalizer       = Normalizer(norm_stats),
            n_gaussians      = n_gaussians,
            dem              = arrays["dem"] if dataset_config.input_config.use_dem else None,
        )

        return dataset, grid, region, layout.global_crop, arrays
