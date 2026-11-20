from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration.dataset                      import DatasetConfig
from models                                     import get_backbone
from pipelines.backbone.dataset.datasets        import PatchDataset
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.spatial         import GridInfo, Patcher
from pipelines.backbone.dataset.stats           import Stats
from pipelines.backbone.inference.model_wrapper import ModelWrapper
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from pipelines.shared.dataset.dataset_files     import DatasetFiles
from pipelines.shared.inference.metadata        import TargetDataset
from pipelines.shared.model.model_builder       import ModelBuilder
from tools.data.io                              import FileIO
from tools.data.regions                         import CropRegion
from tools.monitoring.logger                    import Logger


@dataclass
class Run:

    model            : ModelWrapper
    backbone_name    : str
    input_channels   : list[str]
    target_channels  : list[str]
    dataset_config   : DatasetConfig
    training_dataset : Path
    target           : TargetDataset | None
    split_name       : str
    split_region     : CropRegion
    core_region      : CropRegion | None
    grid             : GridInfo
    dataset          : PatchDataset
    loader           : DataLoader
    checkpoint_meta  : dict
    inputs           : np.ndarray
    targets          : np.ndarray

    @property
    def in_channels(self) -> int:
        return len(self.input_channels)

    @property
    def out_channels(self) -> int:
        return len(self.target_channels)


class RunLoader:

    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory  = Path(run_directory)
        self.logger         = logger
        self.meta_directory = self.run_directory / "meta"

    def _read_json(self, name: str) -> dict:
        return FileIO.load_json(self.meta_directory / name)

    def _dataset_config(self, batch_size: Optional[int], num_workers: int) -> DatasetConfig:
        config             = DatasetConfig.from_dict(self._read_json("dataset_creation_config.json"))
        config.batch_size  = batch_size if batch_size is not None else config.batch_size
        config.num_workers = int(num_workers)

        return config

    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size, device: str) -> torch.nn.Module:
        model_config, _ = BackboneModelConfigIO.load(self.meta_directory)

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(backbone_name, patch_size))

        model, _ = get_backbone(backbone_name, config=model_config, **overrides)

        return model.to(device)

    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path: Path, device: str) -> dict:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["params"])
        model.eval()

        return {
            "epoch"         : int(checkpoint["epoch"]) + 1,
            "best_val_loss" : float(checkpoint["best_val_loss"]),
            "best_epoch"    : int(checkpoint["best_epoch"]) + 1,
        }

    def _region(self, dataset_config: DatasetConfig, split: str, region_override: Optional[CropRegion]) -> tuple[CropRegion, Optional[CropRegion]]:
        regions = dataset_config.split_regions.regions(split)

        if region_override is None:
            return regions[0], None

        return region_override, next(stored for stored in regions if region_override.contains(stored))

    def load(self, *, split: str, batch_size: Optional[int], num_workers: int, device: str, checkpoint_name: str, region_override: Optional[CropRegion] = None, target: Optional[TargetDataset] = None) -> Run:
        self.logger.section("[Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        run_summary    = self._read_json("run_summary.json")
        trained_config = self._dataset_config(batch_size, num_workers)
        dataset_config = trained_config if target is None else target.dataset_config(trained_config)

        backbone_name   = str(run_summary["model_name"])
        input_channels  = list(run_summary["input_channels"])
        target_channels = list(run_summary["target_channels"])
        checkpoint_path = self.run_directory / checkpoint_name

        model           = self._build_model(backbone_name, len(input_channels), len(target_channels), dataset_config.patch.size, device)
        checkpoint_meta = self._load_checkpoint(model, checkpoint_path, device)
        normalizer      = Normalizer(Stats.load(self.meta_directory, self.logger))

        region, core_region = self._region(dataset_config, split, region_override)

        files   = DatasetFiles.read(dataset_config.dataset_directory)
        inputs  = DatasetFiles.crop(files.inputs(), region)
        targets = DatasetFiles.crop(files.targets(), region)

        patcher = Patcher.build(
            spatial_size          = (region.row_size, region.col_size),
            patch_size            = dataset_config.patch.size,
            stride                = dataset_config.patch.stride,
            use_symmetric_padding = dataset_config.patch.use_symmetric_padding,
        )

        dataset = PatchDataset(inputs, targets, patcher, normalizer, count_once=True)
        loader  = DataLoader(dataset, batch_size=dataset_config.batch_size, shuffle=False, num_workers=dataset_config.num_workers, pin_memory=True, drop_last=False)

        self.logger.section(f"[Model]         : '{backbone_name}'")
        self.logger.kv_table({
            "Checkpoint"       : checkpoint_path,
            "Checkpoint epoch" : f"{checkpoint_meta['epoch']} (best epoch {checkpoint_meta['best_epoch']}, best val loss {checkpoint_meta['best_val_loss']:.6g})",
            "Input channels"   : ", ".join(input_channels),
            "Target channels"  : ", ".join(target_channels),
        })

        self.logger.section(f"[Split]         : '{split}'")
        self.logger.kv_table({
            "Scored dataset" : dataset_config.dataset_directory if target is None else f"target {target.describe()}",
            "Trained on"     : trained_config.dataset_directory,
            "Region"         : region.as_tuple(),
            "Region source"  : "stored split region" if core_region is None else f"region override; stored core region {core_region.as_tuple()}",
            "Patches"        : patcher.grid.number_of_patches,
            "Batch size"     : f"{dataset_config.batch_size} (from training)" if batch_size is None else dataset_config.batch_size,
            "Loader workers" : dataset_config.num_workers,
        })

        return Run(
            model            = ModelWrapper(model, device, normalizer),
            backbone_name    = backbone_name,
            input_channels   = input_channels,
            target_channels  = target_channels,
            dataset_config   = dataset_config,
            training_dataset = Path(trained_config.dataset_directory),
            target           = target,
            split_name       = split,
            split_region     = region,
            core_region      = core_region,
            grid             = patcher.grid,
            dataset          = dataset,
            loader           = loader,
            checkpoint_meta  = checkpoint_meta,
            inputs           = inputs,
            targets          = targets,
        )
