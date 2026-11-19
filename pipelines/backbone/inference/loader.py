
from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration.dataset                      import DatasetConfig
from configuration.dataset.gaussian             import GaussianConfig
from tools.data.gaussians                       import GaussianAxis
from tools.data.regions                         import CropRegion
from models                                     import get_backbone
from pipelines.backbone.dataset.datasets        import PatchDataset
from pipelines.shared.model.model_builder       import ModelBuilder
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.stats           import Stats
from pipelines.backbone.dataset.spatial         import GridInfo
from pipelines.backbone.inference.model_wrapper import ModelWrapper
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from pipelines.shared.inference.metadata        import TargetDataset
from pipelines.shared.inference.run_artifacts   import RunArtifactLoader


@dataclass
class Run:

    model              : object
    backbone_name      : str
    in_channels        : int
    out_channels       : int
    z_axis             : np.ndarray
    x_axis_length      : int
    n_gaussians        : int
    activity_threshold : float
    dataset_config     : DatasetConfig
    training_dataset   : Path
    target             : TargetDataset | None
    split_name         : str
    split_region       : CropRegion
    core_region        : CropRegion | None
    global_crop        : CropRegion
    grid               : GridInfo
    dataset            : PatchDataset
    loader             : DataLoader
    checkpoint_meta    : dict
    full_curves        : np.ndarray | None
    complex_inputs     : np.ndarray | None = None
    n_secondaries      : int               = 0
    secondary_labels   : list | None       = None


class RunLoader(RunArtifactLoader):

    PARAMS_PER_GAUSSIAN = 3

    def _persisted_n_gaussians(self, run_summary: dict) -> int:
        return int(run_summary["n_gaussians"])

    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size: Tuple[int, int]):
        model_config, _ = BackboneModelConfigIO.load(self.meta_directory)
        self.model_head = model_config.head

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(backbone_name, patch_size))

        model, _ = get_backbone(backbone_name, config=model_config, **overrides)

        return model

    def _wrap_model(self, model, device: str, norm_stats: Stats, z_axis: np.ndarray, amp_max: float | None) -> ModelWrapper:
        return ModelWrapper(
            model               = model,
            device              = device,
            params_per_gaussian = self.PARAMS_PER_GAUSSIAN,
            normalizer          = Normalizer(norm_stats),
            z_axis              = torch.from_numpy(z_axis),
            amp_max             = amp_max,
        )

    def load(
        self,
        *,
        split           : str,
        batch_size      : Optional[int],
        num_workers     : int,
        device          : str,
        checkpoint_name : str,
        load_cube       : bool = True,
        region_override : Optional[CropRegion] = None,
        target          : Optional[TargetDataset] = None,
    ) -> Run:
        self.logger.section("[Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        run_summary = self._read_json("run_summary.json")

        trained_config = self._build_dataset_config(
            payload     = self._read_json("dataset_creation_config.json"),
            batch_size  = batch_size,
            num_workers = num_workers,
        )
        dataset_config = trained_config if target is None else target.dataset_config(trained_config)

        backbone_name = str(run_summary["model_name"])
        in_channels   = int(run_summary["in_channels"])
        out_channels  = int(run_summary["out_channels"])
        n_gaussians   = self._persisted_n_gaussians(run_summary)

        gaussian           = GaussianConfig.from_dataset(dataset_config.preprocessing_run_directory, dataset_config.parameters_path)
        activity_threshold = gaussian.activity_threshold

        ckpt_path = self.run_directory / checkpoint_name
        model     = self._build_model(backbone_name, in_channels, out_channels, dataset_config.patch.size)
        model     = model.to(device)

        ckpt, z_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["params"])

        model.eval()
        norm_stats = Stats.load(self.run_directory / "meta", self.logger)
        model      = self._wrap_model(model, device, norm_stats, z_axis, norm_stats.clamp.amp_max if norm_stats.clamp.enabled else None)

        dataset, grid, region, layout, arrays = self._build_dataset(
            dataset_config  = dataset_config,
            split_name      = split,
            n_gaussians     = n_gaussians,
            norm_stats      = norm_stats,
            load_cube       = load_cube,
            region_override = region_override,
        )

        dataset_axis = GaussianAxis.build(gaussian.x_min, gaussian.x_max, layout.profile_length)
        self._validate_x_axis(z_axis, dataset_axis, ckpt_path, dataset_config.preprocessing_run_directory)

        core_region = None
        if region_override is not None:
            core_region = next(stored for stored in dataset_config.split_regions.regions(split) if region_override.contains(stored))

        loader = DataLoader(
            dataset,
            batch_size  = dataset_config.batch_size,
            shuffle     = False,
            num_workers = dataset_config.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        self.logger.section(f"[Model]         : '{backbone_name}'")
        self.logger.kv_table({
            "Checkpoint":          ckpt_path,
            "Checkpoint epoch":    f"{ckpt_meta['epoch'] + 1} (best epoch {ckpt_meta['best_epoch'] + 1}, best val loss {ckpt_meta['best_val_loss']:.6g})",
            "Head":                self.model_head,
            "In channels":         in_channels,
            "Out channels":        out_channels,
            "K gaussians":         n_gaussians,
            "Activity thr":        f"{activity_threshold:g}",
            "Z axis (checkpoint)": self._axis_label(z_axis),
            "Z axis (dataset)":    self._axis_label(dataset_axis),
        })

        self.logger.section(f"[Split]         : '{split}'")
        self.logger.kv_table({
            "Scored dataset": dataset_config.preprocessing_run_directory if target is None else f"target {target.describe()}",
            "Trained on":     trained_config.preprocessing_run_directory,
            "Patches":        grid.grid.number_of_patches,
            "Row size":       region.row_size,
            "Col size":       region.col_size,
            "Z axis length":  z_axis.size,
            "Region":         region.as_tuple(),
            "Region source":  ("stored split region" if core_region is None else f"region override; stored core region {core_region.as_tuple()}") if target is None else "target crop",
            "Batch size":     f"{dataset_config.batch_size} (from training)" if batch_size is None else dataset_config.batch_size,
            "Loader workers": dataset_config.num_workers,
        })

        return Run(
            model              = model,
            backbone_name      = backbone_name,
            in_channels        = in_channels,
            out_channels       = out_channels,
            z_axis             = z_axis,
            x_axis_length      = int(z_axis.size),
            n_gaussians        = n_gaussians,
            activity_threshold = activity_threshold,
            dataset_config     = dataset_config,
            training_dataset   = Path(trained_config.preprocessing_run_directory),
            target             = target,
            split_name         = split,
            split_region       = region,
            core_region        = core_region,
            global_crop        = layout.global_crop,
            grid               = grid.grid,
            dataset            = dataset,
            loader             = loader,
            checkpoint_meta    = ckpt_meta,
            complex_inputs     = arrays["inputs"],
            n_secondaries      = arrays["n_secondaries"],
            secondary_labels   = arrays["secondary_labels"],
            full_curves        = arrays["cube"] if load_cube else None,
        )
