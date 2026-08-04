from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration.dataset                      import DatasetConfig
from tools.data.regions                         import CropRegion
from models                                     import get_backbone
from pipelines.backbone.dataset.datasets        import PatchDataset
from pipelines.shared.model.model_builder       import ModelBuilder
from pipelines.backbone.dataset.normalizer      import Normalizer
from pipelines.backbone.dataset.stats           import Stats
from pipelines.backbone.dataset.spatial         import GridInfo
from pipelines.backbone.inference.model_wrapper import ModelWrapper
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from pipelines.shared.inference.run_artifacts   import RunArtifactLoader


class AbsentCube:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory

    def _refuse(self, usage: str) -> None:
        raise ValueError(f"Run '{self.run_directory}' was loaded with load_cube=False, so the raw cube was never read; {usage} needs it. Reload the run with load_cube=True.")

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        self._refuse(f"attribute '{name}'")

    def __getitem__(self, index):
        self._refuse("indexing full_curves")

    def __array__(self, dtype=None, copy=None):
        self._refuse("converting full_curves to an array")


@dataclass
class Run:
    model            : object
    backbone_name    : str
    in_channels      : int
    out_channels     : int
    z_axis           : np.ndarray
    x_axis_length    : int
    n_gaussians      : int
    dataset_config   : DatasetConfig
    split_name       : str
    split_region     : CropRegion
    global_crop      : CropRegion
    grid             : GridInfo
    dataset          : PatchDataset
    loader           : DataLoader
    checkpoint_meta  : dict
    full_curves      : np.ndarray | AbsentCube
    complex_inputs   : np.ndarray | None = None
    n_secondaries    : int               = 0
    secondary_labels : list | None       = None


class RunLoader(RunArtifactLoader):
    PARAMS_PER_GAUSSIAN = 3

    def _persisted_n_gaussians(self, run_summary: dict) -> int:
        if "n_gaussians" not in run_summary:
            raise KeyError(f"meta/run_summary.json of run '{self.run_directory}' carries no 'n_gaussians'; the Gaussian count is read from that field and is no longer derived from out_channels, so re-train the run or write the extraction K it was trained with into the summary.")

        return int(run_summary["n_gaussians"])

    def _validate_out_channels(self, out_channels: int, n_gaussians: int) -> None:
        expected = self.PARAMS_PER_GAUSSIAN * n_gaussians

        if out_channels != expected:
            raise ValueError(f"Run '{self.run_directory}' persists out_channels={out_channels} but n_gaussians={n_gaussians} requires a {expected}-channel parameter head; the run summary describes a head this loader cannot read.")

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
    ) -> Run:

        self.logger.section("[Inference: Load Run]")
        self.logger.subsection(f"Run Directory : {self.run_directory} \n")

        run_summary = self._read_json("run_summary.json")

        dataset_config = self._build_dataset_config(
            payload     = self._read_json("dataset_creation_config.json"),
            batch_size  = batch_size,
            num_workers = num_workers,
        )

        backbone_name = str(run_summary["model_name"])
        in_channels   = int(run_summary["in_channels"])
        out_channels  = int(run_summary["out_channels"])
        n_gaussians   = self._persisted_n_gaussians(run_summary)

        self._validate_out_channels(out_channels, n_gaussians)

        ckpt_path = self.run_directory / checkpoint_name
        model     = self._build_model(backbone_name, in_channels, out_channels, dataset_config.patch.size)
        model     = model.to(device)

        ckpt, z_axis, ckpt_meta = self._load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["params"])

        model.eval()
        norm_stats = Stats.load(self.run_directory / "meta", self.logger)
        model      = self._wrap_model(model, device, norm_stats, z_axis, norm_stats.clamp.amp_max if norm_stats.clamp.enabled else None)

        dataset, grid, region, global_crop, arrays = self._build_dataset(
            dataset_config = dataset_config,
            split_name     = split,
            z_axis         = z_axis,
            n_gaussians    = n_gaussians,
            norm_stats     = norm_stats,
            load_cube      = load_cube,
        )

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
            "Checkpoint":   ckpt_path,
            "Head":         self.model_head,
            "In channels":  in_channels,
            "Out channels": out_channels,
            "K gaussians":  n_gaussians,
        })

        self.logger.section(f"[Split]         : '{split}'")
        self.logger.kv_table({
            "Patches":       grid.grid.number_of_patches,
            "Row size":  region.row_size,
            "Col size":    region.col_size,
            "X-axis length": z_axis.size,
        })

        return Run(
            model            = model,
            backbone_name    = backbone_name,
            in_channels      = in_channels,
            out_channels     = out_channels,
            z_axis           = z_axis,
            x_axis_length    = int(z_axis.size),
            n_gaussians      = n_gaussians,
            dataset_config   = dataset_config,
            split_name       = split,
            split_region     = region,
            global_crop      = global_crop,
            grid             = grid.grid,
            dataset          = dataset,
            loader           = loader,
            checkpoint_meta  = ckpt_meta,
            complex_inputs   = arrays["inputs"],
            n_secondaries    = arrays["n_secondaries"],
            secondary_labels = arrays["secondary_labels"],
            full_curves      = arrays["cube"] if load_cube else AbsentCube(self.run_directory),
        )
