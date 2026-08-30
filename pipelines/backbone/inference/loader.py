"""Rebuilds a trained backbone run for inference from its persisted artifacts.

Reads the run summary, dataset-creation config, checkpoint and normalization statistics,
reconstructs the backbone with the persisted head, and assembles the patch dataset,
dataloader, stitching grid and track metadata into a single `Run` container.
"""

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
    """Placeholder standing in for the raw cube when it was deliberately not loaded.

    Any attribute access, indexing or array conversion raises, so an analysis that
    silently needs the cube fails loudly instead of degrading.

    Attributes:
        run_directory : Directory of the run whose cube was skipped, quoted in the error.
    """

    def __init__(self, run_directory: Path) -> None:
        """Stores the run directory quoted in the refusal message."""

        self.run_directory = run_directory

    def _refuse(self, usage: str) -> None:
        """Raises a ValueError naming the attempted use of the missing cube.

        Args:
            usage : Description of what was attempted, embedded in the message.

        Raises:
            ValueError : Always.
        """

        raise ValueError(f"Run '{self.run_directory}' was loaded with load_cube=False, so the raw cube was never read; {usage} needs it. Reload the run with load_cube=True.")

    def __getattr__(self, name: str):
        """Refuses any non-dunder attribute access on the missing tomogram."""

        if name.startswith("__"):
            raise AttributeError(name)

        self._refuse(f"attribute '{name}'")

    def __getitem__(self, index):
        """Refuses indexing into the missing tomogram."""

        self._refuse("indexing full_curves")

    def __array__(self, dtype=None, copy=None):
        """Refuses conversion of the missing tomogram to a NumPy array."""

        self._refuse("converting full_curves to an array")


@dataclass
class Run:
    """Everything an inference pass needs from one trained backbone run.

    Attributes:
        model            : Device-resident model wrapper mapping input patches to denormalised outputs.
        backbone_name    : Registry name of the backbone architecture.
        in_channels      : Number of input channels the model consumes.
        out_channels     : Number of output channels the model emits (3 * n_gaussians).
        x_axis           : Elevation axis in metres, of shape (n_elevation,).
        x_axis_length    : Number of elevation bins.
        n_gaussians      : Gaussian count K the run was trained with.
        dataset_config   : Dataset-creation configuration restored from the run.
        split_name       : Name of the split being inferred ("train", "val" or "test").
        split_region     : Crop region covered by the inference, including any guard extension.
        core_region      : Unextended split region, set only when a region override was applied.
        global_crop      : Crop region of the preprocessed scene the split sits inside.
        grid             : Patch grid used to stitch per-patch outputs back into a cube.
        dataset          : Patch dataset yielding the normalised input windows.
        loader           : Sequential dataloader over `dataset`.
        checkpoint_meta  : Metadata payload stored alongside the checkpoint weights.
        full_curves      : Raw tomogram of shape (n_elevation, azimuth, range), or a refusing placeholder.
        complex_inputs   : Complex SAR stack of shape (channels, azimuth, range), or None.
        n_secondaries    : Number of secondary tracks in the input stack.
        secondary_labels : Track labels of the secondaries, or None when the full stack is used.
    """

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
    core_region      : CropRegion | None
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
    """Loads a trained backbone run into a `Run` ready for the inference pipeline."""

    PARAMS_PER_GAUSSIAN = 3

    def _persisted_n_gaussians(self, run_summary: dict) -> int:
        """Returns the Gaussian count K recorded in the run summary.

        Args:
            run_summary : Parsed contents of meta/run_summary.json.

        Returns:
            The Gaussian count the run was trained with.

        Raises:
            KeyError : If the summary carries no 'n_gaussians' field.
        """

        if "n_gaussians" not in run_summary:
            raise KeyError(f"meta/run_summary.json of run '{self.run_directory}' carries no 'n_gaussians'; the Gaussian count is read from that field and is no longer derived from out_channels, so re-train the run or write the extraction K it was trained with into the summary.")

        return int(run_summary["n_gaussians"])

    def _validate_out_channels(self, out_channels: int, n_gaussians: int) -> None:
        """Checks that the persisted output width matches a three-parameter-per-Gaussian head.

        Args:
            out_channels : Output-channel count recorded in the run summary.
            n_gaussians  : Gaussian count K recorded in the run summary.

        Raises:
            ValueError : If `out_channels` differs from 3 * `n_gaussians`.
        """

        expected = self.PARAMS_PER_GAUSSIAN * n_gaussians

        if out_channels != expected:
            raise ValueError(f"Run '{self.run_directory}' persists out_channels={out_channels} but n_gaussians={n_gaussians} requires a {expected}-channel parameter head; the run summary describes a head this loader cannot read.")

    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size: Tuple[int, int]):
        """Rebuilds the backbone from the persisted model config and records its head name.

        Args:
            backbone_name : Registry name of the backbone architecture.
            in_channels   : Input-channel count to instantiate the model with.
            out_channels  : Output-channel count to instantiate the model with.
            patch_size    : Patch shape (azimuth, range) used to derive any image-size override.

        Returns:
            The instantiated backbone module, still on CPU and with random weights.
        """

        model_config, _ = BackboneModelConfigIO.load(self.meta_directory)
        self.model_head = model_config.head

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(backbone_name, patch_size))

        model, _ = get_backbone(backbone_name, config=model_config, **overrides)

        return model

    def _wrap_model(self, model, device: str, norm_stats: Stats, z_axis: np.ndarray, amp_max: float | None) -> ModelWrapper:
        """Wraps the model so calls take NumPy patches and return denormalised, clamped parameters.

        Args:
            model      : Backbone module already moved to `device` and switched to eval mode.
            device     : Torch device string the model runs on.
            norm_stats : Normalization statistics used to invert the output scaling.
            z_axis     : Z axis in metres, of shape (n_z,), bounding the μ clamp.
            amp_max    : Amplitude ceiling for the Gaussian clamp, or None to leave amplitudes free.

        Returns:
            The callable model wrapper.
        """

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
        load_cube   : bool = True,
        region_override : Optional[CropRegion] = None,
    ) -> Run:
        """Loads model, dataset, dataloader and track metadata for one split of a trained run.

        Args:
            split           : Split to infer over ("train", "val" or "test").
            batch_size      : Batch size override for the dataloader, or None to keep the trained value.
            num_workers     : Worker count for the dataloader.
            device          : Torch device string the model is moved to.
            checkpoint_name : File name of the checkpoint inside the run directory.
            load_tomogram   : Whether to read the raw tomogram; when False `full_curves` refuses use.
            region_override : Crop region replacing the split region, used for guard-extended inference.

        Returns:
            The assembled `Run` container.
        """

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
            dataset_config  = dataset_config,
            split_name      = split,
            n_gaussians     = n_gaussians,
            norm_stats      = norm_stats,
            load_cube   = load_cube,
            region_override = region_override,
        )

        core_region = dataset_config.split_regions.regions(split)[0] if region_override is not None else None


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
            core_region      = core_region,
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
