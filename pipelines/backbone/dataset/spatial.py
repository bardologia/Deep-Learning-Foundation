"""Spatial handling of the backbone dataset: crop loading and patch grids.

Cropper reads the per-split slices out of the memory-mapped preprocessing artifacts,
GridInfo describes the resulting patch grid and Patcher cuts padded patches from it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing      import Optional, Tuple

import numpy as np

from tools.data.regions      import CropRegion, SplitRegions
from tools.monitoring.logger import Logger

from pipelines.shared.dataset.dataset_spatial import Layout


class Cropper:
    """Loads the input stack, targets and optional aux field of one crop region.

    Attributes:
        layout            : Layout resolving the preprocessing artifact paths and the global crop.
        split_regions     : Crop regions of the train, validation and test splits.
        secondary_indices : Indices of the selected secondary channels, or None for all channels.
        secondary_labels  : Labels of the selected secondary channels, or None for all channels.
    """

    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger, secondary_labels=None, input_config=None) -> None:
        """Resolves the secondary selection and validates the split regions against the artifacts.

        Args:
            layout           : Layout of the preprocessing run to crop from.
            split_regions    : Crop regions of the splits, checked to sit inside the global crop.
            logger           : Logger receiving the crop tables.
            secondary_labels : Labels of the secondary channels to keep, or None to keep all.
            input_config     : Input configuration deciding which artifact groups are read; when None, every group is treated as enabled.
        """
        self.layout            = layout
        self.split_regions     = split_regions
        self.logger            = logger
        self.input_config      = input_config
        self.secondary_indices = layout.secondary_indices(secondary_labels)
        self.secondary_labels  = self._resolve_labels(secondary_labels)

        split_regions.validate_within(layout.global_crop)
        self._validate_parameters_extent()

        self.logger.section("[Cropper Initialized]")
        self.logger.subsection(f"Secondary selection : {', '.join(self.secondary_labels) if self.secondary_labels else 'all channels'}")
        self.logger.metrics_table(split_regions.region_rows(), ["Split", "Crop", "Rows", "Cols"])

    def _resolve_labels(self, secondary_labels) -> list | None:
        """Returns the pass labels of the selected secondaries, or None when all are kept."""
        if self.secondary_indices is None:
            return None
        secondaries = self.layout.channel_labels[1:]
        return [secondaries[index] for index in self.secondary_indices]

    def _validate_parameters_extent(self) -> None:
        """Checks the parameter artifact covers exactly the preprocessing global crop.

        Raises:
            ValueError : If the parameter array's spatial shape differs from the global crop extent, meaning the extraction belongs to another crop.
        """
        parameters_path = self.layout.artifact_path("parameters")
        parameters      = np.load(str(parameters_path), mmap_mode="r", allow_pickle=False)
        expected        = (self.layout.global_crop.row_size, self.layout.global_crop.col_size)

        if tuple(parameters.shape[-2:]) != expected:
            raise ValueError(f"Parameters artifact {parameters_path} has spatial shape {tuple(parameters.shape[-2:])} but the preprocessing global crop is {expected} (row, col); the parameter extraction does not belong to this dataset crop.")

    def _to_local_slices(self, region: CropRegion) -> Tuple[slice, slice]:
        """Returns the row and column slices of a region inside the global crop arrays."""
        return region.local_slices(self.layout.global_crop)

    def _select_channels(self, array: np.ndarray, row_slice: slice, col_slice: slice) -> np.ndarray:
        """Crops an artifact and keeps only the selected secondary channels.

        Args:
            array    : Artifact of shape (channels, row, col) over the global crop.
            row_slice : Row slice of the region.
            col_slice : Range slice of the region.

        Returns:
            The cropped array of shape (selected_passes, region_row, region_range).
        """
        if self.secondary_indices is None:
            return array[..., row_slice, col_slice]

        return np.stack([array[index, row_slice, col_slice] for index in self.secondary_indices])

    def _group_enabled(self, flag: str) -> bool:
        """Returns whether an input group is enabled, treating a missing config as all-enabled."""
        return getattr(self.input_config, flag) if self.input_config is not None else True

    def _group_view(self, artifact: str, flag: str, row_slice: slice, col_slice: slice) -> Optional[np.ndarray]:
        """Memory-maps one artifact group and returns its cropped, chan-selected view.

        Args:
            artifact : Artifact name in the preprocessing layout.
            flag     : Name of the input-config flag gating this group.
            row_slice : Row slice of the region.
            col_slice : Range slice of the region.

        Returns:
            The cropped array of shape (selected_passes, region_row, region_range), or
            None when the group is disabled.
        """
        if not self._group_enabled(flag):
            return None

        array = np.load(str(self.layout.artifact_path(artifact)), mmap_mode="r", allow_pickle=False)

        return self._select_channels(array, row_slice, col_slice)

    def _stack_inputs(self, row_slice: slice, col_slice: slice) -> Tuple[np.ndarray, int, int, int]:
        """Assembles the complex input stack of one region from the enabled groups.

        Args:
            row_slice : Row slice of the region.
            col_slice : Range slice of the region.

        Returns:
            A tuple of the complex stack of shape (layers, region_row, region_range),
            ordered primary, secondaries, derived, and the number of primary,
            secondary and derived layers it holds.
        """
        primary      = np.load(str(self.layout.artifact_path("primary")), mmap_mode="r", allow_pickle=False)
        primary_view = primary[..., row_slice, col_slice]

        secondaries_view = self._group_view("secondaries",    "use_secondaries",    row_slice, col_slice)
        derived_view     = self._group_view("derived", "use_derived", row_slice, col_slice)

        n_primary     = 1 if self._group_enabled("use_primary") else 0
        n_secondaries = secondaries_view.shape[0]    if secondaries_view    is not None else 0
        n_derived     = derived_view.shape[0] if derived_view is not None else 0

        n_channels = n_primary + n_secondaries + n_derived
        row_size   = primary_view.shape[-2]
        col_size   = primary_view.shape[-1]

        inputs_split = np.empty((n_channels, row_size, col_size), dtype=primary.dtype)

        if n_primary:
            inputs_split[0] = primary_view

        if secondaries_view is not None:
            inputs_split[n_primary:n_primary + n_secondaries] = secondaries_view

        if derived_view is not None:
            inputs_split[n_primary + n_secondaries:] = derived_view

        return inputs_split, n_primary, n_secondaries, n_derived

    def _load_targets(self, row_slice: slice, col_slice: slice) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Loads the ground-truth parameters and, when enabled, the Aux of one region.

        Args:
            row_slice : Row slice of the region.
            col_slice : Range slice of the region.

        Returns:
            A pair of the parameter array of shape (3 * gaussians, region_row,
            region_range) and the float32 Aux of shape (region_row, region_range) in
            metres, or None when the Aux channel is disabled.
        """
        parameters       = np.load(str(self.layout.artifact_path("parameters")), mmap_mode="r", allow_pickle=False)
        parameters_split = np.ascontiguousarray(parameters[..., row_slice, col_slice])

        if not self._group_enabled("use_aux"):
            return parameters_split, None

        aux       = np.load(str(self.layout.artifact_path("aux_full")), mmap_mode="r", allow_pickle=False)
        aux_split = np.ascontiguousarray(aux[row_slice, col_slice].astype(np.float32))

        return parameters_split, aux_split

    def _load_cube(self, row_slice: slice, col_slice: slice) -> np.ndarray:
        """Loads the magnitude of the reference cube over one region.

        Args:
            row_slice : Row slice of the region.
            col_slice : Column slice of the region.

        Returns:
            Float32 array of shape (z_bins, region_rows, region_cols).
        """
        cube = np.load(str(self.layout.artifact_path("cube_full")), mmap_mode="r", allow_pickle=False)
        return np.ascontiguousarray(np.abs(cube[:, row_slice, col_slice]).astype(np.float32))

    def _log_crop(self, region: CropRegion, label: Optional[str], read_seconds: float, inputs_split: np.ndarray, aux_split: Optional[np.ndarray], parameters_split: np.ndarray, cube_split: Optional[np.ndarray], n_primary: int, n_secondaries: int, n_derived: int) -> None:
        """Reports the region, read time, shapes and memory footprint of the loaded crop.

        Args:
            region           : Crop region that was read, in preprocessing global coordinates.
            label            : Split and region index quoted in the section title, such as "train 1/2", or None for an unlabelled crop.
            read_seconds     : Wall time the reads out of the memory-mapped artifacts took, in seconds.
            inputs_split     : Assembled complex stack of the region.
            aux_split        : Loaded aux field, or None when disabled.
            parameters_split : Loaded ground-truth parameters of the region.
            cube_split       : Loaded cube magnitude, or None when not requested.
            n_primary        : Number of primary layers in the stack.
            n_secondaries    : Number of secondary layers in the stack.
            n_derived        : Number of derived layers in the stack.
        """
        self.logger.section("[Crop Loaded]" if label is None else f"[Crop Loaded: {label}]")
        self.logger.kv_table({
            "Region"           : f"row {region.row_start}:{region.row_end}, col {region.col_start}:{region.col_end}",
            "Read time"        : f"{read_seconds:.1f}s",
            "Primary"          : inputs_split[:n_primary].shape,
            "Secondaries"      : inputs_split[n_primary:n_primary + n_secondaries].shape,
            "Derived"          : inputs_split[n_primary + n_secondaries:].shape,
            "Selection"        : ", ".join(self.secondary_labels) if self.secondary_labels else "all channels",
            "Inputs (stacked)" : f"{inputs_split.shape}  ({inputs_split.nbytes/1e9:.2f} GB)  [{n_primary} primary + {n_secondaries} secondaries + {n_derived} derived]",
            "Aux"              : f"{aux_split.shape}  ({aux_split.nbytes/1e6:.2f} MB)" if aux_split is not None else "not loaded",
            "Parameters"       : f"{parameters_split.shape}  ({parameters_split.nbytes/1e9:.2f} GB)",
            "Full cube"        : f"{cube_split.shape}  ({cube_split.nbytes/1e9:.2f} GB)" if cube_split is not None else "not loaded",
        })

    def load_split(self, region: CropRegion, load_cube: bool = False, label: Optional[str] = None) -> dict[str, np.ndarray]:
        """Loads every array a dataset part needs for one crop region.

        Args:
            region    : Crop region in preprocessing global coordinates.
            load_cube : Whether to also load the reference cube magnitude.
            label     : Split and region index quoted in the [Crop Loaded] title, such as "train 1/2", or None for an unlabelled crop.

        Returns:
            Dictionary with the complex "inputs" stack, the "aux" (or None), the
            "parameters", the "cube" (or None), the layer counts "n_primary",
            "n_secondaries" and "n_derived", and the "secondary_labels".
        """
        row_slice, col_slice = self._to_local_slices(region)
        start                = time.perf_counter()

        inputs_split, n_primary, n_secondaries, n_derived = self._stack_inputs(row_slice, col_slice)
        parameters_split, aux_split                       = self._load_targets(row_slice, col_slice)
        cube_split                                        = self._load_cube(row_slice, col_slice) if load_cube else None

        self._log_crop(region, label, time.perf_counter() - start, inputs_split, aux_split, parameters_split, cube_split, n_primary, n_secondaries, n_derived)

        return {
            "inputs"           : inputs_split,
            "aux"              : aux_split,
            "parameters"       : parameters_split,
            "cube"             : cube_split,
            "n_primary"        : n_primary,
            "n_secondaries"    : n_secondaries,
            "n_derived"        : n_derived,
            "secondary_labels" : self.secondary_labels,
        }


@dataclass
class GridInfo:
    """Geometry of the patch grid laid over one crop region.

    Attributes:
        n_v                   : Number of patch rows along row.
        n_h                   : Number of patch columns along range.
        pad_top               : Padding added above the region, in pixels.
        pad_bot               : Padding added below the region, in pixels.
        pad_left              : Padding added left of the region, in pixels.
        pad_right             : Padding added right of the region, in pixels.
        patch_size            : Patch extent as (rows, cols) in pixels.
        stride                : Grid step as (rows, cols) in pixels.
        spatial_size          : Region extent as (rows, cols) in pixels, before padding.
        use_symmetric_padding : Whether border patches are mirror-padded instead of zero-padded.
    """

    n_v                   : int
    n_h                   : int
    pad_top               : int
    pad_bot               : int
    pad_left              : int
    pad_right             : int
    patch_size            : Tuple[int, int]
    stride                : Tuple[int, int]
    spatial_size          : Tuple[int, int]
    use_symmetric_padding : bool = True

    @property
    def padding_vertical(self) -> int:
        """Returns the total row padding in pixels."""
        return self.pad_top + self.pad_bot

    @property
    def padding_horizontal(self) -> int:
        """Returns the total range padding in pixels."""
        return self.pad_left + self.pad_right

    @property
    def number_of_patches(self) -> int:
        """Returns the number of patches on the grid."""
        return self.n_v * self.n_h

    @property
    def padded_size(self) -> Tuple[int, int]:
        """Returns the (rows, cols) extent of the region once padded."""
        H, W = self.spatial_size
        return (H + self.padding_vertical, W + self.padding_horizontal)

    def as_dict(self) -> dict:
        """Returns the grid geometry as a JSON-ready dictionary, including the patch count."""
        return {
            "n_v"                   : self.n_v,
            "n_h"                   : self.n_h,
            "pad_top"               : self.pad_top,
            "pad_bot"               : self.pad_bot,
            "pad_left"              : self.pad_left,
            "pad_right"             : self.pad_right,
            "patch_size"            : list(self.patch_size),
            "stride"                : list(self.stride),
            "spatial_size"          : list(self.spatial_size),
            "use_symmetric_padding" : self.use_symmetric_padding,
            "number_of_patches"     : self.number_of_patches,
        }


class Patcher:
    """Cuts padded patches out of a region array following a precomputed grid.

    Attributes:
        grid : Geometry of the patch grid this patcher realizes.
    """

    def __init__(self, grid: GridInfo, patch_coords: list) -> None:
        """Stores the grid geometry and the per-patch slice and padding specification.

        Args:
            grid         : Geometry of the patch grid.
            patch_coords : One entry per patch, each a tuple of the clipped row start and stop, the clipped range start and stop, and either None or a (top, bottom, left, right, mode) padding specification.
        """
        self.grid          = grid
        self._patch_coords = patch_coords

    @classmethod
    def build(cls, spatial_size : Tuple[int, int], patch_size : Tuple[int, int], stride : Tuple[int, int], use_symmetric_padding : bool = True) -> "Patcher":
        """Lays a strided patch grid over a region and precomputes every patch slice.

        The grid covers the whole region; the leftover extent is padded symmetrically
        around the region so no pixel is dropped at the borders.

        Args:
            spatial_size          : Region extent as (rows, cols) in pixels.
            patch_size            : Patch extent as (rows, cols) in pixels.
            stride                : Grid step as (rows, cols) in pixels.
            use_symmetric_padding : Whether border patches are mirror-padded rather than zero-padded.

        Returns:
            A Patcher holding the resulting GridInfo and patch coordinates.
        """
        ph, pw = patch_size
        sv, sh = stride
        H, W   = spatial_size

        n_v = 1 if H <= ph else math.ceil((H - ph) / sv) + 1
        n_h = 1 if W <= pw else math.ceil((W - pw) / sh) + 1

        pad_v = (ph + (n_v - 1) * sv) - H
        pad_h = (pw + (n_h - 1) * sh) - W

        pad_top, pad_bot    = pad_v // 2, pad_v - pad_v // 2
        pad_left, pad_right = pad_h // 2, pad_h - pad_h // 2

        grid = GridInfo(
            n_v                   = n_v,
            n_h                   = n_h,
            pad_top               = pad_top,
            pad_bot               = pad_bot,
            pad_left              = pad_left,
            pad_right             = pad_right,
            patch_size            = (ph, pw),
            stride                = (sv, sh),
            spatial_size          = (H, W),
            use_symmetric_padding = use_symmetric_padding,
        )

        mode         = "symmetric" if use_symmetric_padding else "constant"
        patch_coords = []

        for iv in range(n_v):
            for ih in range(n_h):
                v0 = iv * sv - pad_top
                h0 = ih * sh - pad_left
                v1 = v0 + ph
                h1 = h0 + pw

                pt = max(0, -v0)
                pb = max(0, v1 - H)
                pl = max(0, -h0)
                pr = max(0, h1 - W)

                v0c = max(0, v0)
                v1c = min(H, v1)
                h0c = max(0, h0)
                h1c = min(W, h1)

                pw_spec = None
                if pt or pb or pl or pr:
                    pw_spec = (pt, pb, pl, pr, mode)

                patch_coords.append((v0c, v1c, h0c, h1c, pw_spec))

        return cls(grid, patch_coords)

    def extract(self, array: np.ndarray, idx: int) -> np.ndarray:
        """Cuts one patch out of a region array, padding it at the borders when needed.

        Args:
            array : Region array whose last two axes are (rows, cols).
            idx   : Patch index on the grid.

        Returns:
            A contiguous array with the same leading axes and spatial shape equal to the
            configured patch size.
        """
        v0c, v1c, h0c, h1c, pw_spec = self._patch_coords[idx]
        sub                         = np.ascontiguousarray(array[..., v0c:v1c, h0c:h1c])

        if pw_spec is not None:
            pt, pb, pl, pr, mode = pw_spec
            pad_width = [(0, 0)] * (sub.ndim - 2) + [(pt, pb), (pl, pr)]
            sub       = np.pad(sub, pad_width, mode=mode)

        return sub
