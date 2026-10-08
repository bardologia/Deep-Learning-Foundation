"""Torch datasets serving patched channel stacks and Gaussian parameter targets.

PatchDataset assembles the configured input representation for one crop region and
MultiRegionDataset concatenates several such regions into a single split.
"""

from __future__ import annotations

import bisect
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from configuration.dataset                   import InputConfig, OutputConfig
from pipelines.backbone.dataset.augmentation import SpatialAugmenter
from pipelines.backbone.dataset.normalizer   import Normalizer
from pipelines.backbone.dataset.spatial      import Patcher


class PatchDataset(Dataset):
    """Serves normalized input patches and Gaussian parameter targets from one crop region.

    The dataset holds the whole region in memory (or memory-mapped), cuts patches on
    demand through the Patcher grid, converts the complex stack into the configured
    real-valued representation, augments the train split and normalizes both sides.

    Attributes:
        inputs                 : Complex channel stack of shape (layers, rows, cols) for this region.
        gt_parameters          : Ground-truth Gaussian parameters of shape (3 * n_gaussians, rows, cols).
        grid                   : Patcher defining the patch grid and padding over the region.
        normalizer             : Normalizer applied to inputs and targets; may be assigned after construction.
        input_channels         : Number of channels the assembled input tensor carries.
        gt_channels            : Number of ground-truth channels retained by the output configuration.
        output_channel_indices : Indices into the full parameter stack that the output configuration selects.
    """

    def __init__(
        self,
        inputs        : np.ndarray,
        gt_parameters : np.ndarray,
        grid          : Patcher,
        input_config  : InputConfig,
        output_config : OutputConfig,
        split_name    : str,
        n_secondaries : int,
        n_derived     : int,
        n_primary     : int                        = 1,
        normalizer    : Optional[Normalizer]       = None,
        n_gaussians   : int                        = 1,
        augmenter     : Optional[SpatialAugmenter] = None,
        aux           : Optional[np.ndarray]       = None,
    ) -> None:
        """Validates the stack layout against the input configuration and selects the target channels.

        Args:
            inputs           : Complex channel stack of shape (n_primary + n_secondaries + n_derived, rows, cols).
            gt_parameters    : Ground-truth parameters of shape (3 * available_gaussians, rows, cols), ordered amplitude, mu (metres), sigma (metres) per Gaussian slot.
            grid             : Patcher supplying the patch coordinates and padding.
            input_config     : Selection and representation of the primary, secondaries, derived channels and aux field.
            output_config    : Selection of the ground-truth parameter channels.
            split_name       : Split this dataset belongs to; augmentation runs only for "train".
            n_secondaries    : Number of secondary channels carried by the stack.
            n_derived        : Number of derived channel layers carried by the stack.
            n_primary        : Number of primary layers carried by the stack.
            normalizer       : Normalizer for inputs and targets; may be attached later.
            n_gaussians      : Number of Gaussian slots the output configuration indexes.
            augmenter        : Augmenter applied to train patches, or None to disable augmentation.
            aux              : Optional aux field of shape (rows, cols) in metres.

        Raises:
            ValueError : If the stack layer count disagrees with the declared composition, if the input configuration requests a group the stack does not carry, or if n_gaussians indexes parameter channels beyond those provided.
        """

        self.inputs        = inputs
        self.gt_parameters = gt_parameters
        self.grid          = grid
        self.input_config  = input_config
        self.output_config = output_config
        self.split_name    = split_name
        self.normalizer    = normalizer
        self.n_gaussians   = n_gaussians
        self.augmenter     = augmenter
        self.aux           = aux
        self.input_layers  = int(inputs.shape[0])

        self.n_primary     = n_primary
        self.n_secondaries = n_secondaries
        self.n_derived     = n_derived

        expected_layers = n_primary + n_secondaries + n_derived
        if expected_layers != self.input_layers:
            raise ValueError(f"Input stack has {self.input_layers} layers but the dataset artifacts describe {expected_layers} ({n_primary} primary + {n_secondaries} secondaries + {n_derived} derived)")

        if input_config.use_primary and n_primary == 0:
            raise ValueError("Input config requests the primary but the cropped stack does not carry it. Rebuild the crop with the primary or set use_primary=False.")

        if input_config.use_secondaries and n_secondaries == 0:
            raise ValueError("Input config requests secondaries but the dataset stack contains none. Rebuild the dataset with secondaries or set use_secondaries=False.")

        if input_config.use_derived and n_derived == 0:
            raise ValueError("Input config requests derived but the dataset stack contains none. Rebuild the dataset with derived or set use_derived=False.")

        if input_config.use_aux and aux is None:
            raise ValueError("Input config requests the Aux channel but no Aux array was provided; pass the aux array from the cropper or set use_aux=False.")

        self.input_channels = input_config.total_channels(n_secondaries, n_derived)

        self.output_channel_indices = output_config.selected_indices(n_gaussians = n_gaussians)
        self.gt_channels            = len(self.output_channel_indices)

        available_gt_channels = int(gt_parameters.shape[0])
        required_gt_channels  = (max(self.output_channel_indices) + 1) if self.output_channel_indices else 0
        if required_gt_channels > available_gt_channels:
            raise ValueError(f"Configured n_gaussians={n_gaussians} indexes ground-truth parameter channel {required_gt_channels - 1} but the dataset provides only {available_gt_channels} ({available_gt_channels // 3} Gaussians). Set n_gaussians to match the parameter extraction used for this dataset.")

        self.gt_selected = self._select_gt_channels(gt_parameters)

    def _select_gt_channels(self, gt_parameters: np.ndarray) -> np.ndarray:
        """Returns the leading gt_channels parameter maps of the ground-truth stack."""
        return gt_parameters[:self.gt_channels]

    def _build_input_tensor(self, complex_patch: np.ndarray, aux_patch: Optional[np.ndarray] = None) -> np.ndarray:
        """Converts a complex patch into the configured real-valued input tensor.

        Args:
            complex_patch : Complex patch of shape (layers, patch_h, patch_w) sliced from the stack.
            aux_patch     : Aux patch of shape (patch_h, patch_w), required when use_aux is set.

        Returns:
            Float32 tensor of shape (input_channels, patch_h, patch_w) with the primary,
            secondary, derived and aux channels written in that order.
        """
        derived_offset          = self.n_primary + self.n_secondaries

        primary_data     = complex_patch[               : self.n_primary                        ]
        secondaries_data = complex_patch[self.n_primary : derived_offset                            ]
        derived_data     = complex_patch[derived_offset     : derived_offset + self.n_derived    ]

        p_h, p_w     = complex_patch.shape[-2], complex_patch.shape[-1]
        input_tensor = np.empty((self.input_channels, p_h, p_w), dtype=np.float32)

        offset = 0

        if self.input_config.use_primary:
            n = self.input_config.primary_channels_per_source
            self.input_config.primary_representation.convert_into(input_tensor[offset:offset + n], primary_data)
            offset += n

        if self.input_config.use_secondaries:
            n = self.n_secondaries * self.input_config.secondaries_channels_per_source
            self.input_config.secondaries_representation.convert_into(input_tensor[offset:offset + n], secondaries_data)
            offset += n

        if self.input_config.use_derived:
            n = self.n_derived * self.input_config.derived_channels_per_source
            self.input_config.derived_representation.convert_into(input_tensor[offset:offset + n], derived_data)
            offset += n

        if self.input_config.use_aux:
            input_tensor[offset] = aux_patch
            offset += 1

        return input_tensor

    def _normalize_input_tensor(self, input_tensor: np.ndarray) -> np.ndarray:
        """Normalizes an assembled input tensor.

        Args:
            input_tensor : Tensor of shape (input_channels, patch_h, patch_w) in physical units.

        Returns:
            The tensor in normalized units, same shape.

        Raises:
            RuntimeError : If no normalizer has been attached to this dataset yet.
        """
        if self.normalizer is None:
            raise RuntimeError(f"PatchDataset '{self.split_name}' was indexed before a normalizer was assigned; training on raw physical values would be silently wrong.")

        return self.normalizer.normalize_input(input_tensor)

    def _normalize_gt_params(self, gt_params: np.ndarray) -> np.ndarray:
        """Normalizes a ground-truth parameter patch.

        Args:
            gt_params : Parameters of shape (gt_channels, patch_h, patch_w) in physical units.

        Returns:
            The parameters in normalized units, same shape.

        Raises:
            RuntimeError : If the attached normalizer carries no output statistics.
        """
        if self.normalizer.stats.output_stats is None:
            raise RuntimeError(f"PatchDataset '{self.split_name}' has a normalizer without output stats; ground-truth parameters cannot be normalized.")

        return self.normalizer.normalize_output(gt_params)

    def assemble_window(self, complex_window: np.ndarray, aux_window: Optional[np.ndarray] = None) -> np.ndarray:
        """Builds and normalizes an input tensor for an arbitrary window outside the patch grid.

        Args:
            complex_window : Complex window of shape (layers, z, width).
            aux_window     : Aux window of shape (z, width) in metres, when the Aux channel is enabled.

        Returns:
            Normalized tensor of shape (input_channels, z, width).
        """
        input_tensor = self._build_input_tensor(complex_window, aux_window)
        return self._normalize_input_tensor(input_tensor)

    def __len__(self) -> int:
        """Returns the number of patches on this region's grid."""
        return self.grid.grid.number_of_patches

    def __getitem__(self, idx: int):
        """Returns one training sample.

        Args:
            idx : Patch index on the region grid.

        Returns:
            A (input_tensor, gt_params) pair of normalized arrays with shapes
            (input_channels, patch_h, patch_w) and (gt_channels, patch_h, patch_w).
        """
        complex_patch = self.grid.extract(self.inputs, idx)
        aux_patch     = self.grid.extract(self.aux, idx) if self.input_config.use_aux else None
        input_tensor  = self._build_input_tensor(complex_patch, aux_patch)

        gt_params = self.grid.extract(self.gt_selected, idx).astype(np.float32, copy=False)

        if self.augmenter is not None and self.split_name == "train":
            input_tensor, gt_params = self.augmenter(input_tensor, gt_params)

        input_tensor = self._normalize_input_tensor(input_tensor)
        gt_params    = self._normalize_gt_params(gt_params)

        if self.augmenter is not None and self.split_name == "train":
            input_tensor = self.augmenter.add_noise(input_tensor)

        return input_tensor, gt_params


class MultiRegionDataset(Dataset):
    """Concatenates the patch grids of several disjoint crop regions into one split.

    Attributes:
        parts   : Per-region PatchDataset instances, in concatenation order.
        offsets : Index of the first patch of each part in the concatenated ordering.
        total   : Total number of patches across all parts.
    """

    def __init__(self, parts: list[PatchDataset]) -> None:
        """Builds the concatenated index and checks the parts share a channel structure.

        Args:
            parts : Per-region datasets to concatenate.

        Raises:
            ValueError : If no part is given, or if the parts disagree on the input channel count, the stack composition or the ground-truth channel count.
        """
        if not parts:
            raise ValueError("MultiRegionDataset requires at least one part")

        self.parts   = parts
        self.offsets = []

        total = 0
        for part in parts:
            self.offsets.append(total)
            total += len(part)
        self.total = total

        first = parts[0]

        for part in parts[1:]:
            matches = (
                part.input_channels   == first.input_channels   and
                part.n_primary        == first.n_primary        and
                part.n_secondaries    == first.n_secondaries    and
                part.n_derived == first.n_derived and
                part.gt_channels      == first.gt_channels
            )
            if not matches:
                raise ValueError(f"MultiRegionDataset parts disagree on channel structure: part '{part.split_name}' has (input={part.input_channels}, primary={part.n_primary}, secondaries={part.n_secondaries}, derived={part.n_derived}, gt={part.gt_channels}) but the first part has (input={first.input_channels}, primary={first.n_primary}, secondaries={first.n_secondaries}, derived={first.n_derived}, gt={first.gt_channels}).")

        self.input_config           = first.input_config
        self.output_config          = first.output_config
        self.split_name             = first.split_name
        self.n_gaussians            = first.n_gaussians
        self.input_layers           = first.input_layers
        self.n_primary              = first.n_primary
        self.n_secondaries          = first.n_secondaries
        self.n_derived              = first.n_derived
        self.input_channels         = first.input_channels
        self.output_channel_indices = first.output_channel_indices
        self.gt_channels            = first.gt_channels

    @property
    def normalizer(self):
        """Returns the normalizer shared by the parts."""
        return self.parts[0].normalizer

    @normalizer.setter
    def normalizer(self, normalizer) -> None:
        """Assigns the normalizer to every part."""
        for part in self.parts:
            part.normalizer = normalizer

    def __len__(self) -> int:
        """Returns the total number of patches across all regions."""
        return self.total

    def __getitem__(self, idx: int):
        """Returns the sample at a concatenated index.

        Args:
            idx : Index into the concatenated patch ordering; negative values count from the end.

        Returns:
            The sample the owning part yields for its local index.

        Raises:
            IndexError : If the index falls outside the concatenated range.
        """
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} out of range for {self.total} patches")

        part_index = bisect.bisect_right(self.offsets, idx) - 1

        return self.parts[part_index][idx - self.offsets[part_index]]
