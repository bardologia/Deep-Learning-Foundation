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
    def __init__(
        self,
        inputs           : np.ndarray,
        gt_parameters    : np.ndarray,
        grid             : Patcher,
        input_config     : InputConfig,
        output_config    : OutputConfig,
        split_name       : str,
        n_secondaries    : int,
        n_interferograms : int,
        n_primary        : int                        = 1,
        normalizer       : Optional[Normalizer]       = None,
        n_gaussians      : int                        = 1,
        augmenter        : Optional[SpatialAugmenter] = None,
        dem              : Optional[np.ndarray]       = None,
        kz_field         : Optional[np.ndarray]       = None,
    ) -> None:

        self.inputs        = inputs
        self.gt_parameters = gt_parameters
        self.grid          = grid
        self.input_config  = input_config
        self.output_config = output_config
        self.split_name    = split_name
        self.normalizer    = normalizer
        self.n_gaussians   = n_gaussians
        self.augmenter     = augmenter
        self.dem           = dem
        self.kz_field      = kz_field
        self.input_layers  = int(inputs.shape[0])

        self.n_primary        = n_primary
        self.n_secondaries    = n_secondaries
        self.n_interferograms = n_interferograms

        expected_layers = n_primary + n_secondaries + n_interferograms
        if expected_layers != self.input_layers:
            raise ValueError(f"Input stack has {self.input_layers} layers but the dataset artifacts describe {expected_layers} ({n_primary} primary + {n_secondaries} secondaries + {n_interferograms} interferograms)")

        if input_config.use_primary and n_primary == 0:
            raise ValueError("Input config requests the primary but the cropped stack does not carry it. Rebuild the crop with the primary or set use_primary=False.")

        if input_config.use_secondaries and n_secondaries == 0:
            raise ValueError("Input config requests secondaries but the dataset stack contains none. Rebuild the dataset with secondaries or set use_secondaries=False.")

        if input_config.use_interferograms and n_interferograms == 0:
            raise ValueError("Input config requests interferograms but the dataset stack contains none. Rebuild the dataset with interferograms or set use_interferograms=False.")

        if input_config.use_dem and dem is None:
            raise ValueError("Input config requests the DEM channel but no DEM array was provided; pass the dem array from the cropper or set use_dem=False.")

        self.input_channels = input_config.total_channels(n_secondaries, n_interferograms)

        self.output_channel_indices = output_config.selected_indices(n_gaussians = n_gaussians)
        self.gt_channels            = len(self.output_channel_indices)

        available_gt_channels = int(gt_parameters.shape[0])
        required_gt_channels  = (max(self.output_channel_indices) + 1) if self.output_channel_indices else 0
        if required_gt_channels > available_gt_channels:
            raise ValueError(f"Configured n_gaussians={n_gaussians} indexes ground-truth parameter channel {required_gt_channels - 1} but the dataset provides only {available_gt_channels} ({available_gt_channels // 3} Gaussians). Set n_gaussians to match the parameter extraction used for this dataset.")

    def _build_input_tensor(self, complex_patch: np.ndarray, dem_patch: Optional[np.ndarray] = None) -> np.ndarray:
        ifg_offset          = self.n_primary + self.n_secondaries

        primary_data        = complex_patch[               : self.n_primary                        ]
        secondaries_data    = complex_patch[self.n_primary : ifg_offset                            ]
        interferograms_data = complex_patch[ifg_offset     : ifg_offset + self.n_interferograms    ]

        p_h, p_w     = complex_patch.shape[-2], complex_patch.shape[-1]
        input_tensor = np.empty((self.input_channels, p_h, p_w), dtype=np.float32)

        offset = 0

        if self.input_config.use_primary:
            n = self.input_config.primary_channels_per_pass
            self.input_config.primary_representation.convert_into(input_tensor[offset:offset + n], primary_data)
            offset += n

        if self.input_config.use_secondaries:
            n = self.n_secondaries * self.input_config.secondaries_channels_per_pass
            self.input_config.secondaries_representation.convert_into(input_tensor[offset:offset + n], secondaries_data)
            offset += n

        if self.input_config.use_interferograms:
            n = self.n_interferograms * self.input_config.interferograms_channels_per_pass
            self.input_config.interferograms_representation.convert_into(input_tensor[offset:offset + n], interferograms_data)
            offset += n

        if self.input_config.use_dem:
            input_tensor[offset] = dem_patch
            offset += 1

        return input_tensor

    def _build_output_tensor(self, gt_patch: np.ndarray) -> np.ndarray:
        output_tensor = gt_patch[self.output_channel_indices, ...]

        return np.ascontiguousarray(output_tensor, dtype=np.float32)

    def _normalize_input_tensor(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            raise RuntimeError(f"PatchDataset '{self.split_name}' was indexed before a normalizer was assigned; training on raw physical values would be silently wrong.")

        return self.normalizer.normalize_input(input_tensor)

    def _normalize_gt_params(self, gt_params: np.ndarray) -> np.ndarray:
        if self.normalizer.stats.output_stats is None:
            raise RuntimeError(f"PatchDataset '{self.split_name}' has a normalizer without output stats; ground-truth parameters cannot be normalized.")

        return self.normalizer.normalize_output(gt_params)

    def assemble_window(self, complex_window: np.ndarray, dem_window: Optional[np.ndarray] = None) -> np.ndarray:
        input_tensor = self._build_input_tensor(complex_window, dem_window)
        return self._normalize_input_tensor(input_tensor)

    def __len__(self) -> int:
        return self.grid.grid.number_of_patches

    def __getitem__(self, idx: int):
        complex_patch = self.grid.extract(self.inputs, idx)
        dem_patch     = self.grid.extract(self.dem, idx) if self.input_config.use_dem else None
        input_tensor  = self._build_input_tensor(complex_patch, dem_patch)

        gt_patch  = self.grid.extract(self.gt_parameters, idx)
        gt_params = self._build_output_tensor(gt_patch)

        kz_patch = self.grid.extract(self.kz_field, idx) if self.kz_field is not None else None

        if self.augmenter is not None and self.split_name == "train":
            if kz_patch is not None:
                input_tensor, gt_params, kz_patch = self.augmenter(input_tensor, gt_params, kz_patch)
            else:
                input_tensor, gt_params = self.augmenter(input_tensor, gt_params)

        input_tensor = self._normalize_input_tensor(input_tensor)
        gt_params    = self._normalize_gt_params(gt_params)

        if self.augmenter is not None and self.split_name == "train":
            input_tensor = self.augmenter.add_noise(input_tensor)

        if kz_patch is not None:
            return input_tensor, gt_params, np.ascontiguousarray(kz_patch)

        return input_tensor, gt_params


class MultiRegionDataset(Dataset):
    def __init__(self, parts: list[PatchDataset]) -> None:
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
                part.n_interferograms == first.n_interferograms and
                part.gt_channels      == first.gt_channels
            )
            if not matches:
                raise ValueError(f"MultiRegionDataset parts disagree on channel structure: part '{part.split_name}' has (input={part.input_channels}, primary={part.n_primary}, secondaries={part.n_secondaries}, interferograms={part.n_interferograms}, gt={part.gt_channels}) but the first part has (input={first.input_channels}, primary={first.n_primary}, secondaries={first.n_secondaries}, interferograms={first.n_interferograms}, gt={first.gt_channels}).")

        self.input_config           = first.input_config
        self.output_config          = first.output_config
        self.split_name             = first.split_name
        self.n_gaussians            = first.n_gaussians
        self.input_layers           = first.input_layers
        self.n_primary              = first.n_primary
        self.n_secondaries          = first.n_secondaries
        self.n_interferograms       = first.n_interferograms
        self.input_channels         = first.input_channels
        self.output_channel_indices = first.output_channel_indices
        self.gt_channels            = first.gt_channels

    @property
    def normalizer(self):
        return self.parts[0].normalizer

    @normalizer.setter
    def normalizer(self, normalizer) -> None:
        for part in self.parts:
            part.normalizer = normalizer

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} out of range for {self.total} patches")

        part_index = bisect.bisect_right(self.offsets, idx) - 1

        return self.parts[part_index][idx - self.offsets[part_index]]
