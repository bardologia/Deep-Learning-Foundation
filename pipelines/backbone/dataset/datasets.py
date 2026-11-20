from __future__ import annotations

import bisect
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from pipelines.backbone.dataset.augmentation import SpatialAugmenter
from pipelines.backbone.dataset.normalizer   import Normalizer
from pipelines.backbone.dataset.spatial      import Patcher


class PatchDataset(Dataset):

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, patcher: Patcher, normalizer: Normalizer, count_once: bool, augmenter: Optional[SpatialAugmenter] = None) -> None:
        self.inputs     = inputs
        self.targets    = targets
        self.patcher    = patcher
        self.normalizer = normalizer
        self.count_once = bool(count_once)
        self.augmenter  = augmenter
        self.weight_map = patcher.weight_map(self.count_once)

        self.input_channels  = int(inputs.shape[0])
        self.target_channels = int(targets.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.inputs.nbytes + self.targets.nbytes + self.weight_map.nbytes)

    def window(self, inputs_window: np.ndarray) -> np.ndarray:
        return self.normalizer.normalize_input(inputs_window)

    def __len__(self) -> int:
        return self.patcher.grid.number_of_patches

    def __getitem__(self, idx: int):
        inputs  = self.patcher.extract(self.inputs, idx)
        targets = self.patcher.extract(self.targets, idx)
        weights = self.patcher.cut_weights(self.weight_map, idx)

        if self.augmenter is not None:
            inputs, targets, weights = self.augmenter(inputs, targets, weights)

        inputs  = self.normalizer.normalize_input(inputs)
        targets = self.normalizer.normalize_target(targets)

        if self.augmenter is not None:
            inputs = self.augmenter.add_noise(inputs)

        return inputs, targets, weights


class MultiRegionDataset(Dataset):

    def __init__(self, parts: list[PatchDataset]) -> None:
        self.parts   = parts
        self.offsets = []

        total = 0
        for part in parts:
            self.offsets.append(total)
            total += len(part)
        self.total = total

        self.input_channels  = parts[0].input_channels
        self.target_channels = parts[0].target_channels

    @property
    def normalizer(self) -> Normalizer:
        return self.parts[0].normalizer

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.total

        part_index = bisect.bisect_right(self.offsets, idx) - 1

        return self.parts[part_index][idx - self.offsets[part_index]]
