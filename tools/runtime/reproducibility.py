
from __future__ import annotations

import random

import numpy as np
import torch


class Reproducibility:

    SEED_MODULUS = 2 ** 32

    @staticmethod
    def seed_everything(seed: int) -> None:
        seed = int(seed)

        random.seed(seed)
        np.random.seed(seed % Reproducibility.SEED_MODULUS)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    @staticmethod
    def generator(seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        return generator

    @staticmethod
    def worker_init(base_seed: int):
        return WorkerInitializer(base_seed)


class WorkerInitializer:

    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        seed = (self.base_seed + int(worker_id)) % Reproducibility.SEED_MODULUS

        random.seed(seed)
        np.random.seed(seed)

        info = torch.utils.data.get_worker_info()
        if info is None:
            return

        for augmenter in self._augmenters(info.dataset):
            augmenter.reseed(seed)

    @staticmethod
    def _augmenters(dataset) -> list:
        return [part.augmenter for part in dataset.parts if part.augmenter is not None]
