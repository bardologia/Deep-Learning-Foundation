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


class RngSnapshot:
    def __init__(self) -> None:
        self.python = random.getstate()
        self.numpy  = np.random.get_state()
        self.torch  = torch.get_rng_state()
        self.cuda   = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch)

        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda)


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
        parts = getattr(dataset, "parts", None) or [dataset]

        return [part.augmenter for part in parts if getattr(part, "augmenter", None) is not None]
