from __future__ import annotations

import numpy as np
import torch

from pipelines.backbone.dataset.normalizer import Normalizer


class ModelWrapper:

    def __init__(self, model: torch.nn.Module, device, normalizer: Normalizer) -> None:
        self._model     = model
        self._device    = device
        self.normalizer = normalizer

    @property
    def module(self) -> torch.nn.Module:
        return self._model

    @property
    def device(self):
        return self._device

    def denormalize_target(self, out: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize_target(out)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self._device)

        with torch.no_grad():
            out = self._model(inputs)

        return self.denormalize_target(out).cpu().numpy()
