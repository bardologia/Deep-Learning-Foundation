from __future__ import annotations

import numpy as np
import torch

from tools.data.gaussians import GaussianClamp


class ModelWrapper:
    def __init__(
        self,
        model,
        device,
        *,
        params_per_gaussian: int = 3,
        normalizer=None,
        x_axis: torch.Tensor | None = None,
        amp_max: float | None = None,
    ) -> None:

        self._model               = model
        self._device              = device
        self._params_per_gaussian = params_per_gaussian
        self._normalizer          = normalizer
        self._x_axis              = x_axis
        self._amp_max             = amp_max

    @property
    def module(self):
        return self._model

    @property
    def device(self):
        return self._device

    def denormalize_output(self, out: torch.Tensor) -> torch.Tensor:
        if self._normalizer is not None:
            out = self._normalizer.denormalize_output(out)

        if self._x_axis is not None and self._amp_max is not None:
            out = GaussianClamp.apply(
                out,
                x_axis      = self._x_axis.to(out.device),
                amp_max     = self._amp_max,
                ppg         = self._params_per_gaussian,
                leaky_slope = 0.0,
            )

        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self._device)

        with torch.no_grad():
            out = self._model(x_t)

        out = self.denormalize_output(out)

        return out.cpu().numpy()
