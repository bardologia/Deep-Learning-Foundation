"""Wraps a trained backbone so inference code can call it with NumPy patches.

The wrapper owns the device transfer, the no-grad forward, the inverse of the output
normalization and the physical Gaussian clamp, so every consumer sees denormalised,
in-range parameters.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.data.gaussians import GaussianClamp


class ModelWrapper:
    """Callable adapter turning a torch backbone into a NumPy-in, NumPy-out predictor."""

    def __init__(
        self,
        model,
        device,
        *,
        params_per_gaussian: int = 3,
        normalizer=None,
        z_axis: torch.Tensor | None = None,
        amp_max: float | None = None,
    ) -> None:
        """Stores the model, its device and the output post-processing pieces.

        Args:
            model               : Backbone module in eval mode, already on `device`.
            device              : Torch device the inputs are moved to.
            params_per_gaussian : Number of parameters per Gaussian in the output channels.
            normalizer          : Normalizer inverting the output scaling, or None to skip denormalisation.
            x_axis              : Elevation axis in metres, of shape (n_z,), bounding the μ clamp.
            amp_max             : Amplitude ceiling of the clamp, or None to leave outputs unclamped.
        """

        self._model               = model
        self._device              = device
        self._params_per_gaussian = params_per_gaussian
        self._normalizer          = normalizer
        self._x_axis              = z_axis
        self._amp_max             = amp_max

    @property
    def module(self):
        """The wrapped torch module."""

        return self._model

    @property
    def device(self):
        """The torch device the wrapped module runs on."""

        return self._device

    def denormalize_output(self, out: torch.Tensor) -> torch.Tensor:
        """Inverts the output normalization and applies the physical Gaussian clamp.

        Args:
            out : Raw model output of shape (batch, 3 * K, rows, cols).

        Returns:
            Denormalised parameters of the same shape, with μ inside the z axis,
            σ inside its physical bounds and amplitude below the ceiling when one is set.
        """

        if self._normalizer is not None:
            out = self._normalizer.denormalize_output(out)

        if self._x_axis is not None and self._amp_max is not None:
            out = GaussianClamp.apply(
                out,
                z_axis      = self._x_axis.to(out.device),
                amp_max     = self._amp_max,
                ppg         = self._params_per_gaussian,
                leaky_slope = 0.0,
            )

        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Predicts on a batch of input patches without tracking gradients.

        Args:
            x : Normalised input patches of shape (batch, channels, rows, cols).

        Returns:
            Denormalised, clamped output of shape (batch, 3 * K, rows, cols) on the CPU.
        """

        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self._device)

        with torch.no_grad():
            out = self._model(x_t)

        out = self.denormalize_output(out)

        return out.cpu().numpy()
