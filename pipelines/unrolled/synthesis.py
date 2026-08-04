from __future__ import annotations

import torch

from models.unrolled      import TomoOperator
from tools.data.gaussians import GaussianCurve


class MeasurementSynthesiser:
    def __init__(self, x_axis: torch.Tensor, ppg: int, power_floor: float, noise_std: float) -> None:
        self.x_axis      = x_axis
        self.ppg         = ppg
        self.power_floor = power_floor
        self.noise_std   = noise_std
        self.dx          = float(x_axis[1] - x_axis[0])

    @torch.no_grad()
    def synthesise(self, gt_phys: torch.Tensor, kz_map: torch.Tensor, generator: torch.Generator | None = None) -> tuple:
        curves = GaussianCurve.reconstruct(gt_phys, self.x_axis, self.ppg).float()

        power = curves.sum(dim=1) * self.dx
        mask  = power > self.power_floor

        target       = curves / power.clamp(min=self.power_floor).unsqueeze(1)
        measurements = TomoOperator(kz_map, self.x_axis, self.dx).forward(target.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.noise_std > 0.0:
            shape        = measurements.real.shape
            device       = measurements.device
            dtype        = measurements.real.dtype
            noise        = torch.randn(shape, generator=generator, device=device, dtype=dtype) + 1j * torch.randn(shape, generator=generator, device=device, dtype=dtype)
            measurements = measurements + self.noise_std / (2.0 ** 0.5) * noise

        return measurements, target, mask
