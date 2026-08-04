from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import GammaNetConfig
from ..blocks                    import build_activation


class TomoOperator:
    def __init__(self, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float) -> None:
        spacings = torch.diff(x_axis)
        if not torch.allclose(spacings, torch.full_like(spacings, dx), rtol=1e-3, atol=0.0):
            raise ValueError("TomoOperator requires a uniform elevation axis; the Toeplitz gram kernel is only valid when x_axis spacing is constant")

        self.dx     = dx
        self.length = int(x_axis.shape[0])
        self.pixels = kz_map.permute(0, 2, 3, 1)
        self.kernel = None

        phase         = self.pixels.unsqueeze(-1) * x_axis.reshape(1, 1, 1, 1, -1)
        self.steering = torch.polar(torch.ones_like(phase), phase)

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        pixels = profiles.to(self.steering.dtype).unsqueeze(-1)

        return (self.steering @ pixels).squeeze(-1) * self.dx

    def adjoint(self, measurements: torch.Tensor) -> torch.Tensor:
        pixels = measurements.unsqueeze(-1)

        return (self.steering.mH @ pixels).real.squeeze(-1)

    def _toeplitz_kernel(self) -> torch.Tensor:
        if self.kernel is None:
            offsets = torch.arange(-(self.length - 1), self.length, device=self.pixels.device, dtype=self.pixels.dtype) * self.dx

            accumulated = None
            for track in range(self.pixels.shape[-1]):
                term        = torch.cos(self.pixels[..., track].unsqueeze(-1) * offsets)
                accumulated = term if accumulated is None else accumulated + term

            self.kernel = accumulated

        return self.kernel

    def gram(self, profiles: torch.Tensor) -> torch.Tensor:
        B, H, W, L = profiles.shape

        flat   = profiles.reshape(1, B * H * W, L)
        weight = self._toeplitz_kernel().reshape(B * H * W, 1, 2 * L - 1)

        return functional.conv1d(flat, weight, padding=L - 1, groups=B * H * W).reshape(B, H, W, L) * self.dx


class ProfileProx(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, activation: str):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2

        self.down       = nn.Conv1d(kernel_size, hidden, kernel_size=1)
        self.activation = build_activation(activation)
        self.up         = nn.Conv1d(hidden, 1, kernel_size=kernel_size, padding=self.padding)

    def _strided_windows(self, flat: torch.Tensor) -> torch.Tensor:
        pixels, _, length = flat.shape

        padded      = functional.pad(flat, (self.padding, self.padding))
        width       = length + 2 * self.padding
        half_length = (width - self.kernel_size) // 2 + 1

        return padded.as_strided((pixels, self.kernel_size, half_length), (width, 1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, N = x.shape

        flat       = x.reshape(B * H * W, 1, N)
        correction = self.up(self.activation(self.down(self._strided_windows(flat))))
        refined    = flat + functional.interpolate(correction, size=N, mode="linear", align_corners=False)

        return refined.reshape(B, H, W, N)


class GammaNet(nn.Module):
    def __init__(self, config: GammaNetConfig | None = None):
        super().__init__()
        self.config = config if config is not None else GammaNetConfig()

        if self.config.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.config.n_iterations}")

        L = self.config.n_iterations

        self.raw_steps      = nn.Parameter(torch.full((L,), self._softplus_inverse(self.config.step_init)))
        self.raw_thresholds = nn.Parameter(torch.full((L,), self._softplus_inverse(self.config.threshold_init)))
        self.prox_blocks    = nn.ModuleList([ProfileProx(self.config.prox_hidden, self.config.prox_kernel_size, self.config.activation) for _ in range(L)])

    @staticmethod
    def _softplus_inverse(value: float) -> float:
        return math.log(math.expm1(value))

    def forward(self, measurements: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor) -> torch.Tensor:
        if not measurements.is_complex():
            raise ValueError("GammaNet expects complex-valued coherence measurements of shape (B, T, H, W)")
        if measurements.shape != kz_map.shape:
            raise ValueError(f"measurements {tuple(measurements.shape)} and kz_map {tuple(kz_map.shape)} must share the (B, T, H, W) shape")

        dx        = float(x_axis[1] - x_axis[0])
        n_tracks  = kz_map.shape[1]
        lipschitz = n_tracks * x_axis.shape[0] * dx * dx

        steps      = functional.softplus(self.raw_steps)
        thresholds = functional.softplus(self.raw_thresholds)

        operator       = TomoOperator(kz_map, x_axis, dx)
        measured       = measurements.permute(0, 2, 3, 1).contiguous()
        backprojection = operator.adjoint(measured)
        profile        = functional.relu(backprojection / n_tracks)

        for iteration in range(self.config.n_iterations):
            gradient = (backprojection - operator.gram(profile)) / lipschitz

            profile = profile + steps[iteration] * gradient
            profile = self.prox_blocks[iteration](profile)
            profile = functional.relu(profile - thresholds[iteration])

        return profile.permute(0, 3, 1, 2)
