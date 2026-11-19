
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


N_ROWS, N_COLS, PH, PW, N_Z = 20, 16, 8, 8, 12


class BareModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tap = torch.nn.Identity()

    def forward(self, x):
        x          = self.tap(x)
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + 1.0
        out[:, 1] = 5.0 * x[:, 0]
        out[:, 2] = 3.0 + 0.0 * x[:, 0]

        return out


class TwoSlotModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tap = torch.nn.Identity()

    def forward(self, x):
        x          = self.tap(x)
        B, _, H, W = x.shape
        out        = torch.zeros(B, 6, H, W)

        out[:, 0] = x[:, 0] + 1.0
        out[:, 1] = 5.0 * x[:, 0]
        out[:, 2] = 3.0 + 0.0 * x[:, 0]
        out[:, 3] = x[:, 1] + 1.0
        out[:, 4] = 5.0 * x[:, 1]
        out[:, 5] = 3.0 + 0.0 * x[:, 1]

        return out


class AzShiftModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tap = torch.nn.Identity()

    def forward(self, x):
        x          = self.tap(x)
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + 1.0
        out[:, 1] = 5.0 * torch.roll(x[:, 0], shifts=1, dims=1)
        out[:, 2] = 3.0 + 0.0 * x[:, 0]

        return out


class TinyConvModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 8, 3, padding=1)
        self.act  = torch.nn.ReLU()
        self.head = torch.nn.Conv2d(8, 3, 1)

    def forward(self, x):
        return self.head(self.act(self.conv(x)))


class Wrapper:

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    def __call__(self, x):
        with torch.no_grad():
            return self.module(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy()


def fake_run(n_slots: int = 1, with_gt: bool = True):
    rng            = np.random.default_rng(0)
    complex_inputs = (rng.uniform(0.5, 1.0, size=(2, N_ROWS, N_COLS)) + 1j * rng.uniform(0.0, 0.2, size=(2, N_ROWS, N_COLS))).astype(np.complex64)
    gt_params      = np.zeros((3 * n_slots, N_ROWS, N_COLS), dtype=np.float32)

    for k in range(n_slots):
        gt_params[3 * k]     = 1.5
        gt_params[3 * k + 1] = 20.0
        gt_params[3 * k + 2] = 4.0

    dataset = SimpleNamespace(
        aux             = None,
        gt_parameters   = gt_params if with_gt else None,
        assemble_window = lambda window, aux: np.abs(window).astype(np.float32),
    )

    return SimpleNamespace(
        model              = Wrapper(BareModel() if n_slots == 1 else TwoSlotModel()),
        dataset            = dataset,
        complex_inputs     = complex_inputs,
        full_curves        = np.ones((N_Z, N_ROWS, N_COLS), dtype=np.float32),
        z_axis             = np.linspace(-10.0, 40.0, N_Z).astype(np.float32),
        n_gaussians        = n_slots,
        activity_threshold = 1e-4,
        in_channels        = 2,
        split_region       = SimpleNamespace(row_size=N_ROWS, col_size=N_COLS, row_start=0, col_start=0),
    )
