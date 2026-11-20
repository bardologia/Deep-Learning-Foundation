from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


N_ROWS, N_COLS, PH, PW = 20, 16, 8, 8


class FirstChannelModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tap = torch.nn.Identity()

    def forward(self, x):
        x = self.tap(x)
        return torch.stack([x[:, 0] + 1.0, 5.0 * x[:, 0]], dim=1)


class TwoChannelModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tap = torch.nn.Identity()

    def forward(self, x):
        x = self.tap(x)
        return torch.stack([x[:, 0] + 1.0, 5.0 * x[:, 1]], dim=1)


class RowShiftModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tap = torch.nn.Identity()

    def forward(self, x):
        x = self.tap(x)
        return torch.stack([x[:, 0] + 1.0, 5.0 * torch.roll(x[:, 0], shifts=1, dims=1)], dim=1)


class TinyConvModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 8, 3, padding=1)
        self.act  = torch.nn.ReLU()
        self.head = torch.nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.head(self.act(self.conv(x)))


class Wrapper:

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    @staticmethod
    def denormalize_target(out):
        return out

    def __call__(self, x):
        with torch.no_grad():
            return self.module(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy()


def fake_run(model: torch.nn.Module | None = None):
    rng     = np.random.default_rng(0)
    inputs  = rng.uniform(0.5, 1.0, size=(2, N_ROWS, N_COLS)).astype(np.float32)
    module  = model if model is not None else FirstChannelModel()
    targets = Wrapper(module)(inputs[None])[0] + 0.25

    return SimpleNamespace(
        model           = Wrapper(module),
        dataset         = SimpleNamespace(window=lambda window: np.asarray(window, dtype=np.float32)),
        inputs          = inputs,
        targets         = targets,
        input_channels  = ["a", "b"],
        target_channels = ["first", "second"],
        in_channels     = 2,
        split_region    = SimpleNamespace(row_size=N_ROWS, col_size=N_COLS, row_start=0, col_start=0),
    )
