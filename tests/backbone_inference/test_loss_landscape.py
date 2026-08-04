from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.loss_landscape import FilterNormalizedDirection, LandscapeEvaluator, LossLandscapePlots


class _TinyConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.norm = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        return self.norm(self.conv(x))


def test_direction_is_filter_normalized_and_skips_1d_params():
    model     = _TinyConv()
    direction = FilterNormalizedDirection.build(model, seed=0)

    weight = model.conv.weight.detach().reshape(3, -1)
    d_conv = direction["conv.weight"].reshape(3, -1)

    assert torch.allclose(d_conv.norm(dim=1), weight.norm(dim=1), atol=1e-6)
    assert direction["conv.bias"].abs().sum() == 0
    assert direction["norm.weight"].abs().sum() == 0


def test_directions_differ_by_seed():
    model = _TinyConv()

    d0 = FilterNormalizedDirection.build(model, seed=0)
    d1 = FilterNormalizedDirection.build(model, seed=1)

    assert not torch.allclose(d0["conv.weight"], d1["conv.weight"])


def test_evaluator_restores_weights_and_center_is_minimum_for_quadratic():
    torch.manual_seed(0)

    model    = torch.nn.Linear(4, 1, bias=False)
    x        = torch.randn(64, 4)
    with torch.no_grad():
        target = model(x).clone()

    def loss_fn() -> float:
        with torch.no_grad():
            return float(((model(x) - target) ** 2).mean())

    original  = model.weight.detach().clone()
    direction = {"weight": torch.ones_like(model.weight)}
    zero      = {"weight": torch.zeros_like(model.weight)}

    evaluator = LandscapeEvaluator(model, loss_fn)
    alphas    = np.linspace(-0.5, 0.5, 11)
    cut       = evaluator.grid(alphas, np.zeros(1), direction, zero)[:, 0]

    assert torch.allclose(model.weight.detach(), original, atol=1e-7)
    assert cut[5] == pytest.approx(0.0, abs=1e-10)
    assert cut[0] > cut[5] and cut[-1] > cut[5]
    assert cut[0] == pytest.approx(cut[-1], rel=1e-5)


def test_zero_direction_model_raises():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3))

    with pytest.raises(ValueError):
        FilterNormalizedDirection.build(model, seed=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_direction_lives_on_the_parameter_device():
    model     = _TinyConv().cuda()
    direction = FilterNormalizedDirection.build(model, seed=0)

    for name, param in model.named_parameters():
        assert direction[name].device == param.device


def test_flat_landscape_contour_raises(tmp_path):
    alphas = np.linspace(-0.5, 0.5, 5)
    losses = np.full((alphas.size, alphas.size), 0.37)

    with pytest.raises(ValueError):
        LossLandscapePlots().contour_2d(alphas, alphas, losses, tmp_path / "contour.png")


def test_contour_of_a_varying_landscape_is_written(tmp_path):
    alphas = np.linspace(-0.5, 0.5, 5)
    losses = 0.1 + np.abs(alphas)[:, None] + np.abs(alphas)[None, :]

    path = LossLandscapePlots().contour_2d(alphas, alphas, losses, tmp_path / "contour.png")

    assert path.is_file()
    assert path.stat().st_size > 0
