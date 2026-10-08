"""Tests for the filter-normalized loss-landscape probe around a trained checkpoint.

Covers direction construction and its seeding, the weight-restoring grid evaluator, and
the cut diagnostics (curvature, flatness radius, minimum offset, sharpness) plus the
contour plot writer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.diagnostics                            import LossLandscapeConfig
from pipelines.backbone.inference.analysis.loss_landscape import CutDiagnostics, FilterNormalizedDirection, LandscapeEvaluator, LossLandscapePlots, LossLandscapeRun

from tests.backbone_inference._helpers import StageLogger, build_run_directory
from tests.conftest                    import RecordingLogger, SilentLogger


class _TinyConv(torch.nn.Module):
    """Two-channel convolution followed by batch normalisation, used as a landscape probe target."""
    def __init__(self) -> None:
        """Builds the 3x3 convolution and the batch-norm layer."""
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.norm = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        """Returns the normalised convolution of x, an input of shape (batch, 2, height, width)."""
        return self.norm(self.conv(x))


def test_direction_is_filter_normalized_and_skips_1d_params():
    """Direction filters carry the norm of the matching weight filters, and one-dimensional parameters stay zero."""
    model     = _TinyConv()
    direction = FilterNormalizedDirection.build(model, seed=0)

    weight = model.conv.weight.detach().reshape(3, -1)
    d_conv = direction["conv.weight"].reshape(3, -1)

    assert torch.allclose(d_conv.norm(dim=1), weight.norm(dim=1), atol=1e-6)
    assert direction["conv.bias"].abs().sum() == 0
    assert direction["norm.weight"].abs().sum() == 0


def test_directions_differ_by_seed():
    """Different seeds give different random directions."""
    model = _TinyConv()

    d0 = FilterNormalizedDirection.build(model, seed=0)
    d1 = FilterNormalizedDirection.build(model, seed=1)

    assert not torch.allclose(d0["conv.weight"], d1["conv.weight"])


def test_evaluator_restores_weights_and_center_is_minimum_for_quadratic():
    """The grid restores the original weights and the quadratic cut is symmetric with its minimum at the centre."""
    torch.manual_seed(0)

    model    = torch.nn.Linear(4, 1, bias=False)
    x        = torch.randn(64, 4)
    with torch.no_grad():
        target = model(x).clone()

    def loss_fn() -> float:
        """Returns the mean squared error of the perturbed model against the frozen target."""
        with torch.no_grad():
            return float(((model(x) - target) ** 2).mean())

    original  = model.weight.detach().clone()
    direction = {"weight": torch.ones_like(model.weight)}
    zero      = {"weight": torch.zeros_like(model.weight)}

    evaluator = LandscapeEvaluator(model, loss_fn)
    alphas    = np.linspace(-0.5, 0.5, 11)
    cut       = evaluator.grid(alphas, np.zeros(1), direction, zero, lambda: None)[:, 0]

    assert torch.allclose(model.weight.detach(), original, atol=1e-7)
    assert cut[5] == pytest.approx(0.0, abs=1e-10)
    assert cut[0] > cut[5] and cut[-1] > cut[5]
    assert cut[0] == pytest.approx(cut[-1], rel=1e-5)


def test_zero_direction_model_raises():
    """A model with only one-dimensional parameters yields no direction and raises ValueError."""
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3))

    with pytest.raises(ValueError):
        FilterNormalizedDirection.build(model, seed=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_direction_lives_on_the_parameter_device():
    """Direction tensors are created on the device of the parameters they perturb."""
    model     = _TinyConv().cuda()
    direction = FilterNormalizedDirection.build(model, seed=0)

    for name, param in model.named_parameters():
        assert direction[name].device == param.device


def test_curvature_recovers_a_quadratic_bowl():
    """Curvature of an analytic quadratic cut recovers its quadratic coefficient."""
    alphas = np.linspace(-0.5, 0.5, 21)
    cut    = 1.0 + 3.0 * alphas ** 2

    assert CutDiagnostics.curvature(alphas, cut) == pytest.approx(3.0, rel=1e-6)


def test_flatness_radius_interpolates_the_doubling_step():
    """The flatness radius interpolates the step at which the loss doubles and reports it uncensored."""
    alphas = np.linspace(-0.5, 0.5, 21)
    cut    = 1.0 + 10.0 * np.abs(alphas)

    radius = CutDiagnostics.flatness_radius(alphas, cut)

    assert radius["radius"]   == pytest.approx(0.1, abs=1e-9)
    assert radius["censored"] is False


def test_flatness_radius_censors_a_flat_cut():
    """A cut that never doubles reports the largest sampled step as censored."""
    alphas = np.linspace(-0.5, 0.5, 21)
    cut    = np.full(21, 1.0)
    cut[10] = 0.999

    radius = CutDiagnostics.flatness_radius(alphas, cut)

    assert radius["radius"]   == pytest.approx(0.5)
    assert radius["censored"] is True


def test_min_offset_flags_a_shifted_minimum():
    """The minimum offset recovers the displacement of a shifted quadratic."""
    alphas = np.linspace(-0.5, 0.5, 21)
    cut    = (alphas - 0.2) ** 2

    assert CutDiagnostics.min_offset(alphas, cut) == pytest.approx(0.2, abs=0.05)


def test_sharpness_measures_the_near_field_increase():
    """Sharpness of a V-shaped cut equals the loss increase over the near-field step."""
    alphas = np.linspace(-0.5, 0.5, 21)
    cut    = 1.0 + 4.0 * np.abs(alphas)

    assert CutDiagnostics.sharpness(alphas, cut) == pytest.approx(0.4, rel=1e-6)


def test_flat_landscape_contour_raises(tmp_path):
    """A constant loss surface has no contour levels and raises ValueError."""
    alphas = np.linspace(-0.5, 0.5, 5)
    losses = np.full((alphas.size, alphas.size), 0.37)

    with pytest.raises(ValueError):
        LossLandscapePlots().contour_2d(alphas, alphas, losses, tmp_path / "contour.png")


def test_contour_of_a_varying_landscape_is_written(tmp_path):
    """A varying loss surface writes a non-empty contour figure."""
    alphas = np.linspace(-0.5, 0.5, 5)
    losses = 0.1 + np.abs(alphas)[:, None] + np.abs(alphas)[None, :]

    path = LossLandscapePlots().contour_2d(alphas, alphas, losses, tmp_path / "contour.png")

    assert path.is_file()
    assert path.stat().st_size > 0


def test_single_direction_is_refused_before_the_run_loads(tmp_path):
    """One direction leaves no plane for the 2-D grid, so the analysis refuses it at construction."""
    with pytest.raises(ValueError, match="at least 2 directions"):
        LossLandscapeRun(tmp_path, LossLandscapeConfig(n_directions=1), SilentLogger())


def test_grid_reports_every_evaluated_point():
    """The progress callback fires once per evaluated grid point."""
    model     = torch.nn.Linear(2, 1)
    direction = {name: torch.ones_like(param) for name, param in model.named_parameters()}
    points    = []

    LandscapeEvaluator(model, lambda: 1.0).grid(np.linspace(-0.1, 0.1, 3), np.linspace(-0.1, 0.1, 2), direction, direction, lambda: points.append(1))

    assert len(points) == 6


@pytest.mark.dataset
@pytest.mark.slow
def test_landscape_pass_announces_tracks_and_times_its_stages(tmp_path, test_data_dir, params_dir):
    """A landscape pass on a real run announces its evaluation count, advances one bar per point and times the cuts and the grid."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    config  = LossLandscapeConfig(batch_size=2, max_batches=1, n_directions=2, n_points_1d=11, n_points_2d=3)
    logger  = StageLogger()

    outcome = LossLandscapeRun(run_dir, config, logger).run()

    assert outcome.payload["n_batches"] == 1
    assert any(message.endswith("31 loss evaluations planned (2 cuts x 11 points + 3x3 grid)") for level, message in logger.messages if level == "info")
    assert logger.totals == [31] and logger.advances == 31
    assert logger.timers == ["Loss landscape: 1-D cuts (22 points)", "Loss landscape: 2-D grid (9 points)"]


def test_cut_descriptors_refuse_a_non_finite_cut():
    """A NaN in a cut raises instead of reading as a censored radius or a minimum at the NaN step."""
    alphas = np.linspace(-0.5, 0.5, 11)
    cut    = 1.0 + alphas ** 2
    cut[8] = np.nan

    for descriptor in (CutDiagnostics.sharpness, CutDiagnostics.curvature, CutDiagnostics.flatness_radius, CutDiagnostics.min_offset):
        with pytest.raises(ValueError, match="non-finite value"):
            descriptor(alphas, cut)


def test_diagnose_flags_a_diverged_cut_and_warns(tmp_path):
    """A cut holding NaN is marked invalid with its steps and named in one warning, while the finite cut is described."""
    logger   = RecordingLogger()
    alphas   = np.linspace(-0.5, 0.5, 11)
    finite   = 1.0 + alphas ** 2
    diverged = finite.copy()

    diverged[-1] = np.inf

    diagnostics = LossLandscapeRun(tmp_path, LossLandscapeConfig(), logger)._diagnose(alphas, [finite, diverged])

    warnings = [message for level, message in logger.messages if level == "warning"]

    assert diagnostics[0]["valid"] and diagnostics[0]["min_offset"] == 0.0
    assert diagnostics[1] == {"valid": False, "nonfinite_alphas": [0.5]}
    assert len(warnings) == 1 and "direction 1: 1/11 cut losses are non-finite at alpha +0.500" in warnings[0]


def test_diagnose_refuses_when_every_cut_diverged(tmp_path):
    """With no finite cut left there is nothing to summarize and the pass raises."""
    cut = np.full(11, np.nan)

    with pytest.raises(ValueError, match="Every one of the 1 loss cuts"):
        LossLandscapeRun(tmp_path, LossLandscapeConfig(), RecordingLogger())._diagnose(np.linspace(-0.5, 0.5, 11), [cut])


def test_non_finite_grid_contour_names_the_divergence(tmp_path):
    """A NaN inside the 2-D grid is reported as divergence, not as a flat landscape."""
    alphas       = np.linspace(-0.5, 0.5, 5)
    losses       = 0.1 + np.abs(alphas)[:, None] + np.abs(alphas)[None, :]
    losses[0, 0] = np.nan

    with pytest.raises(ValueError, match="1 of 25 landscape grid losses are non-finite"):
        LossLandscapePlots().contour_2d(alphas, alphas, losses, tmp_path / "contour.png")
