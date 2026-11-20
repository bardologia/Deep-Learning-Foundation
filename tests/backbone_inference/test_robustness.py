
from __future__ import annotations

import numpy as np
import pytest
import torch

from types import SimpleNamespace

from configuration.diagnostics                        import RobustnessConfig
from pipelines.backbone.inference.analysis.robustness import RobustnessCore, RobustnessRun

from tests.backbone_inference._helpers import StageLogger, build_run_directory
from tests.conftest                    import SilentLogger


H, W = 8, 8


class _Model(torch.nn.Module):
    def forward(self, images):
        return (images[:, 1:2].abs() + images[:, 2:3].abs() + 0.5)


class _PositionalModel(_Model):
    def forward(self, images):
        return super().forward(images) + torch.arange(H, dtype=torch.float32)[None, None, :, None]


class _CountingModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return super().forward(images)


def _batches():
    rng    = np.random.default_rng(0)
    images = torch.from_numpy(rng.uniform(0.4, 1.0, size=(2, 3, H, W))).float()

    return [(images, _Model()(images), torch.ones(2, 1, H, W))]


def test_noise_curve_starts_clean_and_degrades():
    batches = _batches()
    core    = RobustnessCore(_Model(), lambda: None)

    rows = core.noise_curve(batches, [0.0, 0.5, 2.0])

    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"] > 0.0
    assert rows[2]["mse"] > rows[1]["mse"]


def test_noise_curve_orders_an_out_of_order_sigma_list():
    batches = _batches()
    core    = RobustnessCore(_Model(), lambda: None)

    rows = core.noise_curve(batches, [0.5, 0.0, 2.0])

    assert [row["sigma"] for row in rows] == [0.0, 0.5, 2.0]
    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)


def test_noise_on_unused_channels_is_harmless():
    batches = _batches()
    core    = RobustnessCore(_Model(), lambda: None)

    unused = core.noise_curve(batches, [0.0, 1.0], channels=[0])
    used   = core.noise_curve(batches, [0.0, 1.0], channels=[1, 2])

    assert unused[1]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert used[1]["mse"]   > 0.0


def test_noise_draws_report_spread():
    batches = _batches()
    core    = RobustnessCore(_Model(), lambda: None)

    rows = core.noise_curve(batches, [0.0, 0.5], draws=3)

    assert rows[0]["mse_std"] is None
    assert rows[1]["mse_std"] is not None


def test_gain_curve_always_holds_the_reference():
    batches = _batches()
    core    = RobustnessCore(_Model(), lambda: None)

    rows  = core.gain_curve(batches, [0.5, 2.0])
    gains = [row["gain"] for row in rows]

    assert gains == [0.5, 1.0, 2.0]
    assert rows[1]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["mse"] > 0.0
    assert rows[2]["mse"] > 0.0


def test_drop_curve_degrades_with_more_channels():
    batches = _batches()
    core    = RobustnessCore(_Model(), lambda: None)

    rows = core.drop_curve(batches, 3, draws=3)

    assert rows[0]["dropped"] == 0
    assert rows[0]["mse"]     == pytest.approx(0.0, abs=1e-12)
    assert rows[2]["mse"]     > 0.0
    assert rows[3]["mse"]     > rows[2]["mse"]
    assert rows[3]["dropped"] == 3


def test_drop_curve_evaluates_each_distinct_subset_once():
    batches = _batches()
    model   = _CountingModel()
    rows    = RobustnessCore(model, lambda: None).drop_curve(batches, 2, draws=8)

    assert model.calls == 4

    assert rows[1]["mse_std"] is not None
    assert rows[2]["mse_std"] is None


def test_shift_consistency_is_zero_for_a_pointwise_model():
    batches = _batches()
    core    = RobustnessCore(_Model(), lambda: None)

    rows = core.shift_consistency(batches, [1, 3])

    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"] == pytest.approx(0.0, abs=1e-12)


def test_shift_consistency_detects_position_dependence():
    rows = RobustnessCore(_PositionalModel(), lambda: None).shift_consistency(_batches(), [2])

    assert rows[0]["mse"] > 1e-6


def test_tolerance_sigma_interpolates_and_censors():
    crossing = [
        {"sigma": 0.0, "mse": 1.0, "mse_std": None},
        {"sigma": 1.0, "mse": 1.5, "mse_std": None},
        {"sigma": 2.0, "mse": 3.5, "mse_std": None},
    ]
    flat = [
        {"sigma": 0.0, "mse": 1.0, "mse_std": None},
        {"sigma": 1.0, "mse": 1.2, "mse_std": None},
    ]

    tolerance = RobustnessCore.tolerance_sigma(crossing)

    assert tolerance["sigma"]    == pytest.approx(1.25)
    assert tolerance["censored"] is False

    censored = RobustnessCore.tolerance_sigma(flat)

    assert censored["sigma"]    == pytest.approx(1.0)
    assert censored["censored"] is True


def test_planned_sweep_counts_match_the_sweeps_run():
    batches = _batches()
    sweeps  = []
    core    = RobustnessCore(_Model(), lambda: sweeps.append(1))

    core.noise_curve(batches, [0.0, 0.5, 2.0], draws=3)
    assert len(sweeps) == RobustnessCore.noise_sweeps([0.0, 0.5, 2.0], 3) == 7

    core.gain_curve(batches, [0.5, 2.0])
    assert len(sweeps) == 7 + RobustnessCore.gain_sweeps([0.5, 2.0]) == 10

    core.drop_curve(batches, 3, draws=2)
    assert len(sweeps) == 10 + RobustnessCore.drop_sweeps(3, 2) == 16

    core.shift_consistency(batches, [1, 2])
    assert len(sweeps) == 18


@pytest.mark.dataset
@pytest.mark.slow
def test_robustness_pass_plans_tracks_and_times_every_stress(tmp_path, test_data_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    config  = RobustnessConfig(batch_size=2, max_batches=1, noise_sigmas=[0.0, 0.5], noise_draws=1, gain_factors=[0.5], shift_pixels=[1], draws_per_count=1)
    logger  = StageLogger()

    outcome = RobustnessRun(run_dir, config, logger).run()

    planned = [message for level, message in logger.messages if level == "info" and "sweeps over" in message]

    assert len(planned) == 1
    assert outcome.payload["n_batches"] == 1
    assert logger.totals == [logger.advances]
    assert [label.split(" (")[0] for label in logger.timers] == ["Robustness: noise", "Robustness: gain", "Robustness: dropout", "Robustness: shift"]


def test_report_labels_the_gain_row_with_the_swept_extremes(tmp_path):
    run     = RobustnessRun(tmp_path, RobustnessConfig(gain_factors=[0.8, 1.25]), SilentLogger())
    results = {
        "noise"           : {"all channels": [{"sigma": 0.0, "mse": 1.0, "mse_std": None}]},
        "noise_tolerance" : {"sigma": 0.0, "censored": True},
        "gain"            : [{"gain": 0.8, "mse": 1.5}, {"gain": 1.0, "mse": 1.0}, {"gain": 1.25, "mse": 1.7}],
        "drop"            : [],
        "shift"           : [],
    }
    run.output_dir.mkdir(parents=True)

    report = run._write_report(SimpleNamespace(backbone_name="unet"), results, {}, 1).read_text(encoding="utf-8")

    assert "Gain 0.8 / 1.25 MSE" in report and "1.5 / 1.7" in report
    assert "Gain 0.5 / 2.0" not in report
