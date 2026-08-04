from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import optuna
import pytest

from pipelines.tuning.plots  import StudyPlotter
from pipelines.tuning.tuners import ParamSampler


SYNTH_SPACE = {
    "encoder_lr" : {"type": "float",               "low": 1e-5, "high": 1e-2, "log": True},
    "dropout"    : {"type": "float",               "low": 0.0,  "high": 0.5},
    "activation" : {"type": "categorical",         "choices": ["relu", "gelu", "silu"]},
    "features"   : {"type": "indexed_categorical", "choices": [[32, 64], [64, 128], [48, 96]]},
}


@pytest.fixture
def completed_study():
    sampler = ParamSampler()

    def objective(trial):
        s    = sampler.sample(trial, SYNTH_SPACE)
        loss = (s["dropout"] - 0.2) ** 2 + (0.0 if s["activation"] == "gelu" else 0.3)

        for epoch in range(3):
            trial.report(loss + (3 - epoch) * 0.05, epoch)

        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=4))
    study.optimize(objective, n_trials=18)
    return study


def test_render_creates_nonempty_files(completed_study, tmp_path):
    plotter = StudyPlotter(_QuietLogger())
    saved   = plotter.render(completed_study, tmp_path / "plots")

    assert len(saved) > 0
    for path in saved:
        assert path.exists()
        assert path.stat().st_size > 0


def test_render_emits_core_single_plots(completed_study, tmp_path):
    plotter = StudyPlotter(_QuietLogger())
    plotter.render(completed_study, tmp_path / "plots")

    out = tmp_path / "plots"
    assert (out / "optimization_history.png").exists()
    assert (out / "param_importances.png").exists()
    assert (out / "edf.png").exists()


def test_render_emits_per_param_slice_and_rank(completed_study, tmp_path):
    plotter = StudyPlotter(_QuietLogger())
    plotter.render(completed_study, tmp_path / "plots")

    slice_dir = tmp_path / "plots" / "slice"
    rank_dir  = tmp_path / "plots" / "rank"

    assert slice_dir.is_dir()
    assert rank_dir.is_dir()
    assert any(slice_dir.glob("*.png"))
    assert any(rank_dir.glob("*.png"))


def test_render_emits_contour_pairs(completed_study, tmp_path):
    plotter = StudyPlotter(_QuietLogger())
    plotter.render(completed_study, tmp_path / "plots")

    contour_dir = tmp_path / "plots" / "contour"
    assert contour_dir.is_dir()
    assert any(contour_dir.glob("*__*.png"))


def test_study_params_sorted_union(completed_study):
    params = StudyPlotter._study_params(completed_study)

    assert params == sorted(params)
    assert "encoder_lr" in params
    assert "features__idx" in params


def test_readable_param_relabels_indexed():
    assert StudyPlotter._readable_param("features__idx") == "features (choice index)"
    assert StudyPlotter._readable_param("dropout")       == "dropout"


def test_relabel_text_maps_objective_value():
    assert StudyPlotter._relabel_text("Objective Value") == StudyPlotter.OBJECTIVE_NAME
    assert StudyPlotter._relabel_text("dropout")         == "dropout"


def test_contour_params_capped_by_max(tmp_path):
    sampler = ParamSampler()
    space   = {f"p{i}": {"type": "float", "low": 0.0, "high": 1.0} for i in range(12)}

    def objective(trial):
        s = sampler.sample(trial, space)
        return sum(s.values())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=15)

    plotter = StudyPlotter(_QuietLogger())
    capped  = plotter._contour_params(study)

    assert len(capped) <= StudyPlotter.CONTOUR_MAX_PARAMS


class _QuietLogger:
    def info(self, *a, **k):    pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k):   pass
