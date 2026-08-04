from __future__ import annotations

from itertools import combinations
from pathlib   import Path

import matplotlib

matplotlib.use("Agg")
import optuna
import optuna.visualization.matplotlib as ovm

from tools.reporting.plotting import PlotBase


class StudyPlotter(PlotBase):
    save_dpi = 300

    CONTOUR_MAX_PARAMS = 8

    OBJECTIVE_NAME = "best validation loss under the trial's own objective (lower is better)"

    def __init__(self, logger) -> None:
        self.logger = logger

    @staticmethod
    def _readable_param(name: str) -> str:
        return f"{name[:-5]} (choice index)" if name.endswith("__idx") else name

    @classmethod
    def _relabel_text(cls, text: str) -> str:
        if text == "Objective Value":
            return cls.OBJECTIVE_NAME
        if text.endswith("__idx"):
            return cls._readable_param(text)
        return text

    @classmethod
    def _relabel_figure(cls, fig) -> None:
        for ax in fig.axes:
            ax.set_xlabel(cls._relabel_text(ax.get_xlabel()))
            ax.set_ylabel(cls._relabel_text(ax.get_ylabel()))
            ax.set_title(cls._relabel_text(ax.get_title()))

            for setter, getter in ((ax.set_xticklabels, ax.get_xticklabels), (ax.set_yticklabels, ax.get_yticklabels)):
                labels  = getter()
                relabel = [cls._relabel_text(t.get_text()) for t in labels]
                if any(new != old.get_text() for new, old in zip(relabel, labels)):
                    setter(relabel)

    def _save_study_figure(self, fig, path: Path) -> Path:
        self._relabel_figure(fig)
        return self._save(fig, path)

    def _render_single(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        saved = []
        for name, plotter in self._plotters(study).items():
            axes = plotter()
            saved.append(self._save_study_figure(axes.figure, out_dir / f"{name}.png"))

        return saved

    def _render_per_param(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        params = self._study_params(study)
        saved  = []

        for name, plotter in (("slice", ovm.plot_slice), ("rank", ovm.plot_rank)):
            for param in params:
                axes = plotter(study, params=[param])
                saved.append(self._save_study_figure(axes.figure, out_dir / name / f"{param}.png"))

        return saved

    def _render_contours(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        params = self._contour_params(study)
        saved  = []

        for p1, p2 in combinations(params, 2):
            axes = ovm.plot_contour(study, params=[p1, p2])
            saved.append(self._save_study_figure(axes.figure, out_dir / "contour" / f"{p1}__{p2}.png"))

        return saved

    def _plotters(self, study: optuna.Study) -> dict:
        return {
            "optimization_history" : lambda: ovm.plot_optimization_history(study, target_name=self.OBJECTIVE_NAME),
            "intermediate_values"  : lambda: ovm.plot_intermediate_values(study),
            "parallel_coordinate"  : lambda: ovm.plot_parallel_coordinate(study, target_name=self.OBJECTIVE_NAME),
            "param_importances"    : lambda: ovm.plot_param_importances(study, target_name=self.OBJECTIVE_NAME),
            "duration_importances" : lambda: ovm.plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="trial duration [s]"),
            "edf"                  : lambda: ovm.plot_edf(study),
            "timeline"             : lambda: ovm.plot_timeline(study),
        }

    @staticmethod
    def _study_params(study: optuna.Study) -> list[str]:
        return sorted({p for t in study.trials for p in t.params})

    def _contour_params(self, study: optuna.Study) -> list[str]:
        params = self._study_params(study)
        if len(params) <= self.CONTOUR_MAX_PARAMS:
            return params

        importances = optuna.importance.get_param_importances(study)
        top         = list(importances)[: self.CONTOUR_MAX_PARAMS]
        self.logger.info(f"Contour plots limited to top {len(top)} of {len(params)} parameters by importance")
        return sorted(top)

    def render(self, study: optuna.Study, out_dir: Path) -> list[Path]:
        self._apply_style()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved  = []
        saved += self._render_single(study, out_dir)
        saved += self._render_per_param(study, out_dir)
        saved += self._render_contours(study, out_dir)

        return saved
