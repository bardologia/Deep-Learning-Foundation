from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.data.io            import FileIO
from tools.metrics.scoring    import RankCorrelation
from tools.reporting.plotting import PlotBase


class SeedMaps:

    PREDICTIONS = "predictions.npy"
    TARGETS     = "targets.npy"

    def __init__(self, run_dirs: list[Path], inference_dirs: list[Path], maps_subdir: str) -> None:
        self.run_dir  = Path(run_dirs[0])
        self.map_dirs = [Path(directory) / maps_subdir for directory in inference_dirs]

    @property
    def channels(self) -> list[str]:
        return list(FileIO.load_json(self.run_dir / "meta" / "run_summary.json")["target_channels"])

    def predictions(self) -> np.ndarray:
        return np.stack([np.load(directory / self.PREDICTIONS) for directory in self.map_dirs])

    def targets(self) -> np.ndarray:
        return np.load(self.map_dirs[0] / self.TARGETS)


class SeedDisagreementPlots(PlotBase):

    CMAP = "inferno"

    def map_figure(self, data: np.ndarray, title: str, colorbar_label: str, path: Path) -> Path:
        return self._imshow_figure(np.asarray(data, dtype=np.float64), x_label="col (pixel)", y_label="row (pixel)", title=title, cmap=self._cmap_with_bad(self.CMAP), colorbar_label=colorbar_label, path=path)

    def risk_coverage_curve(self, rows: list[dict], full_risk: float, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot([row["coverage"] * 100.0 for row in rows], [row["risk"] for row in rows], marker="o", color=self.OKABE_ITO[0], linewidth=1.4)
        ax.axhline(full_risk, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Coverage (% of pixels kept, most confident first)")
        ax.set_ylabel("Mean squared error of the ensemble among kept pixels")
        ax.set_title("Risk-coverage under seed-disagreement confidence")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class LabelSuspects:

    STD_QUANTILE = 0.5
    ERR_QUANTILE = 0.9

    def __init__(self, disagreement_map: np.ndarray, risk_map: np.ndarray) -> None:
        self.disagreement = np.asarray(disagreement_map, dtype=np.float64)
        self.risk         = np.asarray(risk_map, dtype=np.float64)

    def run(self) -> tuple[np.ndarray, dict]:
        both    = np.isfinite(self.disagreement) & np.isfinite(self.risk)
        std_thr = float(np.quantile(self.disagreement[both], self.STD_QUANTILE))
        err_thr = float(np.quantile(self.risk[both], self.ERR_QUANTILE))
        mask    = both & (self.disagreement <= std_thr) & (self.risk > err_thr)

        return mask.astype(np.uint8), {
            "label_suspect_n"       : float(mask.sum()),
            "label_suspect_frac"    : float(mask.mean()),
            "label_suspect_std_thr" : std_thr,
            "label_suspect_err_thr" : err_thr,
        }


class RiskCoverage:

    COVERAGES = tuple(np.round(np.linspace(0.05, 1.0, 20), 3))

    def __init__(self, disagreement_map: np.ndarray, risk_map: np.ndarray) -> None:
        disagreement = np.asarray(disagreement_map, dtype=np.float64).reshape(-1)
        risk         = np.asarray(risk_map, dtype=np.float64).reshape(-1)
        both         = np.isfinite(disagreement) & np.isfinite(risk)
        order        = np.argsort(disagreement[both], kind="stable")

        self.sorted_risk         = risk[both][order]
        self.sorted_disagreement = disagreement[both][order]

    def curve(self) -> list[dict]:
        n    = self.sorted_risk.size
        rows = []

        for coverage in self.COVERAGES:
            keep = max(1, int(round(coverage * n)))
            rows.append({"coverage": float(coverage), "n": keep, "risk": float(self.sorted_risk[:keep].mean()), "threshold": float(self.sorted_disagreement[keep - 1])})

        return rows

    def run(self) -> tuple[list[dict], dict]:
        rows = self.curve()

        return rows, {
            "aurc"                        : float(np.mean([row["risk"] for row in rows])),
            "full_coverage_risk"          : rows[-1]["risk"],
            "risk_at_half_coverage"       : next(row["risk"] for row in rows if row["coverage"] >= 0.5),
            "disagreement_error_spearman" : RankCorrelation.spearman(self.sorted_disagreement, self.sorted_risk),
        }


class SeedDisagreementMaps:

    MAP_FILENAME     = "seed_std.npy"
    SUSPECT_FILENAME = "label_suspect.npy"

    def __init__(self, maps: SeedMaps, output_dir: Path, logger) -> None:
        self.maps        = maps
        self.output_dir  = Path(output_dir)
        self.figures_dir = self.output_dir / "figures" / "seed_disagreement"
        self.logger      = logger

    def _summarize(self, std: np.ndarray) -> dict:
        summary = {}

        for index, name in enumerate(self.maps.channels):
            finite = std[index][np.isfinite(std[index])]

            summary[f"seed_std_{name}_mean"]   = float(finite.mean())
            summary[f"seed_std_{name}_median"] = float(np.median(finite))
            summary[f"seed_std_{name}_p95"]    = float(np.percentile(finite, 95.0))

        return summary

    def _render_figures(self, std: np.ndarray, suspects: np.ndarray, rows: list[dict], full_risk: float) -> dict[str, Path]:
        plots   = SeedDisagreementPlots()
        figures = {name: plots.map_figure(std[index], f"{name}: across-seed standard deviation", f"std {name}", self.figures_dir / f"{name}.png") for index, name in enumerate(self.maps.channels)}

        figures["risk_coverage"] = plots.risk_coverage_curve(rows, full_risk, self.figures_dir / "risk_coverage.png")
        figures["label_suspect"] = plots.map_figure(suspects.astype(np.float64), "Label suspects: seeds agree with each other, not with the target", "suspect", self.figures_dir / "label_suspect.png")

        return figures

    def run(self) -> dict:
        FileIO.ensure_dirs(self.figures_dir)

        predictions = self.maps.predictions()
        targets     = self.maps.targets()
        std         = predictions.std(axis=0, ddof=1)
        risk        = ((predictions.mean(axis=0) - targets) ** 2).mean(axis=0)

        rows, coverage    = RiskCoverage(std.mean(axis=0), risk).run()
        suspects, flagged = LabelSuspects(std.mean(axis=0), risk).run()

        np.save(self.output_dir / self.MAP_FILENAME, std.astype(np.float32))
        np.save(self.output_dir / self.SUSPECT_FILENAME, suspects)

        summary = {**self._summarize(std), **{f"risk_coverage_{key}": value for key, value in coverage.items()}, **flagged}
        figures = self._render_figures(std, suspects, rows, coverage["full_coverage_risk"])

        self.logger.kv_table(summary, title="Seed disagreement summary")

        return {"summary": summary, "figures": figures, "risk_coverage": rows}
