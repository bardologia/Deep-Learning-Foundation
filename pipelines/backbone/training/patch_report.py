
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipelines.backbone.training.experiments import PatchSizeTrialPlanner
from pipelines.shared.training.seed_sweep    import SeedSet
from pipelines.shared.training.sweep_resume  import SweepResumeRoot
from tools.data.io                           import FileIO
from tools.metrics.scoring                   import FiniteScalar, SeedAggregation
from tools.monitoring.logger                 import Logger
from tools.reporting.markdown                import MarkdownDoc, MarkdownTable, ScalarFormatter
from tools.reporting.plotting                import PlotBase
from tools.runtime.completion                import CompletionMarker


@dataclass
class PatchTrialRecord:

    name              : str
    patch_size        : tuple[int, int]
    patch_stride      : tuple[int, int]
    batch_size        : int | None
    status            : str
    test_loss         : float | None
    best_val_loss     : float | None
    best_epoch        : float | None
    n_seeds           : int          = 1
    n_seed_runs       : int          = 1
    test_loss_std     : float | None = None
    best_val_loss_std : float | None = None
    seed_runs         : list[dict]   = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.test_loss is not None


class PatchTrialCollector:

    def __init__(self, runs_root: Path, planner: PatchSizeTrialPlanner, logger: Logger, seeds: list[int]) -> None:
        self.runs_root = Path(runs_root)
        self.planner   = planner
        self.logger    = logger
        self.seeds     = list(seeds)

    def _unit_names(self, name: str) -> list[str]:
        if len(self.seeds) == 1:
            return [name]

        return [SeedSet.run_name(name, seed) for seed in self.seeds]

    def _unit_directory(self, unit_name: str) -> Path | None:
        hits = SweepResumeRoot.matches(self.runs_root, unit_name)

        return hits[0] if hits else None

    def _test_loss(self, run_dir: Path) -> float | None:
        path = run_dir / "meta" / "test_metrics.json"
        if not path.exists():
            return None

        return float(FileIO.load_json(path)["avg_loss"])

    def _checkpoint_fields(self, run_dir: Path) -> tuple[float, int]:
        payload = CompletionMarker.payload(run_dir)

        return float(payload["best_val_loss"]), int(payload["best_epoch"])

    def _collect_seed_run(self, unit_name: str, run_dir: Path | None) -> dict:
        complete                  = run_dir is not None and CompletionMarker.is_complete(run_dir)
        test_loss                 = self._test_loss(run_dir) if run_dir is not None else None
        best_val_loss, best_epoch = self._checkpoint_fields(run_dir) if complete else (None, None)

        return {
            "name"          : unit_name,
            "status"        : "DONE" if complete else "MISSING",
            "test_loss"     : test_loss,
            "best_val_loss" : best_val_loss,
            "best_epoch"    : best_epoch,
        }

    @staticmethod
    def _exclusion(run: dict) -> str | None:
        if run["status"] != "DONE":
            return "no completion marker (run directory missing or run unfinished)"
        if run["test_loss"] is None:
            return "no test metrics"
        if FiniteScalar.coerce(run["test_loss"]) is None:
            return f"test loss {run['test_loss']} is not finite"

        return None

    @classmethod
    def _status(cls, runs: list[dict]) -> str:
        scored   = [run for run in runs if cls._exclusion(run) is None]
        finished = [run for run in runs if run["status"] == "DONE"]

        if len(scored) == len(runs):
            return "DONE"
        if finished:
            return "PARTIAL"
        return "MISSING"

    def _aggregate(self, name: str, overrides: dict, runs: list[dict]) -> PatchTrialRecord:
        means, stds = SeedAggregation.aggregate(runs, ["test_loss", "best_val_loss", "best_epoch"])

        return PatchTrialRecord(
            name              = name,
            patch_size        = tuple(overrides["training.patch_size"]),
            patch_stride      = tuple(overrides["training.patch_stride"]),
            batch_size        = overrides.get("training.batch_size"),
            status            = self._status(runs),
            test_loss         = means.get("test_loss"),
            best_val_loss     = means.get("best_val_loss"),
            best_epoch        = means.get("best_epoch"),
            n_seeds           = sum(1 for run in runs if self._exclusion(run) is None),
            n_seed_runs       = len(runs),
            test_loss_std     = stds.get("test_loss"),
            best_val_loss_std = stds.get("best_val_loss"),
            seed_runs         = runs,
        )

    def collect(self) -> list[PatchTrialRecord]:
        records = []
        for name, overrides in self.planner.plan():
            runs   = [self._collect_seed_run(unit_name, self._unit_directory(unit_name)) for unit_name in self._unit_names(name)]
            record = self._aggregate(name, overrides, runs)

            for run in runs:
                reason = self._exclusion(run)
                if reason is not None:
                    self.logger.warning(f"{run['name']}: {reason}, excluded from the trial mean")

            if record.test_loss is None:
                self.logger.warning(f"{name}: none of its {record.n_seed_runs} planned seed run(s) contributed a finite test loss, excluded from the ranking")

            records.append(record)

        return records


class PatchTrialPlots(PlotBase):

    save_dpi    = 300
    AXIS_LABELS = ("row", "range")

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

        self._apply_style()

    @staticmethod
    def _axis_values(records: list[PatchTrialRecord], axis: int) -> list[int]:
        return sorted({record.patch_size[axis] for record in records})

    def _band(self, ax, records: list[PatchTrialRecord], axis: int, value_key: str, std_key: str, color: str) -> None:
        with_std = [record for record in records if getattr(record, std_key) is not None]
        if len(with_std) < 2:
            return

        sizes = np.array([record.patch_size[axis] for record in with_std], dtype=np.float64)
        means = np.array([getattr(record, value_key) for record in with_std], dtype=np.float64)
        stds  = np.array([getattr(record, std_key)   for record in with_std], dtype=np.float64)

        ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)

    def curve(self, title: str, records: list[PatchTrialRecord], axis: int) -> Path:
        complete = [record for record in records if record.complete]

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        ax.plot([r.patch_size[axis] for r in complete], [r.test_loss for r in complete], marker="o", color="#B03052", label="test loss")
        self._band(ax, complete, axis, "test_loss", "test_loss_std", "#B03052")

        with_val = [record for record in records if record.best_val_loss is not None]
        ax.plot([r.patch_size[axis] for r in with_val], [r.best_val_loss for r in with_val], marker="s", color="#787880", label="best validation loss")
        self._band(ax, with_val, axis, "best_val_loss", "best_val_loss_std", "#787880")

        ax.set_xlabel(f"{self.AXIS_LABELS[axis]} patch length (px)")
        ax.set_ylabel("held-out loss")
        ax.set_title(title)
        ax.legend(frameon=False)

        return self._save(fig, self.out_dir / "curve.png")

    def heatmap(self, title: str, records: list[PatchTrialRecord], rows: list[int], ranges: list[int]) -> Path:
        grid = np.full((len(rows), len(ranges)), np.nan, dtype=np.float64)
        for record in records:
            if record.complete:
                grid[rows.index(record.patch_size[0]), ranges.index(record.patch_size[1])] = record.test_loss

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax, label="seed-mean test loss")

        if np.isfinite(grid).any():
            best_v, best_h = np.unravel_index(np.nanargmin(grid), grid.shape)
            ax.scatter([best_h], [best_v], marker="*", s=130, color="#B03052", edgecolors="white", linewidths=0.6)

        ax.set_xticks(range(len(ranges)))
        ax.set_xticklabels(ranges)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_xlabel("range patch length (px)")
        ax.set_ylabel("row patch length (px)")
        ax.set_title(title)

        return self._save(fig, self.out_dir / "heatmap.png")

    def grid_figure(self, title: str, records: list[PatchTrialRecord]) -> Path:
        rows = self._axis_values(records, 0)
        ranges   = self._axis_values(records, 1)

        if len(rows) > 1 and len(ranges) > 1:
            return self.heatmap(title, records, rows, ranges)

        return self.curve(title, records, axis=0 if len(rows) > 1 else 1)


class PatchTrialsReport:

    def __init__(self, config, planner: PatchSizeTrialPlanner, model_label: str, runs_root: Path, out_dir: Path, logger: Logger) -> None:
        self.config      = config
        self.planner     = planner
        self.model_label = model_label
        self.runs_root   = Path(runs_root)
        self.out_dir     = Path(out_dir)
        self.logger      = logger
        self.seeds       = SeedSet.resolve(config.seeds, config.seed)
        self.plots       = PatchTrialPlots(self.out_dir)

    def _sorted_records(self, records: list[PatchTrialRecord]) -> list[PatchTrialRecord]:
        return sorted(records, key=lambda record: record.patch_size)

    def _best(self, records: list[PatchTrialRecord]) -> PatchTrialRecord | None:
        complete = [record for record in records if record.complete]
        if not complete:
            return None

        return min(complete, key=lambda record: record.test_loss)

    def _summary_payload(self, records: list[PatchTrialRecord]) -> dict:
        best = self._best(records)

        row_sizes, col_sizes = self.planner.patch_sizes()

        return {
            "patch_step"      : self.planner.patch_step(),
            "row_sizes"   : row_sizes,
            "col_sizes"     : col_sizes,
            "dataset"         : Path(self.config.paths.dataset_path).name,
            "dataset_path"    : str(self.config.paths.dataset_path),
            "best_patch_size" : list(best.patch_size) if best else None,
            "best_test_loss"  : best.test_loss       if best else None,
            "trials"          : [{
                "name"              : record.name,
                "patch_size"        : list(record.patch_size),
                "patch_stride"      : list(record.patch_stride),
                "batch_size"        : record.batch_size,
                "status"            : record.status,
                "n_seeds"           : record.n_seeds,
                "n_seed_runs"       : record.n_seed_runs,
                "test_loss"         : record.test_loss,
                "test_loss_std"     : record.test_loss_std,
                "best_val_loss"     : record.best_val_loss,
                "best_val_loss_std" : record.best_val_loss_std,
                "best_epoch"        : record.best_epoch,
                "seed_runs"         : record.seed_runs,
            } for record in records],
        }

    def _write_json(self, records: list[PatchTrialRecord]) -> Path:
        path = self.out_dir / "patch_trials.json"
        FileIO.save_json(self._summary_payload(records), path, indent=2)

        return path

    def _write_figure(self, records: list[PatchTrialRecord]) -> Path:
        return self.plots.grid_figure(Path(self.config.paths.dataset_path).name, records)

    @staticmethod
    def _patch_label(patch_size: tuple[int, int]) -> str:
        return f"{patch_size[0]}x{patch_size[1]}"

    def _summary_table(self, best: PatchTrialRecord | None) -> MarkdownTable:
        table = MarkdownTable(["Best patch (row x col, px)", "Test loss at best"], align=["right", "right"])

        if best is None:
            table.add_row(None, None)
        else:
            table.add_row(self._patch_label(best.patch_size), self._with_std(best.test_loss, best.test_loss_std))

        return table

    @staticmethod
    def _with_std(value: float | None, std: float | None) -> str | None:
        if value is None:
            return None

        return ScalarFormatter.with_std(ScalarFormatter.format_scalar(value), std)

    def _trial_table(self, records: list[PatchTrialRecord], best: PatchTrialRecord | None) -> MarkdownTable:
        table = MarkdownTable(["Patch (row x col, px)", "Stride", "Batch", "Seeds scored", "Test loss", "Best val loss", "Best epoch", "Status"], align=["right"] * 8)

        for record in records:
            label = self._patch_label(record.patch_size)
            patch = f"**{label}**" if best is not None and record is best else label
            table.add_row(
                patch,
                self._patch_label(record.patch_stride),
                record.batch_size,
                record.n_seeds,
                self._with_std(record.test_loss, record.test_loss_std),
                self._with_std(record.best_val_loss, record.best_val_loss_std),
                ScalarFormatter.format_scalar(record.best_epoch),
                record.status,
            )

        return table

    def _write_markdown(self, records: list[PatchTrialRecord], figure_path: Path) -> Path:
        row_sizes, col_sizes = self.planner.patch_sizes()
        best                       = self._best(records)
        doc                        = MarkdownDoc("Patch-size sweep")

        doc.paragraph("The same backbone is trained across every admissible combination of row and column patch lengths on the selected dataset, so the sweep shows how the held-out loss follows the receptive field along each axis.")
        doc.kv_table({
            "Model"          : self.model_label,
            "Dataset"        : Path(self.config.paths.dataset_path).name,
            "Dataset path"   : str(self.config.paths.dataset_path),
            "Patch step"     : self.planner.patch_step(),
            "Row sizes"  : row_sizes,
            "Range sizes"    : col_sizes,
            "Seeds"          : self.seeds,
            "Ranking metric" : "seed-mean test avg_loss at the restored best-validation checkpoint",
        })

        doc.heading("Ranking", level=2)
        doc.table(self._summary_table(best))
        doc.table(self._trial_table(records, best))
        doc.image("loss versus patch lengths", figure_path.relative_to(self.out_dir))

        path = self.out_dir / "report.md"
        doc.save(path)

        return path

    def _log_ranking(self, records: list[PatchTrialRecord]) -> None:
        ranked = [record for record in records if record.complete]
        best   = self._best(records)

        if best is None:
            self.logger.warning(f"No patch shape ranked: none of the {len(records)} trials has a finite seed-mean test loss")
            return

        self.logger.subsection(f"Ranked {len(ranked)}/{len(records)} patch shapes; best {self._patch_label(best.patch_size)} with test loss {self._with_std(best.test_loss, best.test_loss_std)} over {best.n_seeds}/{best.n_seed_runs} seeds")

    def write_all(self) -> list[Path]:
        collector = PatchTrialCollector(runs_root=self.runs_root, planner=self.planner, logger=self.logger, seeds=self.seeds)
        records   = self._sorted_records(collector.collect())

        json_path   = self._write_json(records)
        figure_path = self._write_figure(records)
        md_path     = self._write_markdown(records, figure_path)

        self._log_ranking(records)

        return [json_path, figure_path, md_path]
