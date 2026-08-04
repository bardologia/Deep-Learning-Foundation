from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from pathlib     import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.patch_sweep.planner import PatchSweepPlanner, SweepUnit
from tools.data.io                 import FileIO
from tools.metrics.scoring         import FiniteScalar, SeedAggregation
from tools.monitoring.logger       import Logger
from tools.reporting.markdown      import MarkdownDoc, MarkdownTable, ScalarFormatter
from tools.reporting.plotting      import PlotBase


@dataclass
class SweepRecord:
    unit              : SweepUnit
    status            : str
    duration_s        : float | None
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


class SweepCollector:

    SEED_DIR_PATTERN = re.compile(r"seed\d+")

    def __init__(self, run_dir: Path, planner: PatchSweepPlanner, logger: Logger) -> None:
        self.run_dir      = run_dir
        self.training_dir = run_dir / "training"
        self.planner      = planner
        self.logger       = logger

    def _training_results(self) -> dict:
        path = self.run_dir / "pipeline" / "training_results.json"
        if not path.exists():
            raise FileNotFoundError(f"The sweep report needs {path}; run the training stage first")

        return {result["name"]: result for result in FileIO.load_json(path)}

    def _test_loss(self, unit_dir: Path) -> float | None:
        path = unit_dir / "meta" / "test_metrics.json"
        if not path.exists():
            return None

        return float(FileIO.load_json(path)["avg_loss"])

    def _checkpoint_fields(self, unit_dir: Path) -> tuple[float | None, int | None]:
        path = next(unit_dir.rglob("best_model.pt"), None)
        if path is None:
            return None, None

        checkpoint    = torch.load(path, map_location="cpu", weights_only=False)
        best_val_loss = checkpoint.get("best_val_loss")
        best_epoch    = checkpoint.get("best_epoch")

        del checkpoint
        gc.collect()

        return (float(best_val_loss) if best_val_loss is not None else None,
                int(best_epoch)      if best_epoch    is not None else None)

    def _seed_runs(self, unit_dir: Path) -> list[tuple[str | None, Path]]:
        if unit_dir.is_dir():
            seed_dirs = sorted(d for d in unit_dir.iterdir() if d.is_dir() and self.SEED_DIR_PATTERN.fullmatch(d.name))
            if seed_dirs:
                return [(seed_dir.name, seed_dir) for seed_dir in seed_dirs]

        return [(None, unit_dir)]

    def _collect_seed_run(self, unit: SweepUnit, seed_name: str | None, run_dir: Path, results: dict) -> dict:
        name   = unit.name if seed_name is None else f"{unit.name}/{seed_name}"
        result = results.get(name, {})

        test_loss                 = self._test_loss(run_dir)
        best_val_loss, best_epoch = self._checkpoint_fields(run_dir)

        return {
            "name"          : name,
            "status"        : result.get("status", "MISSING"),
            "duration_s"    : result.get("duration_s"),
            "test_loss"     : test_loss,
            "best_val_loss" : best_val_loss,
            "best_epoch"    : best_epoch,
        }

    @staticmethod
    def _status(runs: list[dict]) -> str:
        statuses = [run["status"] for run in runs]

        if all(status == "DONE" for status in statuses):
            return "DONE"
        if any(status == "DONE" for status in statuses):
            return "PARTIAL"
        return statuses[0] if len(set(statuses)) == 1 else "MISSING"

    @staticmethod
    def _contributing(runs: list[dict], key: str) -> int:
        return sum(1 for run in runs if FiniteScalar.coerce(run.get(key)) is not None)

    def _aggregate(self, unit: SweepUnit, runs: list[dict]) -> SweepRecord:
        means, stds = SeedAggregation.aggregate(runs, ["test_loss", "best_val_loss", "best_epoch", "duration_s"])

        return SweepRecord(
            unit              = unit,
            status            = self._status(runs),
            duration_s        = means.get("duration_s"),
            test_loss         = means.get("test_loss"),
            best_val_loss     = means.get("best_val_loss"),
            best_epoch        = means.get("best_epoch"),
            n_seeds           = self._contributing(runs, "test_loss"),
            n_seed_runs       = len(runs),
            test_loss_std     = stds.get("test_loss"),
            best_val_loss_std = stds.get("best_val_loss"),
            seed_runs         = runs,
        )

    def collect(self) -> list[SweepRecord]:
        results = self._training_results()

        records = []
        for unit in self.planner.units():
            unit_dir = self.training_dir / unit.name
            runs     = [self._collect_seed_run(unit, seed_name, run_dir, results) for seed_name, run_dir in self._seed_runs(unit_dir)]
            record   = self._aggregate(unit, runs)

            if record.test_loss is None:
                self.logger.warning(f"{unit.name}: no test metrics, excluded from the ranking")

            records.append(record)

        return records


class SweepPlots(PlotBase):
    save_dpi    = 300
    AXIS_LABELS = ("azimuth", "range")

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

        self._apply_style()

    @staticmethod
    def _axis_values(records: list[SweepRecord], axis: int) -> list[int]:
        return sorted({record.unit.patch_size[axis] for record in records})

    def _band(self, ax, records: list[SweepRecord], axis: int, value_key: str, std_key: str, color: str) -> None:
        with_std = [record for record in records if getattr(record, std_key) is not None]
        if len(with_std) < 2:
            return

        sizes  = np.array([record.unit.patch_size[axis] for record in with_std], dtype=np.float64)
        means  = np.array([getattr(record, value_key) for record in with_std], dtype=np.float64)
        stds   = np.array([getattr(record, std_key)   for record in with_std], dtype=np.float64)

        ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)

    def curve(self, dataset: str, records: list[SweepRecord], axis: int) -> Path:
        complete = [record for record in records if record.complete]

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        ax.plot([r.unit.patch_size[axis] for r in complete], [r.test_loss for r in complete], marker="o", color="#B03052", label="test loss")
        self._band(ax, complete, axis, "test_loss", "test_loss_std", "#B03052")

        with_val = [record for record in records if record.best_val_loss is not None]
        ax.plot([r.unit.patch_size[axis] for r in with_val], [r.best_val_loss for r in with_val], marker="s", color="#787880", label="best validation loss")
        self._band(ax, with_val, axis, "best_val_loss", "best_val_loss_std", "#787880")

        ax.set_xlabel(f"{self.AXIS_LABELS[axis]} patch length (px)")
        ax.set_ylabel("held-out loss")
        ax.set_title(dataset)
        ax.legend(frameon=False)

        return self._save(fig, self.out_dir / "curves" / f"{dataset}.png")

    def heatmap(self, dataset: str, records: list[SweepRecord], azimuths: list[int], ranges: list[int]) -> Path:
        grid = np.full((len(azimuths), len(ranges)), np.nan, dtype=np.float64)
        for record in records:
            if record.complete:
                grid[azimuths.index(record.unit.patch_size[0]), ranges.index(record.unit.patch_size[1])] = record.test_loss

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax, label="seed-mean test loss")

        if np.isfinite(grid).any():
            best_v, best_h = np.unravel_index(np.nanargmin(grid), grid.shape)
            ax.scatter([best_h], [best_v], marker="*", s=130, color="#B03052", edgecolors="white", linewidths=0.6)

        ax.set_xticks(range(len(ranges)))
        ax.set_xticklabels(ranges)
        ax.set_yticks(range(len(azimuths)))
        ax.set_yticklabels(azimuths)
        ax.set_xlabel("range patch length (px)")
        ax.set_ylabel("azimuth patch length (px)")
        ax.set_title(dataset)

        return self._save(fig, self.out_dir / "heatmaps" / f"{dataset}.png")

    def dataset_figure(self, dataset: str, records: list[SweepRecord]) -> Path:
        azimuths = self._axis_values(records, 0)
        ranges   = self._axis_values(records, 1)

        if len(azimuths) > 1 and len(ranges) > 1:
            return self.heatmap(dataset, records, azimuths, ranges)

        return self.curve(dataset, records, axis=0 if len(azimuths) > 1 else 1)

    def best_versus_datasets(self, best: dict[str, tuple[int, int]]) -> Path:
        names = list(best)

        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        ax.plot(range(len(names)), [best[name][0] for name in names], marker="o", linestyle="none", color="#B03052", label="azimuth")
        ax.plot(range(len(names)), [best[name][1] for name in names], marker="s", linestyle="none", color="#787880", label="range")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        ax.set_xlabel("dataset")
        ax.set_ylabel("best patch length (px)")
        ax.set_title("Best patch lengths per dataset")
        ax.legend(frameon=False)

        return self._save(fig, self.out_dir / "best_patch_vs_dataset.png")


class PatchSweepReport:
    def __init__(self, records: list[SweepRecord], planner: PatchSweepPlanner, out_dir: Path, logger: Logger) -> None:
        self.records = records
        self.planner = planner
        self.out_dir = out_dir
        self.logger  = logger
        self.plots   = SweepPlots(out_dir)

    def _by_dataset(self) -> dict[str, list[SweepRecord]]:
        groups: dict[str, list[SweepRecord]] = {}
        for record in self.records:
            groups.setdefault(record.unit.dataset, []).append(record)

        return {dataset: sorted(group, key=lambda r: r.unit.patch_size) for dataset, group in groups.items()}

    def _best(self, group: list[SweepRecord]) -> SweepRecord | None:
        complete = [record for record in group if record.complete]
        if not complete:
            return None

        return min(complete, key=lambda record: record.test_loss)

    def _summary_payload(self, groups: dict[str, list[SweepRecord]]) -> dict:
        per_dataset = {}
        for dataset, group in groups.items():
            best = self._best(group)

            per_dataset[dataset] = {
                "dataset_path"    : str(group[0].unit.dataset_path),
                "best_patch_size" : list(best.unit.patch_size) if best else None,
                "best_test_loss"  : best.test_loss             if best else None,
                "units"           : [{
                    "name"              : record.unit.name,
                    "patch_size"        : list(record.unit.patch_size),
                    "patch_stride"      : list(record.unit.patch_stride),
                    "batch_size"        : record.unit.batch_size,
                    "status"            : record.status,
                    "n_seeds"           : record.n_seeds,
                    "n_seed_runs"       : record.n_seed_runs,
                    "test_loss"         : record.test_loss,
                    "test_loss_std"     : record.test_loss_std,
                    "best_val_loss"     : record.best_val_loss,
                    "best_val_loss_std" : record.best_val_loss_std,
                    "best_epoch"        : record.best_epoch,
                    "seed_runs"         : record.seed_runs,
                } for record in group],
            }

        azimuth_sizes, range_sizes = self.planner.patch_sizes()

        return {
            "patch_step"    : self.planner.patch_step(),
            "azimuth_sizes" : azimuth_sizes,
            "range_sizes"   : range_sizes,
            "datasets"      : per_dataset,
        }

    def _write_json(self, groups: dict[str, list[SweepRecord]]) -> Path:
        path = self.out_dir / "patch_sweep.json"
        FileIO.save_json(self._summary_payload(groups), path, indent=2)

        return path

    def _write_dataset_figures(self, groups: dict[str, list[SweepRecord]]) -> list[Path]:
        return [self.plots.dataset_figure(dataset, group) for dataset, group in groups.items()]

    def _write_best_plot(self, groups: dict[str, list[SweepRecord]]) -> Path | None:
        best = {dataset: self._best(group) for dataset, group in groups.items()}
        best = {dataset: record.unit.patch_size for dataset, record in best.items() if record is not None}

        if len(best) < 2:
            self.logger.warning("Fewer than two datasets have a ranked best patch; the best-versus-datasets figure is skipped")
            return None

        return self.plots.best_versus_datasets(best)

    @staticmethod
    def _patch_label(patch_size: tuple[int, int]) -> str:
        return f"{patch_size[0]}x{patch_size[1]}"

    def _summary_table(self, groups: dict[str, list[SweepRecord]]) -> MarkdownTable:
        table = MarkdownTable(["Dataset", "Best patch (az x rg, px)", "Test loss at best"], align=["left", "right", "right"])

        for dataset, group in groups.items():
            best = self._best(group)

            if best is None:
                table.add_row(dataset, None, None)
                continue

            table.add_row(
                dataset,
                self._patch_label(best.unit.patch_size),
                self._with_std(best.test_loss, best.test_loss_std),
            )

        return table

    @staticmethod
    def _with_std(value: float | None, std: float | None) -> str | None:
        if value is None:
            return None

        rendered = ScalarFormatter.format_scalar(value)
        return rendered if std is None else f"{rendered} ± {ScalarFormatter.format_scalar(std)}"

    def _group_table(self, group: list[SweepRecord], best: SweepRecord | None) -> MarkdownTable:
        table = MarkdownTable(["Patch (az x rg, px)", "Stride", "Batch", "Seeds scored", "Test loss", "Best val loss", "Best epoch", "Status"], align=["right"] * 8)

        for record in group:
            label = self._patch_label(record.unit.patch_size)
            patch = f"**{label}**" if best is not None and record is best else label
            table.add_row(
                patch,
                self._patch_label(record.unit.patch_stride),
                record.unit.batch_size,
                record.n_seeds,
                self._with_std(record.test_loss, record.test_loss_std),
                self._with_std(record.best_val_loss, record.best_val_loss_std),
                ScalarFormatter.format_scalar(record.best_epoch),
                record.status,
            )

        return table

    def _write_markdown(self, groups: dict[str, list[SweepRecord]], figure_paths: list[Path], best_plot: Path | None) -> Path:
        config                     = self.planner.config
        azimuth_sizes, range_sizes = self.planner.patch_sizes()
        doc                        = MarkdownDoc("Patch-size sweep")

        doc.paragraph("Best patch lengths per dataset: the same backbone is trained across every admissible combination of azimuth and range patch lengths on each selected dataset, all on the traditional reduced stack. The datasets differ by the boxcar window used in preprocessing, so the sweep shows how the best receptive field follows the filter footprint along each axis.")
        doc.kv_table({
            "Backbone"       : f"{config.backbone_name}-{config.backbone_head}",
            "Datasets"       : ", ".join(groups),
            "Secondaries"    : ", ".join(config.paths.secondary_labels),
            "Patch step"     : self.planner.patch_step(),
            "Azimuth sizes"  : azimuth_sizes,
            "Range sizes"    : range_sizes,
            "Seeds"          : config.seeds or [config.seed],
            "Ranking metric" : "seed-mean test avg_loss at the restored best-validation checkpoint",
        })

        doc.heading("Best patch lengths per dataset", level=2)
        doc.table(self._summary_table(groups))

        if best_plot is not None:
            doc.image("best patch lengths per dataset", best_plot.relative_to(self.out_dir))

        for (dataset, group), figure_path in zip(groups.items(), figure_paths):
            best = self._best(group)

            doc.heading(f"Dataset {dataset}", level=2)
            doc.bold_kv("Path", str(group[0].unit.dataset_path))
            doc.blank()
            doc.table(self._group_table(group, best))
            doc.image(f"loss versus patch lengths on {dataset}", figure_path.relative_to(self.out_dir))

        path = self.out_dir / "report.md"
        doc.save(path)

        return path

    def write_all(self) -> list[Path]:
        groups = self._by_dataset()

        json_path    = self._write_json(groups)
        figure_paths = self._write_dataset_figures(groups)
        best_plot    = self._write_best_plot(groups)
        md_path      = self._write_markdown(groups, figure_paths, best_plot)

        written = [json_path, *figure_paths, md_path]
        if best_plot is not None:
            written.append(best_plot)

        return written
