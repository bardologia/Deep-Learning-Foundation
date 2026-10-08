"""Collection, plotting and reporting of patch-size trial results.

Each trial is scored by its seed-mean test loss at the restored best-validation
checkpoint; the report ranks the patch shapes swept by the patch trials mode and
shows how the held-out loss follows the receptive field along each axis.
"""

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
    """Seed-aggregated outcome of one patch-shape trial.

    Attributes:
        name              : Base run name of the trial.
        patch_size        : Patch geometry as (rows, cols) in pixels.
        patch_stride      : Patch stride as (rows, cols) in pixels.
        batch_size        : Batch size the trial trained at, None when not overridden.
        status            : Aggregated run status: DONE, PARTIAL or MISSING.
        test_loss         : Seed-mean test loss at the restored best-validation checkpoint.
        best_val_loss     : Seed-mean best validation loss.
        best_epoch        : Seed-mean epoch index of the best validation loss.
        n_seeds           : Number of seed runs that contributed a finite test loss.
        n_seed_runs       : Number of seed runs planned for this trial.
        test_loss_std     : Across-seed standard deviation of the test loss, or None.
        best_val_loss_std : Across-seed standard deviation of the best validation loss.
        seed_runs         : Per-seed record dicts backing the aggregate.
    """

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
        """Whether the trial is ranked, i.e. it carries a test loss."""
        return self.test_loss is not None


class PatchTrialCollector:
    """Gathers per-trial training outcomes from a patch sweep's runs root.

    The planned seeds drive the collection: every planned seed run is looked up,
    whether or not its directory exists, so a seed that never started, never
    finished or diverged is named and counted instead of silently thinning the
    trial mean.

    Attributes:
        runs_root : Directory holding one run directory per trial, seed runs nested.
        planner   : Planner enumerating the swept patch shapes and their overrides.
        logger    : Logger warning about seed runs excluded from a trial mean.
        seeds     : Seeds every trial was planned at.
    """

    def __init__(self, runs_root: Path, planner: PatchSizeTrialPlanner, logger: Logger, seeds: list[int]) -> None:
        """Initializes the collector against a patch sweep's runs root.

        Args:
            runs_root : Directory the trial run directories live under.
            planner   : Planner enumerating the swept patch shapes.
            logger    : Logger for collection warnings.
            seeds     : Seeds every trial was planned at; several seeds nest one seed<N> run per seed under the trial, one seed trains in the trial directory itself.
        """
        self.runs_root = Path(runs_root)
        self.planner   = planner
        self.logger    = logger
        self.seeds     = list(seeds)

    def _unit_names(self, name: str) -> list[str]:
        """Returns the run names of the units planned for a trial, one per seed or the trial name alone."""
        if len(self.seeds) == 1:
            return [name]

        return [SeedSet.run_name(name, seed) for seed in self.seeds]

    def _unit_directory(self, unit_name: str) -> Path | None:
        """Resolves the run directory of a planned unit under the runs root.

        Args:
            unit_name : Planned unit run name, nested for a seed run.

        Returns:
            The directory, matched with or without the entry's label prefix, or
            None when the unit never created one.

        Raises:
            ValueError : If several directories match the unit, which leaves the trial ambiguous.
        """
        hits = SweepResumeRoot.matches(self.runs_root, unit_name)

        if len(hits) > 1:
            raise ValueError(f"unit '{unit_name}' matches {len(hits)} run directories under {self.runs_root}: {[str(hit) for hit in hits]}; remove or move the extra ones before writing the report")

        return hits[0] if hits else None

    def _test_loss(self, run_dir: Path) -> float | None:
        """Reads the average test loss of one run, or None when it has no test metrics.

        Args:
            run_dir : Run directory of a single seed run.

        Returns:
            The test average loss, NaN included, or None.
        """
        path = run_dir / "meta" / "test_metrics.json"
        if not path.exists():
            return None

        return float(FileIO.load_json(path)["avg_loss"])

    def _checkpoint_fields(self, run_dir: Path) -> tuple[float, int]:
        """Reads the best validation loss and epoch from a completed run's marker.

        Args:
            run_dir : Run directory of a single seed run carrying a completion marker.

        Returns:
            Tuple of the best validation loss and best epoch.
        """
        payload = CompletionMarker.payload(run_dir)

        return float(payload["best_val_loss"]), int(payload["best_epoch"])

    def _collect_seed_run(self, unit_name: str, run_dir: Path | None) -> dict:
        """Builds the record of one planned seed run from its metrics and completion marker.

        Args:
            unit_name : Run name of the unit.
            run_dir   : Directory of this seed run, or None when it was never created.

        Returns:
            Dict with the run name, status, test loss, best validation loss
            and best epoch; a run without a completion marker is MISSING.
        """
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
        """Returns why a seed run contributes no test loss to its trial mean, or None when it does.

        Args:
            run : Per-seed record dict.
        """
        if run["status"] != "DONE":
            return "no completion marker (run directory missing or run unfinished)"
        if run["test_loss"] is None:
            return "no test metrics"
        if FiniteScalar.coerce(run["test_loss"]) is None:
            return f"test loss {run['test_loss']} is not finite"

        return None

    @classmethod
    def _status(cls, runs: list[dict]) -> str:
        """Reduces the per-seed records to one trial status.

        Args:
            runs : Per-seed record dicts.

        Returns:
            "DONE" when every planned seed finished and contributed a finite test
            loss, "PARTIAL" when at least one finished but not all contributed,
            and "MISSING" when none finished.
        """
        scored   = [run for run in runs if cls._exclusion(run) is None]
        finished = [run for run in runs if run["status"] == "DONE"]

        if len(scored) == len(runs):
            return "DONE"
        if finished:
            return "PARTIAL"
        return "MISSING"

    def _aggregate(self, name: str, overrides: dict, runs: list[dict]) -> PatchTrialRecord:
        """Averages the per-seed metrics of one trial into a PatchTrialRecord.

        Args:
            name      : Base run name of the trial.
            overrides : Planner overrides carrying the trial's patch geometry.
            runs      : Per-seed record dicts.

        Returns:
            The seed-aggregated PatchTrialRecord, carrying means, standard
            deviations and the backing per-seed records.
        """
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
        """Builds one aggregated record per planned patch trial.

        Returns:
            One PatchTrialRecord per trial, in planner order. Every planned seed
            run excluded from its trial mean is warned about by name and reason;
            a trial left without a finite test loss is kept but excluded from the
            ranking.
        """
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
    """Publication-quality figures of loss against patch geometry.

    Attributes:
        out_dir : Directory receiving the figure files.
    """

    save_dpi    = 300
    AXIS_LABELS = ("row", "range")

    def __init__(self, out_dir: Path) -> None:
        """Applies the shared plot style and stores the output directory.

        Args:
            out_dir : Directory receiving the figure files.
        """
        self.out_dir = out_dir

        self._apply_style()

    @staticmethod
    def _axis_values(records: list[PatchTrialRecord], axis: int) -> list[int]:
        """Returns the sorted distinct patch lengths in pixels along one axis."""
        return sorted({record.patch_size[axis] for record in records})

    def _band(self, ax, records: list[PatchTrialRecord], axis: int, value_key: str, std_key: str, color: str) -> None:
        """Shades the across-seed one-standard-deviation band around a loss curve.

        Args:
            ax        : Matplotlib axes to draw on.
            records   : Records backing the curve.
            axis      : 0 for rows, 1 for cols; selects the patch length used as abscissa.
            value_key : PatchTrialRecord attribute holding the mean.
            std_key   : PatchTrialRecord attribute holding the standard deviation.
            color     : Fill colour of the band.
        """
        with_std = [record for record in records if getattr(record, std_key) is not None]
        if len(with_std) < 2:
            return

        sizes = np.array([record.patch_size[axis] for record in with_std], dtype=np.float64)
        means = np.array([getattr(record, value_key) for record in with_std], dtype=np.float64)
        stds  = np.array([getattr(record, std_key)   for record in with_std], dtype=np.float64)

        ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)

    def curve(self, title: str, records: list[PatchTrialRecord], axis: int) -> Path:
        """Plots test and validation loss against patch length along one axis.

        Args:
            title   : Figure title.
            records : Records of the sweep.
            axis    : 0 for rows, 1 for cols.

        Returns:
            Path of the written figure.
        """
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
        """Plots the seed-mean test loss over the row-by-range patch grid.

        Args:
            title    : Figure title.
            records  : Records of the sweep.
            rows : Row patch lengths in pixels, ordered as the grid rows.
            ranges   : Range patch lengths in pixels, ordered as the grid columns.

        Returns:
            Path of the written figure, with the lowest-loss cell starred.
        """
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
        """Draws the heatmap for a two-dimensional grid, otherwise the one-axis curve.

        Args:
            title   : Figure title.
            records : Records of the sweep.

        Returns:
            Path of the written figure.
        """
        rows = self._axis_values(records, 0)
        ranges   = self._axis_values(records, 1)

        if len(rows) > 1 and len(ranges) > 1:
            return self.heatmap(title, records, rows, ranges)

        return self.curve(title, records, axis=0 if len(rows) > 1 else 1)


class PatchTrialsReport:
    """Writes the patch sweep's JSON summary, loss figure and markdown report.

    Attributes:
        config      : Entry config the sweep ran under, supplying paths and seeds.
        planner     : Planner supplying the grid description and trial overrides.
        model_label : Model identity printed in the report banner.
        runs_root   : Directory holding the trial run directories.
        out_dir     : Directory receiving the report artifacts.
        logger      : Logger for report warnings and the ranking summary.
        seeds       : Seeds every trial was planned at.
        plots       : Figure writer bound to out_dir.
    """

    def __init__(self, config, planner: PatchSizeTrialPlanner, model_label: str, runs_root: Path, out_dir: Path, logger: Logger) -> None:
        """Initializes the report writer.

        Args:
            config      : Entry config the sweep ran under, supplying paths and seeds.
            planner     : Planner supplying the grid description and trial overrides.
            model_label : Model identity printed in the report banner.
            runs_root   : Directory holding the trial run directories.
            out_dir     : Directory receiving the report artifacts.
            logger      : Logger for report warnings and the ranking summary.
        """
        self.config      = config
        self.planner     = planner
        self.model_label = model_label
        self.runs_root   = Path(runs_root)
        self.out_dir     = Path(out_dir)
        self.logger      = logger
        self.seeds       = SeedSet.resolve(config.seeds, config.seed)
        self.plots       = PatchTrialPlots(self.out_dir)

    def _sorted_records(self, records: list[PatchTrialRecord]) -> list[PatchTrialRecord]:
        """Returns the records sorted by patch size."""
        return sorted(records, key=lambda record: record.patch_size)

    def _best(self, records: list[PatchTrialRecord]) -> PatchTrialRecord | None:
        """Returns the record with the lowest test loss, or None if none ranked."""
        complete = [record for record in records if record.complete]
        if not complete:
            return None

        return min(complete, key=lambda record: record.test_loss)

    def _summary_payload(self, records: list[PatchTrialRecord]) -> dict:
        """Builds the machine-readable sweep summary.

        Args:
            records : Records sorted by patch size.

        Returns:
            Dict with the patch step, the admissible axis sizes, the dataset, the
            best patch size, best test loss and the full per-trial record list.
        """
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
        """Writes the summary payload to patch_trials.json and returns its path."""
        path = self.out_dir / "patch_trials.json"
        FileIO.save_json(self._summary_payload(records), path, indent=2)

        return path

    def _write_figure(self, records: list[PatchTrialRecord]) -> Path:
        """Writes the loss figure over the swept patch grid and returns its path."""
        return self.plots.grid_figure(Path(self.config.paths.dataset_path).name, records)

    @staticmethod
    def _patch_label(patch_size: tuple[int, int]) -> str:
        """Formats an (rows, cols) pixel pair as an 'AxR' label."""
        return f"{patch_size[0]}x{patch_size[1]}"

    def _summary_table(self, best: PatchTrialRecord | None) -> MarkdownTable:
        """Builds the one-row table of the best patch size and its test loss.

        Args:
            best : The winning record, or None when no trial ranked.

        Returns:
            The markdown table, with empty cells when no trial ranked.
        """
        table = MarkdownTable(["Best patch (row x col, px)", "Test loss at best"], align=["right", "right"])

        if best is None:
            table.add_row(None, None)
        else:
            table.add_row(self._patch_label(best.patch_size), self._with_std(best.test_loss, best.test_loss_std))

        return table

    @staticmethod
    def _with_std(value: float | None, std: float | None) -> str | None:
        """Formats a scalar with its across-seed standard deviation, or None if absent.

        Args:
            value : Mean value, or None.
            std   : Standard deviation, or None to print the mean alone.

        Returns:
            The formatted cell text, or None when the value is missing.
        """
        if value is None:
            return None

        return ScalarFormatter.with_std(ScalarFormatter.format_scalar(value), std)

    def _trial_table(self, records: list[PatchTrialRecord], best: PatchTrialRecord | None) -> MarkdownTable:
        """Builds the per-trial table, with the winning patch in bold.

        Args:
            records : Records sorted by patch size.
            best    : The winning record, or None.

        Returns:
            The markdown table listing patch geometry, batch size, seeds scored,
            losses and status.
        """
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
        """Writes report.md with the configuration banner, ranking tables and figure.

        Args:
            records     : Records sorted by patch size.
            figure_path : Path of the loss figure over the swept patch grid.

        Returns:
            Path of the written markdown report.
        """
        row_sizes, col_sizes = self.planner.patch_sizes()
        best                       = self._best(records)
        doc                        = MarkdownDoc("Patch-size sweep")

        doc.paragraph("The same backbone is trained across every admissible combination of row and column patch lengths on the selected dataset, so the sweep shows how the held-out loss follows the receptive field along each axis.")
        doc.kv_table({
            "Model"          : self.model_label,
            "Dataset"        : Path(self.config.paths.dataset_path).name,
            "Dataset path"   : str(self.config.paths.dataset_path),
            "Secondaries"    : ", ".join(self.config.paths.secondary_labels),
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
        """Logs how many trials ranked and the winning patch with its seed-mean test loss.

        Args:
            records : Records of the sweep.
        """
        ranked = [record for record in records if record.complete]
        best   = self._best(records)

        if best is None:
            self.logger.warning(f"No patch shape ranked: none of the {len(records)} trials has a finite seed-mean test loss")
            return

        self.logger.subsection(f"Ranked {len(ranked)}/{len(records)} patch shapes; best {self._patch_label(best.patch_size)} with test loss {self._with_std(best.test_loss, best.test_loss_std)} over {best.n_seeds}/{best.n_seed_runs} seeds")

    def write_all(self) -> list[Path]:
        """Collects the trial records, writes every report artifact and logs the ranking.

        Returns:
            Paths of the JSON summary, the loss figure and the markdown report.
        """
        collector = PatchTrialCollector(runs_root=self.runs_root, planner=self.planner, logger=self.logger, seeds=self.seeds)
        records   = self._sorted_records(collector.collect())

        json_path   = self._write_json(records)
        figure_path = self._write_figure(records)
        md_path     = self._write_markdown(records, figure_path)

        self._log_ranking(records)

        return [json_path, figure_path, md_path]
