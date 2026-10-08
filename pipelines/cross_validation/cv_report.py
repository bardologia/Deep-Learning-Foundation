"""Aggregate reporting across the folds of a cross-validation sweep.

Writes a markdown report with the fold plan, per-fold training summary, and
metric means and standard deviations across folds, plus a machine-readable
`cv_summary.json` and one per-split comparison report.
"""

from __future__ import annotations

import math
from pathlib import Path

from pipelines.shared.comparison.comparison_report import ComparisonReport
from pipelines.shared.comparison.trial_collection  import TrialRecord
from pipelines.cross_validation.folds              import FoldPlanner
from tools.data.io             import FileIO
from tools.reporting.reporting import MetricSectionGrouper
from tools.metrics.scoring     import FiniteScalar, SeedAggregation
from tools.monitoring.logger   import Logger
from tools.reporting.markdown  import MarkdownTable, ScalarFormatter
from tools.runtime.run_tag     import RunTag


class CrossValidationReport:
    """Aggregates fold records into the cross-validation deliverables.

    Attributes:
        base_records     : One record per fold holding training results only.
        records_by_split : Fold records carrying inference metrics, keyed by split.
        planner          : Planner describing the row blocks each fold was built from.
        out_dir          : Directory the reports are written to.
        model_name       : Label of the model all folds share.
        embed_images     : Whether per-split comparison reports inline images.
        logger           : Logger used for progress reporting.
        seed_dispersion  : Within-fold seed statistics keyed by fold name.
        has_seed_sweep   : True when at least one fold aggregates several seeds.
        grouper          : Groups metric keys into report sections.
        timestamp        : Generation timestamp stamped into the aggregate report.
    """

    def __init__(
        self,
        base_records     : list[TrialRecord],
        records_by_split : dict[str, list[TrialRecord]],
        planner          : FoldPlanner,
        out_dir          : Path,
        model_name       : str,
        embed_images     : bool,
        logger           : Logger,
        seed_dispersion  : dict | None = None,
    ) -> None:
        """Binds the report to the collected fold records.

        Args:
            base_records     : One record per fold holding training results.
            records_by_split : Fold records with inference metrics, keyed by split.
            planner          : Planner describing each fold's row blocks.
            out_dir          : Directory the reports are written to.
            model_name       : Label of the model all folds share.
            embed_images     : Whether per-split comparison reports inline images.
            logger           : Logger used for progress reporting.
            seed_dispersion  : Within-fold seed statistics keyed by fold name.
        """
        self.base_records     = base_records
        self.records_by_split = records_by_split
        self.planner          = planner
        self.out_dir          = out_dir
        self.model_name       = model_name
        self.embed_images     = embed_images
        self.logger           = logger
        self.seed_dispersion  = seed_dispersion or {}
        self.has_seed_sweep   = any(entry["n_seeds"] > 1 for entry in self.seed_dispersion.values())
        self.grouper          = MetricSectionGrouper()
        self.timestamp        = RunTag.timestamp()

    def _seed_std(self, fold_name: str, split: str | None, key: str) -> float | None:
        """Returns the within-fold seed standard deviation of one quantity.

        Args:
            fold_name : Name of the fold.
            split     : Dataset split for a metric, or None for the training checkpoint quantity `best_val_loss`.
            key       : Metric or checkpoint key.

        Returns:
            Seed standard deviation, or None when the fold was not seed-swept
            or the key carries no dispersion.
        """
        entry = self.seed_dispersion.get(fold_name)
        if entry is None:
            return None

        if split is None:
            return entry["best_val_loss_std"] if key == "best_val_loss" else None

        return entry["splits"].get(split, {}).get(key)

    def _require_all_folds(self) -> None:
        """Verifies every fold trained and, where requested, produced metrics.

        Raises:
            ValueError : If a fold is missing, has no checkpoint data, or lacks metrics on a split being reported.
        """
        n_total = self.planner.n_folds

        if len(self.base_records) != n_total:
            collected = [record.name for record in self.base_records]
            raise ValueError(f"Cross-validation aggregate requires all {n_total} folds, but only {len(self.base_records)} were collected: {collected}")

        scoreless = [record.name for record in self.base_records if not record.checkpoint]
        if scoreless:
            raise ValueError(f"Cross-validation aggregate requires every fold to have trained; folds missing checkpoint data: {scoreless}")

        for split, records in self.records_by_split.items():
            missing = [record.name for record in records if not record.metrics]
            if missing:
                raise ValueError(f"Cross-validation aggregate requires metrics for all {n_total} folds on split '{split}'; missing folds: {missing}")

    def _warn_omitted(self) -> None:
        """Warns, per split, about the metrics no fold holds as a finite value, which the aggregate tables and the summary JSON leave out."""
        for split, records in self.records_by_split.items():
            omitted = [key for key in MetricSectionGrouper.scalar_keys(records) if all(FiniteScalar.coerce(record.metrics.get(key)) is None for record in records)]
            if omitted:
                self.logger.warning(f"split {split}: {omitted} non-finite in every fold, omitted from the aggregate")

    def _format_std(self, std: float | None, n_used: int, scale: float = 1.0) -> str:
        """Formats a standard deviation, blank when fewer than two folds contributed.

        Args:
            std    : Sample standard deviation across folds.
            n_used : Number of folds that contributed a finite value.
            scale  : Divisor applied before formatting, for unit conversion.

        Returns:
            Formatted cell text, or the empty-cell placeholder.
        """
        if self._json_std(std, n_used) is None:
            return ScalarFormatter.EMPTY

        return ScalarFormatter.format_scalar(std / scale)

    def _json_std(self, std: float | None, n_used: int) -> float | None:
        """Returns the standard deviation, or None when it is undefined or non-finite."""
        if n_used < 2 or std is None or not math.isfinite(std):
            return None
        return std

    def _fold_label(self, fold_name: str, split: str | None) -> str:
        """Returns a fold's column header, carrying the seeds used over the seeds held when the fold was seed-swept.

        Args:
            fold_name : Name of the fold.
            split     : Split whose seed coverage is shown, or None for a training-only column.
        """
        entry = self.seed_dispersion.get(fold_name)
        if not self.has_seed_sweep or entry is None or split is None:
            return f"`{fold_name}`"

        return f"`{fold_name}` ({entry['n_used'][split]}/{entry['n_seeds']} seeds)"

    def _aggregate_table(self, keys: list[str], records: list[TrialRecord], split: str | None = None) -> list[str]:
        """Renders a metric-by-fold table with the across-fold mean and std.

        Args:
            keys    : Metric keys forming the table rows.
            records : Fold records forming the per-fold columns.
            split   : Split the metrics belong to, used to look up seed dispersion.

        Returns:
            Markdown lines of the table followed by a blank line.
        """
        n_total     = len(records)
        fold_labels = [self._fold_label(record.name, split) for record in records]
        table       = MarkdownTable(["Metric", "Mean", "Std", "N (folds used)", *fold_labels])

        for key in keys:
            values  = [record.metrics.get(key) for record in records]
            numeric = [value for value in (FiniteScalar.coerce(value) for value in values) if value is not None]

            if not numeric:
                continue

            mean, std  = SeedAggregation.mean_std(numeric)
            mean_cell  = ScalarFormatter.format_scalar(mean)
            std_cell   = self._format_std(std, len(numeric))
            count_cell = f"{len(numeric)}/{n_total}"
            fold_cells = [ScalarFormatter.with_std(ScalarFormatter.format_scalar(value), self._seed_std(record.name, split, key)) for record, value in zip(records, values)]

            table.add_row(f"`{key}`", mean_cell, std_cell, count_cell, *fold_cells)

        return [*table.render(), ""]

    def _write_aggregate(self) -> Path:
        """Writes the fold plan, training summary, and metric aggregates.

        Returns:
            Path of the written `cv_aggregate_report.md`.
        """
        lines = [
            "# Cross-Validation Aggregate Report",
            f"\n_Generated {self.timestamp}_\n",
            f"Model `{self.model_name}`, {self.planner.n_folds} folds. All aggregates are reported as mean and sample standard deviation across folds.\n",
        ]

        if self.has_seed_sweep:
            lines += ["Each fold was trained under multiple seeds. Per-fold cells show the seed mean ± within-fold seed standard deviation; the Mean and Std columns aggregate the per-fold seed means across folds.\n"]

        lines += ["## Fold Plan\n"]

        plan_table = MarkdownTable(["Fold", "Test row", "Val row", "Train row regions"])
        for plan in self.planner.plans():
            test_region   = plan.split_regions.regions("test")[0]
            val_region    = plan.split_regions.regions("val")[0]
            train_regions = plan.split_regions.regions("train")
            train_text    = ", ".join(f"[{r.row_start}, {r.row_end})" for r in train_regions)
            test_text     = f"[{test_region.row_start}, {test_region.row_end})"
            val_text      = f"[{val_region.row_start}, {val_region.row_end})"

            plan_table.add_row(plan.fold_index, test_text, val_text, train_text)

        lines += [*plan_table.render(), ""]

        lines += self._training_aggregate()

        for split, records in self.records_by_split.items():
            lines += [f"## Split `{split}` — Metric Aggregates\n"]

            for title, keys in self.grouper.group(MetricSectionGrouper.scalar_keys(records)):
                lines += [f"### {title}\n", *self._aggregate_table(keys, records, split)]

        out = self.out_dir / "cv_aggregate_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    SUMMARY_QUANTITIES = [
        ("Best val loss",  "checkpoint",      "best_val_loss", 1.0),
        ("Best epoch",     "checkpoint",      "best_epoch",    1.0),
        ("Duration (min)", "training_result", "duration_s",    60.0),
    ]

    def _training_aggregate(self) -> list[str]:
        """Returns the per-fold training table and its across-fold summary.

        Reports the training status, best epoch, best validation loss, epochs
        run, and wall-clock duration in minutes for every fold, plus the seed
        count when the folds were seed-swept.
        """
        lines   = ["## Training Across Folds\n"]
        headers = ["Fold", "Status", "Best epoch", "Best val loss", "Epochs run", "Duration"]
        if self.has_seed_sweep:
            headers.insert(1, "Seeds")
        table = MarkdownTable(headers)

        collected = {key: [] for _, _, key, _ in self.SUMMARY_QUANTITIES}

        for record in self.base_records:
            for _, source, key, _ in self.SUMMARY_QUANTITIES:
                value = getattr(record, source).get(key)
                if isinstance(value, (int, float)):
                    collected[key].append(float(value))

            duration_s = record.training_result.get("duration_s")
            duration   = f"{duration_s / 60:.1f} min" if duration_s is not None else "—"
            status     = record.training_result.get("status", "—")
            epoch_cell = ScalarFormatter.format_scalar(record.checkpoint.get("best_epoch"))
            loss_cell  = ScalarFormatter.with_std(ScalarFormatter.format_scalar(record.checkpoint.get("best_val_loss")), self._seed_std(record.name, None, "best_val_loss"))
            run_cell   = ScalarFormatter.format_scalar(record.checkpoint.get("n_train_epochs"))

            cells = [f"`{record.name}`", status, epoch_cell, loss_cell, run_cell, duration]
            if self.has_seed_sweep:
                cells.insert(1, str(self.seed_dispersion.get(record.name, {}).get("n_seeds", 1)))

            table.add_row(*cells)

        lines += [*table.render(), ""]

        n_total = self.planner.n_folds

        summary = MarkdownTable(["Quantity", "Mean", "Std", "Folds"])
        for label, _, key, scale in self.SUMMARY_QUANTITIES:
            values = collected[key]
            if not values:
                continue
            mean, std  = SeedAggregation.mean_std(values)
            mean_cell  = ScalarFormatter.format_scalar(mean / scale)
            std_cell   = self._format_std(std, len(values), scale)
            count_cell = f"{len(values)}/{n_total}"

            summary.add_row(label, mean_cell, std_cell, count_cell)

        if not summary.is_empty():
            lines += [*summary.render(), ""]

        return lines

    def _write_summary_json(self) -> Path:
        """Writes the machine-readable aggregate to `cv_summary.json`.

        Returns:
            Path of the written JSON file, holding per-split metric means,
            standard deviations, fold counts, per-fold values, and the
            aggregated best validation loss.
        """
        payload: dict = {
            "model"   : self.model_name,
            "n_folds" : self.planner.n_folds,
            "folds"   : [record.name for record in self.base_records],
            "splits"  : {},
        }

        n_total = self.planner.n_folds

        for split, records in self.records_by_split.items():
            scored = [record for record in records if record.metrics]
            keys   = MetricSectionGrouper.scalar_keys(scored)

            split_payload = {}
            for key in keys:
                values = [value for value in (FiniteScalar.coerce(record.metrics.get(key)) for record in scored) if value is not None]
                if not values:
                    continue
                mean, std = SeedAggregation.mean_std(values)
                split_payload[key] = {
                    "mean"     : mean,
                    "std"      : self._json_std(std, len(values)),
                    "n_used"   : len(values),
                    "n_total"  : n_total,
                    "per_fold" : {record.name: record.metrics.get(key) for record in scored},
                }

                if self.has_seed_sweep:
                    split_payload[key]["per_fold_seed_std"] = {record.name: self._seed_std(record.name, split, key) for record in scored}

            payload["splits"][split] = split_payload

        best_losses = [value for value in (FiniteScalar.coerce(record.checkpoint.get("best_val_loss")) for record in self.base_records) if value is not None]
        if best_losses:
            mean, std = SeedAggregation.mean_std(best_losses)
            payload["best_val_loss"] = {"mean": mean, "std": self._json_std(std, len(best_losses)), "n_used": len(best_losses), "n_total": n_total}

            if self.has_seed_sweep:
                payload["best_val_loss"]["per_fold_seed_std"] = {record.name: self._seed_std(record.name, None, "best_val_loss") for record in self.base_records}

        if self.seed_dispersion:
            payload["fold_status"] = {fold: entry["status"] for fold, entry in self.seed_dispersion.items()}

        if self.has_seed_sweep:
            payload["seeds_per_fold"]      = {fold: entry["n_seeds"] for fold, entry in self.seed_dispersion.items()}
            payload["seeds_used_per_fold"] = {fold: entry["n_used"] for fold, entry in self.seed_dispersion.items()}
            payload["missing_seeds"]       = {fold: entry["missing_seeds"] for fold, entry in self.seed_dispersion.items()}

        out = self.out_dir / "cv_summary.json"
        return FileIO.save_json(payload, out, indent=2)

    def write_all(self) -> list[Path]:
        """Writes the aggregate report, per-split comparisons, and the summary JSON.

        Returns:
            Paths of every written file.

        Raises:
            ValueError : If any fold is missing training or metric data.
        """
        self._require_all_folds()
        self._warn_omitted()

        FileIO.ensure_dir(self.out_dir)

        written = [self._write_aggregate()]

        for split, records in self.records_by_split.items():
            report = ComparisonReport(
                records         = records,
                out_dir         = self.out_dir / split,
                reference_model = self.model_name,
                embed_images    = self.embed_images,
                logger          = self.logger,
                rank_models     = False,
            )
            written.extend(report.write_all())

        written.append(self._write_summary_json())

        return written
