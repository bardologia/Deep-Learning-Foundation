from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pipelines.shared.comparison.comparison_report import ComparisonReport
from pipelines.shared.comparison.trial_collection  import TrialRecord
from pipelines.cross_validation.folds              import FoldPlanner
from tools.data.io             import FileIO
from tools.reporting.reporting import MetricSectionGrouper
from tools.metrics.scoring     import FiniteScalar
from tools.monitoring.logger   import Logger
from tools.reporting.markdown  import MarkdownTable, ScalarFormatter
from tools.runtime.run_tag     import RunTag


class CrossValidationReport:
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
        entry = self.seed_dispersion.get(fold_name)
        if entry is None:
            return None

        if split is None:
            return entry["best_val_loss_std"] if key == "best_val_loss" else None

        return entry["splits"].get(split, {}).get(key)

    def _seed_annotated(self, cell: str, std: float | None) -> str:
        if std is None or not math.isfinite(std):
            return cell

        return f"{cell} ± {ScalarFormatter.format_scalar(std)}"

    def _require_all_folds(self) -> None:
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

    def _mean_std(self, values: list[float]) -> tuple[float, float]:
        n    = len(values)
        mean = float(np.mean(values))

        if n < 2:
            return mean, float("nan")

        return mean, float(np.std(values, ddof=1))

    def _format_std(self, std: float, n_used: int) -> str:
        if n_used < 2 or not math.isfinite(std):
            return ScalarFormatter.EMPTY
        return ScalarFormatter.format_scalar(std)

    def _json_std(self, std: float, n_used: int) -> float | None:
        if n_used < 2 or not math.isfinite(std):
            return None
        return std

    def _aggregate_table(self, keys: list[str], records: list[TrialRecord], split: str | None = None) -> list[str]:
        n_total     = len(records)
        fold_labels = [f"`{record.name}`" for record in records]
        table       = MarkdownTable(["Metric", "Mean", "Std", "N (folds used)", *fold_labels])

        for key in keys:
            values  = [record.metrics.get(key) for record in records]
            numeric = [value for value in (FiniteScalar.coerce(value) for value in values) if value is not None]

            if not numeric:
                continue

            mean, std  = self._mean_std(numeric)
            mean_cell  = ScalarFormatter.format_scalar(mean)
            std_cell   = self._format_std(std, len(numeric))
            count_cell = f"{len(numeric)}/{n_total}"
            fold_cells = [self._seed_annotated(ScalarFormatter.format_scalar(value), self._seed_std(record.name, split, key)) for record, value in zip(records, values)]

            table.add_row(f"`{key}`", mean_cell, std_cell, count_cell, *fold_cells)

        return [*table.render(), ""]

    def _write_aggregate(self) -> Path:
        lines = [
            "# Cross-Validation Aggregate Report",
            f"\n_Generated {self.timestamp}_\n",
            f"Model `{self.model_name}`, {self.planner.n_folds} folds. All aggregates are reported as mean and sample standard deviation across folds.\n",
        ]

        if self.has_seed_sweep:
            lines += ["Each fold was trained under multiple seeds. Per-fold cells show the seed mean ± within-fold seed standard deviation; the Mean and Std columns aggregate the per-fold seed means across folds.\n"]

        lines += ["## Fold Plan\n"]

        plan_table = MarkdownTable(["Fold", "Test azimuth", "Val azimuth", "Train azimuth regions"])
        for plan in self.planner.plans():
            test_region   = plan.split_regions.regions("test")[0]
            val_region    = plan.split_regions.regions("val")[0]
            train_regions = plan.split_regions.regions("train")
            train_text    = ", ".join(f"[{r.azimuth_start}, {r.azimuth_end})" for r in train_regions)
            test_text     = f"[{test_region.azimuth_start}, {test_region.azimuth_end})"
            val_text      = f"[{val_region.azimuth_start}, {val_region.azimuth_end})"

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
        lines = ["## Training Across Folds\n"]
        table = MarkdownTable(["Fold", "Best epoch", "Best val loss", "Epochs run", "Duration"])

        collected = {key: [] for _, _, key, _ in self.SUMMARY_QUANTITIES}

        for record in self.base_records:
            for _, source, key, _ in self.SUMMARY_QUANTITIES:
                value = getattr(record, source).get(key)
                if isinstance(value, (int, float)):
                    collected[key].append(float(value))

            duration_s = record.training_result.get("duration_s")
            duration   = f"{duration_s / 60:.1f} min" if duration_s is not None else "—"
            epoch_cell = ScalarFormatter.format_scalar(record.checkpoint.get("best_epoch"))
            loss_cell  = self._seed_annotated(ScalarFormatter.format_scalar(record.checkpoint.get("best_val_loss")), self._seed_std(record.name, None, "best_val_loss"))
            run_cell   = ScalarFormatter.format_scalar(record.checkpoint.get("n_train_epochs"))

            table.add_row(f"`{record.name}`", epoch_cell, loss_cell, run_cell, duration)

        lines += [*table.render(), ""]

        n_total = self.planner.n_folds

        summary = MarkdownTable(["Quantity", "Mean", "Std", "Folds"])
        for label, _, key, scale in self.SUMMARY_QUANTITIES:
            values = collected[key]
            if not values:
                continue
            mean, std  = self._mean_std(values)
            mean_cell  = ScalarFormatter.format_scalar(mean / scale)
            std_cell   = self._format_std(std / scale, len(values))
            count_cell = f"{len(values)}/{n_total}"

            summary.add_row(label, mean_cell, std_cell, count_cell)

        if not summary.is_empty():
            lines += [*summary.render(), ""]

        return lines

    def _write_summary_json(self) -> Path:
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
                mean, std = self._mean_std(values)
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
            mean, std = self._mean_std(best_losses)
            payload["best_val_loss"] = {"mean": mean, "std": self._json_std(std, len(best_losses)), "n_used": len(best_losses), "n_total": n_total}

            if self.has_seed_sweep:
                payload["best_val_loss"]["per_fold_seed_std"] = {record.name: self._seed_std(record.name, None, "best_val_loss") for record in self.base_records}

        if self.has_seed_sweep:
            payload["seeds_per_fold"] = {fold: entry["n_seeds"] for fold, entry in self.seed_dispersion.items()}

        out = self.out_dir / "cv_summary.json"
        return FileIO.save_json(payload, out, indent=2)

    def write_all(self) -> list[Path]:
        self._require_all_folds()

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
