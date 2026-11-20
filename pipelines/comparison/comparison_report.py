
from __future__ import annotations

import re
from pathlib import Path

from pipelines.comparison.trial_collector          import TrialRecord
from pipelines.shared.comparison.comparison_report import ComparisonReportBase
from tools.data.io                                 import FileIO
from tools.metrics.scoring                         import FiniteScalar, MetricOrientation
from tools.reporting.markdown                      import MarkdownTable, ScalarFormatter
from tools.reporting.reporting                     import MetricSectionGrouper, ReportAssets
from tools.monitoring.logger                       import Logger


class TrialComparisonReport(ComparisonReportBase):

    def __init__(
        self,
        records         : list[TrialRecord],
        out_dir         : Path,
        compare_images  : bool,
        embed_images    : bool,
        inference_split : str,
        logger          : Logger,
        seed_dispersion : dict | None = None,
    ) -> None:
        self.records         = records
        self.out_dir         = Path(out_dir)
        self.compare_images  = compare_images
        self.inference_split = inference_split
        self.logger          = logger
        self.seed_dispersion = seed_dispersion or {}
        self.has_seed_sweep  = any(entry["n_seeds"] > 1 for entry in self.seed_dispersion.values())
        self.assets          = ReportAssets(base=self.out_dir, embed_images=embed_images)

    def _write_overview(self) -> Path:
        lines  = self.assets.header("Trial Comparison Overview")
        lines += [f"Metrics and figures come from each trial's newest completed inference on the `{self.inference_split}` split; a trial without one is listed but not ranked.\n"]
        lines += ["## Runs\n"]

        if self.has_seed_sweep:
            lines += ["Seed-swept trials aggregate their seed runs: metrics and losses are seed means (± sample std), while the inference run, figures, and report link come from one representative seed. A Seeds cell of the form used/total marks a trial whose least-covered metric mean rests on fewer seeds than the trial holds, because some seed lacks an inference on the split or reports a non-finite value.\n"]

        headers = ["Run", "Seeds", "Inference run", "Checkpoint", "Best val loss", "Epochs", "Report"] if self.has_seed_sweep else ["Run", "Inference run", "Checkpoint", "Best val loss", "Epochs", "Report"]
        table   = MarkdownTable(tuple(headers))

        for r in self.records:
            inference  = f"`{r.inference_dir.name}`" if r.has_inference else "_(no inference)_"
            epoch      = ScalarFormatter.format_scalar(r.checkpoint.get("epoch"))
            best_epoch = ScalarFormatter.format_scalar(r.checkpoint.get("best_epoch"))
            best_std   = self.seed_dispersion.get(r.name, {}).get("best_val_loss_std")
            best_loss  = ScalarFormatter.with_std(ScalarFormatter.format_scalar(r.checkpoint.get("best_val_loss")), best_std)
            n_epochs   = ScalarFormatter.format_scalar(r.checkpoint.get("n_train_epochs"))
            report_lnk = f"[report.md]({self.assets.rel(r.report_path)})" if r.report_path else "—"

            cells = [f"`{r.name}`", inference, f"{epoch} (best {best_epoch})", best_loss, n_epochs, report_lnk]
            if self.has_seed_sweep:
                cells.insert(1, self._seeds_cell(r.name))
            table.add_row(*cells)

        lines += table.render()
        lines.append("")
        lines += self._leaderboard()

        out = self.out_dir / "overview.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _leaderboard(self) -> list[str]:
        scored = [r for r in self.records if r.metrics]
        if not scored:
            return ["## Leaderboard\n", "_No inference metrics available._\n"]

        headline_intro = (
            "`Score` is a magnitude-aware composite in [0, 1] (per metric, 1 = best and 0 = worst by min-max, then "
            "averaged; missing metrics score 0). `Mean rank` is the tie-averaged ordinal rank (1 = best; missing "
            "metrics rank last), `Wins` counts metrics where the trial is best, `Δ` is the score gap to the leader. "
            "Trials are ordered by `Score`. Per-metric cells show the rank; the best is in **bold**."
        )

        return self._rank_section("Run", "Leaderboard", headline_intro, self.HEADLINE_METRICS, scored)

    def _write_metrics(self) -> Path:
        scored = [r for r in self.records if r.metrics]

        lines  = self.assets.header("Metrics Comparison")
        lines += ["> Best value per metric in **bold** (↓ lower is better, ↑ higher is better).\n"]

        if not scored:
            lines += ["_No inference metrics available._\n"]
        else:
            all_keys = MetricSectionGrouper.scalar_keys(scored)
            for title, keys in MetricSectionGrouper().group(all_keys):
                lines += [f"## {title}\n", *self._metric_table(keys, scored)]

        out = self.out_dir / "metrics_comparison.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _metric_table(self, keys: list[str], scored: list[TrialRecord]) -> list[str]:
        headers = ["Run"]
        best    : dict[str, float | None] = {}

        for key in keys:
            direction = MetricOrientation.direction(key)
            arrow     = {"higher": " ↑", "lower": " ↓", None: ""}[direction]
            headers.append(f"`{key}`{arrow}")

            numeric = [v for v in (FiniteScalar.coerce(r.metrics.get(key)) for r in scored) if v is not None]
            best[key] = (max(numeric) if direction == "higher" else min(numeric)) if direction is not None and numeric else None

        table = MarkdownTable(headers)

        for r in scored:
            cells = [f"`{r.name}`"]
            for key in keys:
                value  = r.metrics.get(key)
                cell   = ScalarFormatter.format_scalar(value)
                finite = FiniteScalar.coerce(value)
                if best[key] is not None and finite is not None and finite == best[key]:
                    cell = f"**{cell}**"
                cells.append(ScalarFormatter.with_std(cell, self._metric_seed_std(r.name, key)))
            table.add_row(*cells)

        return [*table.render(), ""]

    def _figure_subdirs(self) -> list[str]:
        return sorted({path.name for r in self.records if r.figures_dir is not None for path in r.figures_dir.iterdir() if path.is_dir()}, key=ReportAssets.natural_key)

    def _write_figure_section(self, subdir: str) -> Path | None:
        resolved = [(r, r.figure_subdir(subdir)) for r in self.records if r.has_inference]
        scored   = [(r, path, {p.name for p in path.glob("*.png")}) for r, path in resolved if path is not None]

        all_names = {name for _record, _path, names in scored for name in names}
        if not all_names:
            return None

        sorted_names = sorted(all_names, key=ReportAssets.natural_key)

        lines  = self.assets.header(f"Figure Comparison – {subdir}")
        lines += [f"> Figures from the `{subdir}/` directory. Only trials with a completed inference run are shown.\n"]

        for name in sorted_names:
            lines.append(f"## `{name}`\n")
            for record, path, names in scored:
                if name in names:
                    lines.append(f"*{record.name}*  \n![]({self.assets.src(path / name)})\n")
                else:
                    lines.append(f"*{record.name}* — _(not in this run)_\n")
            lines.append("")

        slug = re.sub(r"[^a-z0-9]+", "_", subdir.lower()).strip("_")
        out  = self.out_dir / f"figures_{slug}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_figure_galleries(self) -> list[Path]:
        written = []
        for subdir in self._figure_subdirs():
            path = self._write_figure_section(subdir)
            if path:
                self.logger.info(f"Figures [{subdir}] : {path.name}")
                written.append(path)

        if not written:
            self.logger.warning("No trial with a completed inference holds figures in any sub-directory; no figure gallery written")

        return written

    def _write_summary_json(self) -> Path:
        trials = []
        for r in self.records:
            entry = {
                "name"          : r.name,
                "run_dir"       : str(r.run_dir),
                "inference_dir" : str(r.inference_dir) if r.inference_dir else None,
                "report_path"   : str(r.report_path) if r.report_path else None,
                "checkpoint"    : r.checkpoint,
                "metrics"       : r.metrics,
            }

            if r.name in self.seed_dispersion:
                entry["seed_dispersion"] = self.seed_dispersion[r.name]

            trials.append(entry)

        payload = {"inference_split": self.inference_split, "trials": trials}

        return FileIO.save_json(payload, self.out_dir / "comparison_summary.json", indent=2)

    def write_all(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.logger.section("Writing comparison reports")

        written = [self._write_overview(), self._write_metrics(), self._write_summary_json()]
        self.logger.info(f"Overview, metrics and summary written for split '{self.inference_split}'")

        if self.compare_images:
            written += self._write_figure_galleries()

        return written
