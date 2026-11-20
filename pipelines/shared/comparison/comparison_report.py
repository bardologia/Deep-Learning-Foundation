
from __future__ import annotations

import re
from pathlib import Path

from pipelines.shared.comparison.trial_collection import TrialRecord
from tools.data.io                                import FileIO
from tools.metrics.ranking                        import RankingComputer, RankingResult
from tools.metrics.scoring                        import FiniteScalar, MetricOrientation
from tools.metrics.significance                   import SignificanceVsLeader
from tools.monitoring.logger                      import Logger
from tools.reporting.markdown                     import MarkdownTable, ScalarFormatter
from tools.reporting.reporting                    import MetricSectionGrouper, ReportAssets


class ComparisonReportBase:

    SIGNIFICANCE_ALPHA = 0.05

    HEADLINE_METRICS = [
        ("rmse_mean", "RMSE"),
        ("mae_mean",  "MAE"),
        ("r2_mean",   "R²"),
    ]

    seed_dispersion: dict = {}

    _floors_warned: frozenset = frozenset()

    def _metric_seed_std(self, name: str, key: str) -> float | None:
        entry = self.seed_dispersion.get(name)
        return entry["metrics"].get(key) if entry else None

    def _seeds_cell(self, name: str) -> str:
        entry = self.seed_dispersion.get(name)
        if entry is None:
            return "1"

        n_seeds = entry["n_seeds"]
        n_used  = min(entry["n_used"].values(), default=n_seeds)

        if n_used == n_seeds:
            return str(n_seeds)

        return f"{n_used}/{n_seeds}"

    def _ranking(self, metrics: list[tuple[str, str]], scored: list[TrialRecord]) -> RankingResult:
        trials = [(r.name, r.metrics) for r in scored]
        return RankingComputer(metrics, trials).compute()

    def _significance_vs_leader(self, metric_key: str, order: list[str]) -> dict[str, dict] | None:
        per_seed = {}
        for name in order:
            entry = self.seed_dispersion.get(name, {})
            if "seed_metrics" in entry:
                per_seed[name] = {seed: metrics.get(metric_key) for seed, metrics in entry["seed_metrics"].items()}

        leader = order[0]
        if leader not in per_seed or len(per_seed) < 2:
            return None

        return SignificanceVsLeader().compute(per_seed, leader)

    @classmethod
    def _significance_cell(cls, entry: dict | None) -> str:
        if entry is None:
            return "n/a (no seed runs)"

        if entry["p_adjusted"] is None:
            n_pairs = entry["n_pairs"]
            return f"n/a ({n_pairs} shared seed{'s' if n_pairs != 1 else ''})"

        p    = entry["p_adjusted"]
        mark = "*" if p < cls.SIGNIFICANCE_ALPHA else ""
        return f"{p:.3g}{mark}"

    @staticmethod
    def _effect_cell(entry: dict | None) -> str:
        if entry is None or entry.get("cohen_d") is None:
            return "n/a"

        return f"{entry['cohen_d']:.2f}"

    def _significance_floor_note(self, significance: dict[str, dict]) -> list[str]:
        floors = sorted({(entry["n_pairs"], entry["p_min"]) for entry in significance.values() if entry["p_min"] is not None and entry["p_min"] >= self.SIGNIFICANCE_ALPHA})

        if not floors:
            return []

        clauses = [f"with n={n_pairs} paired seeds the smallest attainable two-sided p is {p_min:.4g}" for n_pairs, p_min in floors]
        note    = f"{'; '.join(clauses)}; no such row can reach * at {self.SIGNIFICANCE_ALPHA:g}, so a missing star reflects the seed count, not an absence of difference."

        unseen = frozenset(floors) - self._floors_warned
        if unseen:
            self._floors_warned = self._floors_warned | unseen
            self.logger.warning(f"Significance vs leader: {note}")

        return [f"> {note}\n"]

    def _rank_section(self, entity: str, title: str, intro: str, metrics: list[tuple[str, str]], scored: list[TrialRecord]) -> list[str]:
        if not metrics:
            return [f"## {title}\n", "_No applicable metrics available._\n"]

        result = self._ranking(metrics, scored)
        leader = result.leader_composite()

        lines = [f"## {title}\n", f"{intro}\n"]

        order        = list(result.order())
        significance = self._significance_vs_leader(metrics[0][0], order)

        header = ["#", entity, "Score", "Mean rank", "Wins", "Δ", *[label for _, label in metrics]]
        if significance is not None:
            header.append(f"p vs #1 ({metrics[0][1]})")
            header.append(f"d vs #1 ({metrics[0][1]})")

        table = MarkdownTable(header)

        for position, name in enumerate(order, start=1):
            cells = []
            for key, _ in metrics:
                cell = RankingResult.format_rank(result.metric_rank(name, key))
                if name in result.metric_leaders(key):
                    cell = f"**{cell}**"
                cells.append(cell)

            delta = result.composite[name] - leader

            if significance is not None:
                cells.append("—" if position == 1 else self._significance_cell(significance.get(name)))
                cells.append("—" if position == 1 else self._effect_cell(significance.get(name)))

            table.add_row(
                position,
                f"`{name}`",
                f"{result.composite[name]:.3f}",
                f"{result.mean_rank[name]:.2f}",
                result.wins[name],
                f"{delta:+.3f}",
                *cells,
            )

        lines += table.render()
        lines.append("")

        if significance is not None:
            lines.append(f"> p vs #1: two-sided paired permutation test on per-seed {metrics[0][1]} against the composite leader, Holm-adjusted across rows; * marks p < {self.SIGNIFICANCE_ALPHA:g}.\n")
            lines += self._significance_floor_note(significance)

        return lines


class ComparisonReport(ComparisonReportBase):

    def __init__(self, records: list[TrialRecord], out_dir: Path, embed_images: bool, logger: Logger, rank_models: bool = True, seed_dispersion: dict | None = None, model_name: str | None = None) -> None:
        self.records         = records
        self.out_dir         = out_dir
        self.model_name      = model_name
        self.embed_images    = embed_images
        self.logger          = logger
        self.rank_models     = rank_models
        self.seed_dispersion = seed_dispersion or {}
        self.has_seed_sweep  = any(entry["n_seeds"] > 1 for entry in self.seed_dispersion.values())
        self.assets          = ReportAssets(base=out_dir, embed_images=embed_images)
        self.timestamp       = self.assets.timestamp

    def _write_overview(self) -> Path:
        if self.rank_models:
            lines = self.assets.header("Benchmark Overview")
        else:
            lines  = self.assets.header("Cross-Validation Fold Overview")
            lines += [f"Folds of the single model `{self.model_name}`; each fold tests a different disjoint row region, so fold-to-fold differences reflect data heterogeneity, not model quality. No ranking is implied.\n"]

        if self.rank_models:
            lines += ["## Model Capacity\n"]
            lines += self._capacity_table()
            lines.append("")

        lines += ["## Training\n"]
        if self.has_seed_sweep:
            lines += ["Seed-swept models aggregate their seed runs: losses and metrics are seed means (± sample std), while the inference figures and report come from one representative seed. A Seeds cell of the form used/total marks a model whose least-covered metric mean rests on fewer seeds than the model holds, because some seed lacks an inference on the split or reports a non-finite value.\n"]
        lines += self._training_table()
        lines.append("")

        lines += ["## Inference\n"]
        lines += self._inference_table()
        lines.append("")

        if self.rank_models:
            lines += self._leaderboard()

        out = self.out_dir / "benchmark_overview.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _capacity_table(self) -> list[str]:
        table = MarkdownTable(["Model", "Parameters"])

        for r in self.records:
            parameters = f"{r.parameters:,}" if r.parameters is not None else "—"

            table.add_row(f"`{r.name}`", parameters)

        return table.render()

    def _training_table(self) -> list[str]:
        headers = ["Model", "Status", "Best epoch", "Best val loss", "Epochs run", "Duration"]
        if self.has_seed_sweep:
            headers.insert(1, "Seeds")
        table = MarkdownTable(headers)

        for r in self.records:
            duration_s = r.training_result.get("duration_s")
            duration   = f"{duration_s / 60:.1f} min" if duration_s is not None else "—"

            best_val_loss_std = self.seed_dispersion.get(r.name, {}).get("best_val_loss_std")

            cells = [
                f"`{r.name}`",
                r.training_result.get("status", "—"),
                ScalarFormatter.format_scalar(r.checkpoint.get("best_epoch")),
                ScalarFormatter.with_std(ScalarFormatter.format_scalar(r.checkpoint.get("best_val_loss")), best_val_loss_std),
                ScalarFormatter.format_scalar(r.checkpoint.get("n_train_epochs")),
                duration,
            ]
            if self.has_seed_sweep:
                cells.insert(1, self._seeds_cell(r.name))
            table.add_row(*cells)

        return table.render()

    def _inference_table(self) -> list[str]:
        table = MarkdownTable(["Model", "Inference run", "Figures", "Report"])

        for r in self.records:
            inference = f"`{r.inference_dir.name}`" if r.has_inference else "pending"
            report_md = f"[report.md]({self.assets.rel(r.report_path)})" if r.report_path else "—"

            table.add_row(f"`{r.name}`", inference, len(r.figures), report_md)

        return table.render()

    def _leaderboard(self) -> list[str]:
        scored = [r for r in self.records if r.metrics]
        if not scored:
            return ["## Leaderboard\n", "_No inference metrics available yet._\n"]

        headline_intro = (
            "`Score` is a magnitude-aware composite in [0, 1] (per metric, 1 = best and 0 = worst by min-max, then "
            "averaged; missing metrics score 0). `Mean rank` is the tie-averaged ordinal rank (1 = best), `Wins` "
            "counts metrics where the model is best, `Δ` is the score gap to the leader. Models are ordered by `Score`."
        )

        return self._rank_section("Model", "Leaderboard", headline_intro, self.HEADLINE_METRICS, scored)

    def _write_metrics(self) -> Path:
        scored = [r for r in self.records if r.metrics]

        if self.rank_models:
            lines  = self.assets.header("Test Metrics Comparison")
            lines += ["> Best value per metric in **bold** (↓ lower is better, ↑ higher is better).\n"]
            if self.has_seed_sweep:
                lines += ["> Each model was trained under multiple seeds; cells show the seed mean ± seed standard deviation.\n"]
        else:
            lines  = self.assets.header("Per-Fold Test Metrics")
            lines += ["> Folds of one model across disjoint row regions; no value is highlighted as best (↓ lower is better, ↑ higher is better).\n"]

        if not scored:
            lines += ["_No inference metrics available yet._\n"]
        else:
            all_keys = MetricSectionGrouper.scalar_keys(scored)

            for title, keys in MetricSectionGrouper().group(all_keys):
                lines += [f"## {title}\n", *self._metric_table(keys, scored)]

        out = self.out_dir / "metrics_comparison.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _metric_table(self, keys: list[str], scored: list[TrialRecord]) -> list[str]:
        table = MarkdownTable(["Metric", *[f"`{r.name}`" for r in scored]])

        for key in keys:
            direction = MetricOrientation.direction(key)
            arrow     = {"higher": " ↑", "lower": " ↓", None: ""}[direction]
            values    = [r.metrics.get(key) for r in scored]
            numeric   = [v for v in (FiniteScalar.coerce(value) for value in values) if v is not None]

            best = None
            if self.rank_models and direction is not None and numeric:
                best = max(numeric) if direction == "higher" else min(numeric)

            cells = []
            for r, value in zip(scored, values):
                cell   = ScalarFormatter.format_scalar(value)
                finite = FiniteScalar.coerce(value)
                if best is not None and finite is not None and finite == best:
                    cell = f"**{cell}**"
                cell = ScalarFormatter.with_std(cell, self._metric_seed_std(r.name, key))
                cells.append(cell)

            table.add_row(f"`{key}`{arrow}", *cells)

        return [*table.render(), ""]

    @staticmethod
    def _figure_names(record: TrialRecord) -> list[str]:
        return [path.relative_to(record.inference_dir / "figures").as_posix() for path in record.figures]

    def _write_figures(self) -> list[Path]:
        scored = [r for r in self.records if r.has_inference]
        names  = sorted({name for r in scored for name in self._figure_names(r)}, key=ReportAssets.natural_key)
        groups = {}

        for name in names:
            groups.setdefault(name.split("/")[0], []).append(name)

        return [self._write_figure_file(scored, group, f"Figures – {channel}", f"figures_{re.sub(r'[^a-z0-9]+', '_', channel.lower()).strip('_')}.md") for channel, group in groups.items()]

    def _write_figure_file(self, scored: list[TrialRecord], names: list[str], title: str, filename: str) -> Path:
        lines  = self.assets.header(title)
        lines += ["> Only runs with at least one completed inference are shown.\n"]

        for name in names:
            lines.append(f"## `{name}`\n")
            for r in scored:
                path = r.inference_dir / "figures" / name
                if path.exists():
                    lines.append(f"*{r.name}*  \n![]({self.assets.src(path)})\n")
                else:
                    lines.append(f"*{r.name}* — _(not found)_\n")
            lines.append("")

        out = self.out_dir / filename
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_summary_json(self) -> Path:
        payload = []
        for r in self.records:
            entry = {
                "name"            : r.name,
                "run_dir"         : str(r.run_dir),
                "parameters"      : r.parameters,
                "training_result" : r.training_result,
                "checkpoint"      : r.checkpoint,
                "run_summary"     : r.run_summary,
                "trainer_config"  : r.trainer_config,
                "inference_dir"   : str(r.inference_dir) if r.inference_dir else None,
                "metrics"         : r.metrics,
                "figures"         : self._figure_names(r),
                "report_path"     : str(r.report_path) if r.report_path else None,
            }

            if r.name in self.seed_dispersion:
                entry["seed_dispersion"] = self.seed_dispersion[r.name]

            payload.append(entry)

        out = self.out_dir / "comparison_summary.json"
        return FileIO.save_json(payload, out, indent=2)

    def write_all(self) -> list[Path]:
        FileIO.ensure_dir(self.out_dir)

        written = [self._write_overview(), self._write_metrics()]
        written.extend(self._write_figures())
        written.append(self._write_summary_json())

        return written
