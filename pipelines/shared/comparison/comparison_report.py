from __future__ import annotations

import math
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
    HEADLINE_METRICS = [
        ("curve_rmse_gt",                "RMSE"),
        ("curve_mae_gt",                 "MAE"),
        ("overall_r2_gt",                "R²"),
        ("psnr_db_gt",                   "PSNR"),
        ("pixel_r2_gt_mean",             "Pixel R²"),
        ("pixel_cosine_gt_mean",         "Cosine"),
        ("ssim_gt_elev_mean",            "SSIM elev"),
        ("pixel_peak_err_units_mean_gt", "Peak err"),
    ]

    HEADLINE_GROUPS = [
        ("Curve error",           ["curve_rmse_gt", "curve_mae_gt", "psnr_db_gt"]),
        ("Variance explained",    ["overall_r2_gt", "pixel_r2_gt_mean"]),
        ("Structural similarity", ["ssim_gt_elev_mean"]),
        ("Shape (cosine)",        ["pixel_cosine_gt_mean"]),
        ("Peak location",         ["pixel_peak_err_units_mean_gt"]),
    ]

    seed_dispersion: dict = {}

    def _seed_annotated(self, cell: str, std: float | None) -> str:
        if std is None or not math.isfinite(std):
            return cell

        return f"{cell} ± {ScalarFormatter.format_scalar(std)}"

    def _metric_seed_std(self, name: str, key: str) -> float | None:
        entry = self.seed_dispersion.get(name)
        return entry["metrics"].get(key) if entry else None

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

    @staticmethod
    def _significance_cell(entry: dict | None) -> str:
        if entry is None or entry.get("p_adjusted") is None:
            return "n/a"

        p    = entry["p_adjusted"]
        mark = "*" if p < 0.05 else ""
        return f"{p:.3g}{mark}"

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
            lines.append(f"> p vs #1: two-sided paired permutation test on per-seed {metrics[0][1]} against the composite leader, Holm-adjusted across rows; * marks p < 0.05.\n")

        return lines

    def _grouped_section(self, entity: str, title: str, intro: str, groups: list[tuple[str, list[str]]], scored: list[TrialRecord]) -> list[str]:
        metrics = [(key, key) for _, keys in groups for key in keys]
        if not metrics:
            return [f"## {title}\n", "_No applicable metrics available._\n"]

        result    = self._ranking(metrics, scored)
        breakdown = result.group_breakdown(groups)
        if not breakdown.labels:
            return [f"## {title}\n", "_No applicable metrics available._\n"]

        leader = breakdown.leader_overall()

        lines = [f"## {title}\n", f"{intro}\n"]

        table = MarkdownTable(["#", entity, "Grouped score", "Δ", *breakdown.labels])

        for position, name in enumerate(breakdown.order(), start=1):
            cells = []
            for label in breakdown.labels:
                score = breakdown.group_score[name][label]
                rank  = breakdown.group_rank[label][name]
                cell  = f"{score:.3f} ({RankingResult.format_rank(rank)})"
                if name in breakdown.group_leaders(label):
                    cell = f"**{cell}**"
                cells.append(cell)

            delta = breakdown.overall[name] - leader

            table.add_row(position, f"`{name}`", f"{breakdown.overall[name]:.3f}", f"{delta:+.3f}", *cells)

        lines += table.render()
        lines.append("")
        return lines


class ComparisonReport(ComparisonReportBase):

    FIGURE_GROUPS = [
        ("Profile reconstructions",     re.compile(r"^profiles_")),
        ("Per-pixel metric maps",       re.compile(r"^(pixel_|metric_histograms)")),
        ("Gaussian parameter analysis", re.compile(r"^param_")),
        ("Active Gaussian count",       re.compile(r"^active_count")),
        ("SSIM and elevation curves",   re.compile(r"^(ssim_|elev_metric)")),
        ("Azimuth slices",              re.compile(r"^slice_azimuth_")),
        ("Elevation slices",            re.compile(r"^slice_elev_")),
        ("Range slices",                re.compile(r"^slice_range_")),
    ]

    def __init__(self, records: list[TrialRecord], out_dir: Path, reference_model: str, embed_images: bool, logger: Logger, rank_models: bool = True, seed_dispersion: dict | None = None) -> None:
        self.records         = records
        self.out_dir         = out_dir
        self.reference_model = reference_model
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
            lines += [f"Folds of the single model `{self.reference_model}`; each fold tests a different disjoint azimuth region, so fold-to-fold differences reflect data heterogeneity, not model quality. No ranking is implied.\n"]

        if self.rank_models:
            lines += ["## Model Capacity\n"]
            lines += [f"Reference model: `{self.reference_model}`.\n"]
            lines += self._capacity_table()
            lines.append("")

        lines += ["## Training\n"]
        if self.has_seed_sweep:
            lines += ["Seed-swept models aggregate their seed runs: losses and metrics are seed means (± sample std), while the inference figures and report come from one representative seed.\n"]
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
        table = MarkdownTable(["Model", "Parameters", "Δ vs reference", "Width scale", "Scaled attributes"])

        for r in self.records:
            parameters = f"{r.parameters:,}" if r.parameters is not None else "—"
            deviation  = f"{r.size_match['deviation_pct']:+.3f} %" if "deviation_pct" in r.size_match else "—"
            scale      = f"{r.size_match['scale']:.4f}" if "scale" in r.size_match else "—"
            overrides  = ", ".join(f"`{k}={v}`" for k, v in r.size_match["overrides"].items()) if "overrides" in r.size_match and r.size_match["overrides"] else "_(defaults)_"

            table.add_row(f"`{r.name}`", parameters, deviation, scale, overrides)

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
                self._seed_annotated(ScalarFormatter.format_scalar(r.checkpoint.get("best_val_loss")), best_val_loss_std),
                ScalarFormatter.format_scalar(r.checkpoint.get("n_train_epochs")),
                duration,
            ]
            if self.has_seed_sweep:
                cells.insert(1, str(self.seed_dispersion.get(r.name, {}).get("n_seeds", 1)))
            table.add_row(*cells)

        return table.render()

    def _inference_table(self) -> list[str]:
        table = MarkdownTable(["Model", "Inference run", "Figures", "Animations", "Report"])

        for r in self.records:
            inference = f"`{r.inference_dir.name}`" if r.has_inference else "pending"
            report_md = f"[report.md]({self.assets.rel(r.report_path)})" if r.report_path else "—"

            table.add_row(f"`{r.name}`", inference, len(r.figures), len(r.animations), report_md)

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
        grouped_intro = (
            "Correlated headline metrics are averaged within a group first, so the curve-error metrics do not outvote "
            "the independent axes. `Grouped score` averages the group scores with equal weight; each cell is the group "
            "score with its group rank in parentheses (best in **bold**)."
        )

        lines  = self._rank_section("Model",    "Leaderboard",         headline_intro, self.HEADLINE_METRICS, scored)
        lines += self._grouped_section("Model", "Grouped Leaderboard", grouped_intro,  self.HEADLINE_GROUPS,  scored)
        return lines

    def _write_metrics(self) -> Path:
        scored = [r for r in self.records if r.metrics]

        if self.rank_models:
            lines  = self.assets.header("Test Metrics Comparison")
            lines += ["> Best value per metric in **bold** (↓ lower is better, ↑ higher is better). Per-bin array metrics are excluded.\n"]
            if self.has_seed_sweep:
                lines += ["> Each model was trained under multiple seeds; cells show the seed mean ± seed standard deviation.\n"]
        else:
            lines  = self.assets.header("Per-Fold Test Metrics")
            lines += ["> Folds of one model across disjoint azimuth regions; no value is highlighted as best (↓ lower is better, ↑ higher is better). Per-bin array metrics are excluded.\n"]

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
                cell = self._seed_annotated(cell, self._metric_seed_std(r.name, key))
                cells.append(cell)

            table.add_row(f"`{key}`{arrow}", *cells)

        return [*table.render(), ""]

    def _write_media(self, groups: list[tuple[str, re.Pattern]] | None, subdir: str, title: str) -> list[Path]:
        scored = [r for r in self.records if r.has_inference]

        if groups is None:
            return [self._write_media_file(self._media_names(scored, subdir), subdir, title, "gif_comparison.md")]

        names   : set[str]   = {name for r in scored for name in self._record_media(r, subdir)}
        claimed : set[str]   = set()
        written : list[Path] = []

        for group_title, pattern in groups + [("Other figures", re.compile(r".*"))]:
            group_names = sorted((n for n in names if n not in claimed and pattern.search(n)), key=ReportAssets.natural_key)
            if not group_names:
                continue
            claimed.update(group_names)

            slug = re.sub(r"[^a-z0-9]+", "_", group_title.lower()).strip("_")
            out  = self._write_media_file(group_names, subdir, f"{title} – {group_title}", f"figures_{slug}.md")
            written.append(out)

        return written

    def _media_names(self, scored: list[TrialRecord], subdir: str) -> list[str]:
        names = {name for r in scored for name in self._record_media(r, subdir)}
        return sorted(names, key=ReportAssets.natural_key)

    def _record_media(self, record: TrialRecord, subdir: str) -> list[str]:
        media = record.figures if subdir == "figures" else record.animations
        return [path.name for path in media]

    def _write_media_file(self, names: list[str], subdir: str, title: str, filename: str) -> Path:
        scored = [r for r in self.records if r.has_inference]

        lines = self.assets.header(title)
        lines += ["> Only trials with at least one completed inference run are shown.\n"]

        if not names:
            lines.append("_No media available yet._\n")

        for media_name in names:
            lines.append(f"## `{media_name}`\n")
            for r in scored:
                media_path = r.inference_dir / subdir / media_name
                if media_path.exists():
                    lines.append(f"*{r.name}*  \n![]({self.assets.src(media_path)})\n")
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
                "size_match"      : r.size_match,
                "training_result" : r.training_result,
                "checkpoint"      : r.checkpoint,
                "run_summary"     : r.run_summary,
                "trainer_config"  : r.trainer_config,
                "inference_dir"   : str(r.inference_dir) if r.inference_dir else None,
                "metrics"         : r.metrics,
                "figures"         : [path.name for path in r.figures],
                "animations"      : [path.name for path in r.animations],
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
        written.extend(self._write_media(self.FIGURE_GROUPS, "figures", "Figures"))
        written.extend(self._write_media(None, "animations", "Animation Comparison"))
        written.append(self._write_summary_json())

        return written
