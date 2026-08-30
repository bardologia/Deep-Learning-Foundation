"""Per-layer activation diagnostics of a trained backbone on real batches.

Records activation statistics through forward hooks, flags dead channels, saturation
and collapsed dynamic range, and writes depth profiles, histograms and a markdown
report into the run's analysis directory.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from tools.data.io                                   import FileIO
from tools.diagnostics.activation_recorder           import ActivationRecorder, LayerActivationStats
from tools.diagnostics.activation_xray_analysis      import ActivationIssueDetector, ActivationXraySummarizer
from tools.diagnostics.activation_xray_plots         import ActivationXrayPlots
from tools.diagnostics.weight_xray_analysis          import SEVERITY_RANK
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class ActivationXrayRun(AnalysisRun):
    """Runs the activation x-ray on a single training run directory."""

    SUMMARY_FILENAME = "activation_xray.json"
    REPORT_FILENAME  = "activation_xray.md"

    def _record(self, run) -> dict[str, dict]:
        """Collects per-layer activation statistics over the configured number of batches.

        Args:
            run : Loaded run bundling the model, the dataloader and the dataset.

        Returns:
            Per-layer statistics keyed by module name, as returned by the recorder.
        """
        recorder = ActivationRecorder(run.model.module)
        recorder.attach_stats()

        with torch.no_grad():
            for index, batch in enumerate(run.loader):
                if index >= self.config.max_batches:
                    break
                run.model(batch[0])

        recorder.detach()
        return recorder.stats()

    def _render_figures(self, ordered_stats: list[dict], flagged: list) -> dict[str, Path]:
        """Renders the depth profiles and the histograms of the most severely flagged layers.

        Args:
            ordered_stats : Per-layer statistics in forward-call order.
            flagged       : Layer reports whose severity is not "ok".

        Returns:
            Figure paths keyed by figure name.
        """
        plots    = ActivationXrayPlots(self.config)
        severity = {report.name: report.severity for report in flagged}

        figures = {
            "zero_frac_by_depth"          : plots.zero_frac_by_depth(ordered_stats, severity, self.output_dir / "plots" / "zero_frac_by_depth.png"),
            "std_by_depth"                : plots.std_by_depth(ordered_stats, severity, self.output_dir / "plots" / "std_by_depth.png"),
            "dead_channels_by_depth"      : plots.dead_channels_by_depth(ordered_stats, severity, self.output_dir / "plots" / "dead_channels_by_depth.png"),
            "max_abs_by_depth"            : plots.max_abs_by_depth(ordered_stats, severity, self.output_dir / "plots" / "max_abs_by_depth.png"),
            "effective_channels_by_depth" : plots.effective_channels_by_depth(ordered_stats, severity, self.output_dir / "plots" / "effective_channels_by_depth.png"),
            "dynamic_range_by_depth"      : plots.dynamic_range_by_depth(ordered_stats, severity, self.output_dir / "plots" / "dynamic_range_by_depth.png"),
        }

        by_severity = sorted(flagged, key=lambda report: -SEVERITY_RANK[report.severity])

        for report in by_severity[: self.config.max_layer_histograms]:
            safe_name = report.name.replace(".", "_")
            figures[f"hist_{safe_name}"] = plots.layer_histogram(report.stats, LayerActivationStats.HIST_EDGES, self.output_dir / "plots" / "histograms" / f"{safe_name}.png")

        return figures

    def _write_report(self, run, reports: list, summary: dict, figures: dict[str, Path]) -> Path:
        """Writes the markdown report of the x-ray.

        Args:
            run     : Loaded run, used for the backbone name in the title.
            reports : Per-layer reports in forward-call order.
            summary : Verdict, counts and worst layers from the summarizer.
            figures : Figure paths keyed by figure name.

        Returns:
            Path of the written report.
        """
        doc = MarkdownDoc(title=f"Activation x-ray: {run.backbone_name}")
        doc.paragraph(
            f"Per-layer activation statistics on {summary['batches']} real '{self.config.split}' batches. Verdict: **{summary['verdict']}**. "
            "Depth profiles are ordered by first forward call and coloured by module family; flagged layers carry a severity ring. "
            "Channel utilisation is the entropy-effective number of channels (by mean |activation| share) over the channel count: "
            "1.0 means all channels carry equal mass, low values mean the layer routes its signal through a few channels. "
            "Dynamic range is the p99/p50 ratio of |activation|; large values indicate heavy-tailed activations."
        )

        doc.kv_table({
            "Layers"         : summary["layers"],
            "Flagged layers" : summary["flagged_layers"],
            "Issues"         : summary["issues"],
            "Critical"       : summary["severity_counts"]["critical"],
            "Warnings"       : summary["severity_counts"]["warning"],
            "Worst layers"   : ", ".join(summary["worst_layers"]) if summary["worst_layers"] else "none",
        })

        issue_table = MarkdownTable(("Layer", "Severity", "Code", "Message"))
        for report in reports:
            for issue in report.issues:
                issue_table.add_row(f"`{report.name}`", issue.severity, f"`{issue.code}`", issue.message)
        if any(report.issues for report in reports):
            doc.heading("Issues", level=2)
            doc.table(issue_table)

        doc.heading("Layer statistics", level=2)
        stats_table = MarkdownTable(("#", "Layer", "Type", "zero%", "dead ch", "eff ch%", "std", "p99|a|", "max|a|", "p99/p50"))
        for index, report in enumerate(reports):
            s = report.stats

            dead    = f"{s['dead_channels']}/{s['n_channels']}" if s["n_channels"] is not None else "—"
            eff     = f"{s['effective_channel_frac'] * 100.0:.0f}" if s["effective_channel_frac"] is not None else "—"
            dynamic = f"{s['dynamic_range']:.3g}" if s["dynamic_range"] is not None else "—"

            stats_table.add_row(str(index), f"`{s['name']}`", s["module_type"], f"{s['zero_frac'] * 100.0:.1f}", dead, eff, f"{s['std']:.3g}", f"{s['abs_p99']:.3g}", f"{s['max_abs']:.3g}", dynamic)
        doc.table(stats_table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        """Records, analyses, plots and reports the activations of this run.

        Returns:
            Payload with the "summary" verdict block, the ordered per-layer "layers"
            statistics and the flat list of detected "issues", as written to JSON.
        """
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run       = self._load_run()
        all_stats = self._record(run)

        detector = ActivationIssueDetector(self.config)
        reports  = detector.run(all_stats)
        reports.sort(key=lambda report: report.stats["first_seen"])
        summary  = ActivationXraySummarizer().build(reports, self.run_dir, min(self.config.max_batches, len(run.loader)))

        ordered_stats = [report.stats for report in reports]
        flagged       = [report for report in reports if report.severity != "ok"]

        figures = self._render_figures(ordered_stats, flagged) if self.config.make_plots else {}

        payload = {"summary": summary, "layers": ordered_stats, "issues": [{"layer": r.name, "severity": i.severity, "code": i.code, "message": i.message} for r in reports for i in r.issues]}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, reports, summary, figures)

        self.logger.ok(f"{self.run_dir.name}: {summary['verdict']} ({summary['flagged_layers']}/{summary['layers']} layers flagged) -> {report_path}")

        return payload


class ActivationXrayBatch(RunBatch):
    """Runs the activation x-ray over every run matching the configured filter."""

    SELECTOR_ACTION = "x-ray"
    SECTION_TITLE   = "Activation x-ray"
    RUN_CLASS       = ActivationXrayRun
