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

    SUMMARY_FILENAME = "activation_xray.json"
    REPORT_FILENAME  = "activation_xray.md"

    def _record(self, run) -> dict[str, dict]:
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
        plots   = ActivationXrayPlots()
        figures = {
            "zero_frac_by_depth"     : plots.zero_frac_by_depth(ordered_stats, self.output_dir / "plots" / "zero_frac_by_depth.png"),
            "std_by_depth"           : plots.std_by_depth(ordered_stats, self.output_dir / "plots" / "std_by_depth.png"),
            "dead_channels_by_depth" : plots.dead_channels_by_depth(ordered_stats, self.output_dir / "plots" / "dead_channels_by_depth.png"),
            "max_abs_by_depth"       : plots.max_abs_by_depth(ordered_stats, self.output_dir / "plots" / "max_abs_by_depth.png"),
        }

        by_severity = sorted(flagged, key=lambda report: -SEVERITY_RANK[report.severity])

        for report in by_severity[: self.config.max_layer_histograms]:
            safe_name = report.name.replace(".", "_")
            figures[f"hist_{safe_name}"] = plots.layer_histogram(report.stats, LayerActivationStats.HIST_EDGES, self.output_dir / "plots" / "histograms" / f"{safe_name}.png")

        return figures

    def _write_report(self, run, reports: list, summary: dict, figures: dict[str, Path]) -> Path:
        doc = MarkdownDoc(title=f"Activation x-ray: {run.backbone_name}")
        doc.paragraph(f"Per-layer activation statistics on {summary['batches']} real '{self.config.split}' batches. Verdict: **{summary['verdict']}**.")

        doc.kv_table({
            "Layers"         : summary["layers"],
            "Flagged layers" : summary["flagged_layers"],
            "Issues"         : summary["issues"],
            "Critical"       : summary["severity_counts"]["critical"],
            "Warnings"       : summary["severity_counts"]["warning"],
        })

        issue_table = MarkdownTable(("Layer", "Severity", "Code", "Message"))
        for report in reports:
            for issue in report.issues:
                issue_table.add_row(f"`{report.name}`", issue.severity, f"`{issue.code}`", issue.message)
        if any(report.issues for report in reports):
            doc.heading("Issues", level=2)
            doc.table(issue_table)

        doc.heading("Layer statistics", level=2)
        stats_table = MarkdownTable(("#", "Layer", "Type", "zero%", "dead ch", "std", "max|a|"))
        for index, report in enumerate(reports):
            s = report.stats
            dead = f"{s['dead_channels']}/{s['n_channels']}" if s["n_channels"] is not None else "—"
            stats_table.add_row(str(index), f"`{s['name']}`", s["module_type"], f"{s['zero_frac'] * 100.0:.1f}", dead, f"{s['std']:.3g}", f"{s['max_abs']:.3g}")
        doc.table(stats_table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
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

    SELECTOR_ACTION = "x-ray"
    SECTION_TITLE   = "Activation x-ray"
    RUN_CLASS       = ActivationXrayRun
