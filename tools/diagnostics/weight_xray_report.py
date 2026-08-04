from __future__ import annotations

import json
from dataclasses import asdict
from pathlib     import Path

from tools.reporting.markdown  import MarkdownDoc, MarkdownTable
from tools.reporting.reporting import ReportAssets

from configuration.diagnostics import WeightXrayConfig


SEVERITY_STYLE = {"critical": "err", "warning": "warn", "info": "muted", "ok": "ok"}


class WeightXrayReport:
    def __init__(self, config: WeightXrayConfig) -> None:
        self.config = config
        self.assets = ReportAssets(config.report_directory, embed_images=config.embed_images)

    def _log_console(self, logger, reports: list, summary: dict) -> None:
        logger.kv_table({
            "Checkpoint"      : summary["checkpoint"],
            "Tensors"         : summary["tensors"],
            "Parameters"      : f"{summary['parameters']:,}",
            "Flagged tensors" : summary["flagged_tensors"],
            "Total issues"    : summary["issues"],
            "Verdict"         : summary["verdict"],
        }, title="Summary")

        counts = summary["severity_counts"]
        logger.kv_table({
            "Critical" : counts["critical"],
            "Warning"  : counts["warning"],
            "Info"     : counts["info"],
            "Clean"    : counts["ok"],
        }, title="Tensors by worst severity")

        if summary["issue_codes"]:
            logger.kv_table(summary["issue_codes"], title="Issues by type")

        flagged = [report for report in reports if report.severity != "ok"]
        if not flagged:
            logger.ok("No anomalies detected")
            return

        ordering = {"critical": 0, "warning": 1, "info": 2}
        flagged  = sorted(flagged, key=lambda report: ordering.get(report.severity, 3))

        rows = [{
            "Tensor"   : report.name,
            "Role"     : report.role,
            "Shape"    : "x".join(str(dim) for dim in report.shape),
            "Severity" : report.severity,
            "Findings" : "; ".join(issue.code for issue in report.issues),
        } for report in flagged]

        logger.metrics_table(rows, columns=["Tensor", "Role", "Shape", "Severity", "Findings"], title="Flagged tensors")

    def _issue_table(self, reports: list) -> MarkdownTable:
        ordering = {"critical": 0, "warning": 1, "info": 2}
        flagged  = sorted((report for report in reports if report.severity != "ok"), key=lambda report: ordering.get(report.severity, 3))

        table = MarkdownTable(["Tensor", "Role", "Shape", "Severity", "Finding", "Detail"], align=["left", "left", "left", "left", "left", "left"])
        for report in flagged:
            for issue in report.issues:
                table.add_row(report.name, report.role, "x".join(str(dim) for dim in report.shape), issue.severity, issue.code, issue.message)

        return table

    def _metric_table(self, reports: list) -> MarkdownTable:
        table = MarkdownTable(["Tensor", "Role", "Params", "Std", "Dead %", "Rank ratio", "Spectral", "Severity"], align=["left", "left", "right", "right", "right", "right", "right", "left"])
        for report in reports:
            table.add_row(
                report.name,
                report.role,
                f"{report.count:,}",
                f"{report.std:.3g}",
                f"{100.0 * report.frac_dead:.1f}",
                f"{report.rank_ratio:.2f}" if report.rank_ratio is not None else "-",
                f"{report.spectral_norm:.3g}" if report.spectral_norm is not None else "-",
                report.severity,
            )
        return table

    def _build_markdown(self, reports: list, summary: dict, plot_paths: list) -> Path:
        doc = MarkdownDoc(title="Model weight x-ray")

        doc.bold_kv("Checkpoint", summary["checkpoint"])
        doc.bold_kv("Verdict", summary["verdict"])
        doc.blank()

        doc.heading("Summary", level=2)
        doc.kv_table({
            "Tensors"         : summary["tensors"],
            "Parameters"      : f"{summary['parameters']:,}",
            "Flagged tensors" : summary["flagged_tensors"],
            "Total issues"    : summary["issues"],
            "Critical"        : summary["severity_counts"]["critical"],
            "Warning"         : summary["severity_counts"]["warning"],
            "Info"            : summary["severity_counts"]["info"],
        })

        if summary["issue_codes"]:
            doc.heading("Issues by type", level=2)
            doc.kv_table(summary["issue_codes"], header=("Finding", "Count"))

        doc.heading("Findings", level=2)
        issue_table = self._issue_table(reports)
        if issue_table.is_empty():
            doc.paragraph("No anomalies detected.")
        else:
            doc.table(issue_table)

        if plot_paths:
            doc.heading("Diagnostic plots", level=2)
            for path in plot_paths:
                doc.raw(" ".join(self.assets.image(Path(path).stem, Path(path))))

        doc.heading("Per-tensor metrics", level=2)
        doc.table(self._metric_table(reports))

        return doc.save(self.config.report_markdown_path)

    def _write_json(self, reports: list, summary: dict) -> Path:
        payload = {
            "summary" : summary,
            "tensors" : [{**asdict(report), "severity": report.severity} for report in reports],
        }

        path = self.config.report_json_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def write(self, logger, reports: list, summary: dict, plot_paths: list) -> dict:
        self._log_console(logger, reports, summary)

        markdown_path = self._build_markdown(reports, summary, plot_paths)
        json_path     = self._write_json(reports, summary)

        outputs = {"report": markdown_path, "json": json_path}
        logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")
        return outputs
