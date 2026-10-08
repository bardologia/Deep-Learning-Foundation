"""Console, markdown and JSON reporting for the model weight x-ray.

Prints the summary and flagged tensors to the console, assembles the markdown
report with the findings table, embedded diagnostic plots and the per-tensor
metric table, and serialises the full report payload to JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib     import Path

from tools.reporting.markdown  import MarkdownDoc, MarkdownTable
from tools.reporting.reporting import ReportAssets

from configuration.diagnostics import WeightXrayConfig


class WeightXrayReport:
    """Writes the console, markdown and JSON outputs of a weight x-ray.

    Attributes:
        config : Weight x-ray configuration holding the output paths.
        assets : Helper that links or embeds the diagnostic plot images.
    """

    def __init__(self, config: WeightXrayConfig) -> None:
        """Initialises the reporter.

        Args:
            config : Weight x-ray configuration for the checkpoint.
        """
        self.config = config
        self.assets = ReportAssets(config.report_directory, embed_images=config.embed_images)

    def _log_console(self, logger, reports: list, summary: dict, provenance: dict) -> None:
        """Prints the summary, severity counts, issue codes and flagged tensors to the console.

        Args:
            logger     : Logger providing the table renderers.
            reports    : Per-tensor reports carrying their issues.
            summary    : Checkpoint-level summary from XraySummarizer.
            provenance : Report provenance from WeightXray, naming the checkpoint entry and epochs.
        """
        logger.kv_table({
            "Checkpoint"      : summary["checkpoint"],
            "Entry"           : provenance["checkpoint"]["key"],
            "Epoch (stored)"  : provenance["checkpoint"]["epoch"],
            "Tensors"         : summary["tensors"],
            "Parameters"      : f"{summary['parameters']:,}",
            "Flagged tensors" : summary["flagged_tensors"],
            "Total issues"    : summary["issues"],
            "Unchecked"       : self._unchecked_text(summary),
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
        """Returns the markdown table of every finding, ordered by descending severity."""
        ordering = {"critical": 0, "warning": 1, "info": 2}
        flagged  = sorted((report for report in reports if report.severity != "ok"), key=lambda report: ordering.get(report.severity, 3))

        table = MarkdownTable(["Tensor", "Role", "Shape", "Severity", "Finding", "Detail"], align=["left", "left", "left", "left", "left", "left"])
        for report in flagged:
            for issue in report.issues:
                table.add_row(report.name, report.role, "x".join(str(dim) for dim in report.shape), issue.severity, issue.code, issue.message)

        return table

    @staticmethod
    def _checked_cell(report, check: str, value, fmt: str) -> str:
        """Returns a metric cell, "skipped" when the check was size-limited and "-" when the tensor has no matrix statistics."""
        if check in report.skipped_checks:
            return "skipped"
        if value is None:
            return "-"
        return format(value, fmt)

    def _metric_table(self, reports: list) -> MarkdownTable:
        """Returns the markdown table of per-tensor metrics in state-dict order."""
        table = MarkdownTable(["Tensor", "Role", "Params", "Std", "Dead %", "Rank ratio", "Spectral", "Dup. pairs", "Severity"], align=["left", "left", "right", "right", "right", "right", "right", "right", "left"])
        for report in reports:
            table.add_row(
                report.name,
                report.role,
                f"{report.count:,}",
                f"{report.std:.3g}",
                f"{100.0 * report.frac_dead:.1f}",
                self._checked_cell(report, "spectral",   report.rank_ratio,      ".2f"),
                self._checked_cell(report, "spectral",   report.spectral_norm,   ".3g"),
                self._checked_cell(report, "duplicates", report.duplicate_units, "d"),
                report.severity,
            )
        return table

    def _provenance_rows(self, provenance: dict) -> dict:
        """Returns the provenance entries rendered in the report header, from the checkpoint entry to the library versions."""
        checkpoint = provenance["checkpoint"]

        architecture = provenance["architecture"]
        unit_axis    = f"{architecture} (transposed convolutions read on dim 1)" if architecture is not None else "unavailable (every weight read on dim 0)"

        return {
            "Checkpoint entry"             : checkpoint["key"],
            "Epoch (stored, 0-based)"      : checkpoint["epoch"],
            "Best epoch (stored, 0-based)" : checkpoint["best_epoch"],
            "Module types from"            : unit_axis,
            "Created"                      : provenance["created"],
            "Git commit"                   : provenance["git_commit"],
            "Host"                         : provenance["host"],
            "Python"                       : provenance["python"],
            "Torch"                        : provenance["torch"],
        }

    def _threshold_rows(self, provenance: dict) -> dict:
        """Returns the decision thresholds and size limits behind every flag, from the effective configuration."""
        config = provenance["config"]
        rows   = {key.split(".", 1)[1]: value for key, value in config.items() if key.startswith("thresholds.")}

        rows["svd_max_dim"]         = config["svd_max_dim"]
        rows["duplicate_max_units"] = config["duplicate_max_units"]
        return rows

    def _build_markdown(self, reports: list, summary: dict, provenance: dict, plot_paths: list) -> Path:
        """Assembles and saves the markdown report.

        Args:
            reports    : Per-tensor reports carrying statistics and issues.
            summary    : Checkpoint-level summary from XraySummarizer.
            provenance : Report provenance from WeightXray.
            plot_paths : Diagnostic figure paths to embed.

        Returns:
            Path of the saved markdown file.
        """
        doc = MarkdownDoc(title="Model weight x-ray")

        doc.bold_kv("Checkpoint", summary["checkpoint"])
        doc.bold_kv("Verdict", summary["verdict"])
        doc.blank()

        doc.heading("Provenance", level=2)
        doc.kv_table(self._provenance_rows(provenance), code_keys=False)

        doc.heading("Thresholds and limits", level=2)
        doc.kv_table(self._threshold_rows(provenance), header=("Setting", "Value"))

        doc.heading("Summary", level=2)
        doc.kv_table({
            "Tensors"         : summary["tensors"],
            "Parameters"      : f"{summary['parameters']:,}",
            "Flagged tensors" : summary["flagged_tensors"],
            "Total issues"    : summary["issues"],
            "Critical"        : summary["severity_counts"]["critical"],
            "Warning"         : summary["severity_counts"]["warning"],
            "Info"            : summary["severity_counts"]["info"],
            "Unchecked"       : self._unchecked_text(summary),
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

    @staticmethod
    def _unchecked_text(summary: dict) -> str:
        """Returns the tensors each size-limited check skipped as one phrase, e.g. "spectral 0, duplicates 3"."""
        return ", ".join(f"{check} {count}" for check, count in summary["unchecked"].items())

    def _write_json(self, reports: list, summary: dict, provenance: dict) -> Path:
        """Serialises the summary, the provenance and every tensor report to JSON and returns the path."""
        payload = {
            "summary"    : summary,
            "provenance" : provenance,
            "tensors"    : [{**asdict(report), "severity": report.severity} for report in reports],
        }

        path = self.config.report_json_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def write(self, logger, reports: list, summary: dict, provenance: dict, plot_paths: list) -> dict:
        """Emits the console, markdown and JSON reports.

        Args:
            logger     : Logger providing the table renderers.
            reports    : Per-tensor reports carrying statistics and issues.
            summary    : Checkpoint-level summary from XraySummarizer.
            provenance : Report provenance from WeightXray: creation time, checkpoint entry and epochs, configuration, code and host.
            plot_paths : Diagnostic figure paths to embed in the markdown.

        Returns:
            Mapping of "report" and "json" to the written file paths.
        """
        self._log_console(logger, reports, summary, provenance)

        markdown_path = self._build_markdown(reports, summary, provenance, plot_paths)
        json_path     = self._write_json(reports, summary, provenance)

        outputs = {"report": markdown_path, "json": json_path}
        logger.kv_table({name: str(path) for name, path in outputs.items()}, title="Outputs")
        return outputs
