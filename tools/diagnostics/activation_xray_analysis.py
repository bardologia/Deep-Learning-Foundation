from __future__ import annotations

from tools.diagnostics.activation_recorder  import LayerActivationStats
from tools.diagnostics.weight_xray_analysis import SEVERITY_RANK, Issue


class ActivationLayerReport:

    def __init__(self, stats: dict) -> None:
        self.stats  = stats
        self.issues = []

    @property
    def name(self) -> str:
        return self.stats["name"]

    @property
    def severity(self) -> str:
        worst = max((SEVERITY_RANK[issue.severity] for issue in self.issues), default=0)
        return {0: "ok", 1: "info", 2: "warning", 3: "critical"}[worst]


class ActivationIssueDetector:

    def __init__(self, config) -> None:
        self.config = config

    def _check_finite(self, report: ActivationLayerReport) -> list[Issue]:
        frac = report.stats["nonfinite_frac"]
        if frac > 0.0:
            return [Issue("critical", "nonfinite_activations", f"{frac * 100.0:.4f}% of activations are NaN or Inf")]
        return []

    def _check_dead_layer(self, report: ActivationLayerReport) -> list[Issue]:
        zero_frac = report.stats["zero_frac"]

        if zero_frac >= self.config.dead_zero_frac_critical:
            return [Issue("critical", "dead_layer", f"{zero_frac * 100.0:.1f}% of activations are exactly zero")]
        if zero_frac >= self.config.dead_zero_frac_warn:
            return [Issue("warning", "mostly_dead_layer", f"{zero_frac * 100.0:.1f}% of activations are exactly zero")]
        return []

    def _check_dead_channels(self, report: ActivationLayerReport) -> list[Issue]:
        frac = report.stats["dead_channel_frac"]

        if frac is not None and frac >= self.config.dead_channel_frac_warn:
            return [Issue("warning", "dead_channels", f"{report.stats['dead_channels']}/{report.stats['n_channels']} channels never activate above {LayerActivationStats.DEAD_CHANNEL_ABS:.0e}")]
        return []

    def _check_explode(self, report: ActivationLayerReport) -> list[Issue]:
        if report.stats["max_abs"] >= self.config.explode_abs_threshold:
            return [Issue("warning", "exploding_activations", f"max |activation| {report.stats['max_abs']:.3g} exceeds {self.config.explode_abs_threshold:.0e}")]
        return []

    def _check_constant(self, report: ActivationLayerReport) -> list[Issue]:
        if report.stats["std"] <= self.config.constant_std_threshold and report.stats["zero_frac"] < 1.0:
            return [Issue("warning", "constant_output", f"activation std {report.stats['std']:.3g} is effectively constant")]
        return []

    def detect(self, report: ActivationLayerReport) -> None:
        report.issues += self._check_finite(report)
        report.issues += self._check_dead_layer(report)
        report.issues += self._check_dead_channels(report)
        report.issues += self._check_explode(report)
        report.issues += self._check_constant(report)

    def run(self, all_stats: dict[str, dict]) -> list[ActivationLayerReport]:
        reports = [ActivationLayerReport(stats) for stats in all_stats.values()]
        for report in reports:
            self.detect(report)
        return reports


class ActivationXraySummarizer:

    def build(self, reports: list[ActivationLayerReport], run_dir, n_batches: int) -> dict:
        issues  = [issue for report in reports for issue in report.issues]
        flagged = [report for report in reports if report.severity != "ok"]

        severity_counts = {level: 0 for level in ("critical", "warning", "info", "ok")}
        for report in reports:
            severity_counts[report.severity] += 1

        code_counts: dict = {}
        for issue in issues:
            code_counts[issue.code] = code_counts.get(issue.code, 0) + 1

        return {
            "run_directory"   : str(run_dir),
            "layers"          : len(reports),
            "batches"         : n_batches,
            "flagged_layers"  : len(flagged),
            "issues"          : len(issues),
            "severity_counts" : severity_counts,
            "issue_codes"     : dict(sorted(code_counts.items(), key=lambda item: item[1], reverse=True)),
            "verdict"         : self._verdict(severity_counts),
        }

    def _verdict(self, severity_counts: dict) -> str:
        if severity_counts["critical"] > 0:
            return "critical issues detected"
        if severity_counts["warning"] > 0:
            return "warnings detected"
        if severity_counts["info"] > 0:
            return "minor observations only"
        return "clean"
