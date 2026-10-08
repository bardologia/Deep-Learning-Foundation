"""Anomaly detection over recorded activation statistics.

Turns the per-layer summaries produced by ActivationRecorder into severity-
tagged findings (dead layers, dead channels, exploding or constant outputs,
channel collapse) and a run-level verdict.
"""

from __future__ import annotations

from tools.diagnostics.activation_recorder  import LayerActivationStats
from tools.diagnostics.weight_xray_analysis import SEVERITY_RANK, Issue


class ActivationLayerReport:
    """Findings attached to the activation statistics of a single layer.

    Attributes:
        stats  : Finalised statistics mapping from LayerActivationStats.
        issues : Issue records accumulated by the detector.
    """

    def __init__(self, stats: dict) -> None:
        """Initialises an empty report around one layer's statistics.

        Args:
            stats : Finalised activation statistics for the layer.
        """
        self.stats  = stats
        self.issues = []

    @property
    def name(self) -> str:
        """Dotted module path of the layer."""
        return self.stats["name"]

    @property
    def severity(self) -> str:
        """Worst severity among the layer's issues, or "ok" when there are none."""
        worst = max((SEVERITY_RANK[issue.severity] for issue in self.issues), default=0)
        return {0: "ok", 1: "info", 2: "warning", 3: "critical"}[worst]


class ActivationIssueDetector:
    """Applies configured activation thresholds to produce per-layer issues.

    Attributes:
        config : Activation x-ray configuration holding the severity thresholds.
    """

    def __init__(self, config) -> None:
        """Initialises the detector.

        Args:
            config : Configuration exposing the dead, explode, constant and channel-collapse thresholds.
        """
        self.config = config

    def _check_finite(self, report: ActivationLayerReport) -> list[Issue]:
        """Returns a critical issue when any activation is NaN or Inf."""
        frac = report.stats["nonfinite_frac"]
        if frac > 0.0:
            return [Issue("critical", "nonfinite_activations", f"{frac * 100.0:.4f}% of activations are NaN or Inf")]
        return []

    def _check_dead_layer(self, report: ActivationLayerReport) -> list[Issue]:
        """Returns an issue when the exact-zero activation fraction crosses a threshold."""
        zero_frac = report.stats["zero_frac"]

        if zero_frac >= self.config.dead_zero_frac_critical:
            return [Issue("critical", "dead_layer", f"{zero_frac * 100.0:.1f}% of activations are exactly zero")]
        if zero_frac >= self.config.dead_zero_frac_warn:
            return [Issue("warning", "mostly_dead_layer", f"{zero_frac * 100.0:.1f}% of activations are exactly zero")]
        return []

    def _check_dead_channels(self, report: ActivationLayerReport) -> list[Issue]:
        """Returns a warning when too many channels never activate above the dead cutoff."""
        frac = report.stats["dead_channel_frac"]

        if frac is not None and frac >= self.config.dead_channel_frac_warn:
            return [Issue("warning", "dead_channels", f"{report.stats['dead_channels']}/{report.stats['n_channels']} channels never activate above {LayerActivationStats.DEAD_CHANNEL_ABS:.0e}")]
        return []

    def _check_explode(self, report: ActivationLayerReport) -> list[Issue]:
        """Returns a warning when the peak absolute activation exceeds the explode threshold."""
        if report.stats["max_abs"] >= self.config.explode_abs_threshold:
            return [Issue("warning", "exploding_activations", f"max |activation| {report.stats['max_abs']:.3g} exceeds {self.config.explode_abs_threshold:.0e}")]
        return []

    def _check_constant(self, report: ActivationLayerReport) -> list[Issue]:
        """Returns a warning when a non-fully-zero layer has effectively no activation spread."""
        if report.stats["std"] <= self.config.constant_std_threshold and report.stats["zero_frac"] < 1.0:
            return [Issue("warning", "constant_output", f"activation std {report.stats['std']:.3g} is effectively constant")]
        return []

    def _check_channel_collapse(self, report: ActivationLayerReport) -> list[Issue]:
        """Returns a warning when activation mass concentrates on a small share of channels."""
        frac = report.stats["effective_channel_frac"]
        n_ch = report.stats["n_channels"]

        if frac is None or n_ch is None:
            return []
        if report.stats["zero_frac"] >= self.config.dead_zero_frac_warn:
            return []

        if frac < self.config.channel_collapse_frac_warn:
            return [Issue("warning", "channel_collapse", f"activation mass concentrates on ~{frac * n_ch:.1f} of {n_ch} channels (effective fraction {frac * 100.0:.1f}%)")]
        return []

    def detect(self, report: ActivationLayerReport) -> None:
        """Appends every applicable issue to one layer report in place.

        Args:
            report : Layer report whose issues list is extended.
        """
        report.issues += self._check_finite(report)
        report.issues += self._check_dead_layer(report)
        report.issues += self._check_dead_channels(report)
        report.issues += self._check_explode(report)
        report.issues += self._check_constant(report)
        report.issues += self._check_channel_collapse(report)

    def run(self, all_stats: dict[str, dict]) -> list[ActivationLayerReport]:
        """Builds and populates a layer report for every recorded layer.

        Args:
            all_stats : Finalised activation statistics keyed by module name.

        Returns:
            One populated ActivationLayerReport per input layer.
        """
        reports = [ActivationLayerReport(stats) for stats in all_stats.values()]
        for report in reports:
            self.detect(report)
        return reports


class ActivationXraySummarizer:
    """Condenses per-layer activation reports into a run-level summary."""

    def build(self, reports: list[ActivationLayerReport], run_dir, n_batches: int, reused_modules: dict, silent_layers: dict) -> dict:
        """Builds the run-level summary of an activation x-ray.

        Args:
            reports        : Populated per-layer reports.
            run_dir        : Run directory the capture belongs to.
            n_batches      : Number of batches pushed through the model.
            reused_modules : Leaf modules that fired more than once per forward, mapped to their call count, each recorded as one report per call site.
            silent_layers  : Hooked leaf modules that never produced a tensor, mapped to their module type, which have no report.

        Returns:
            Mapping with the run directory, the recorded, hooked and batch
            counts, flagged layer and issue totals, severity and issue-code
            histograms, the five worst layer names, the reused modules, the
            silent layers, and a textual verdict.
        """
        issues  = [issue for report in reports for issue in report.issues]
        flagged = [report for report in reports if report.severity != "ok"]

        severity_counts = {level: 0 for level in ("critical", "warning", "info", "ok")}
        for report in reports:
            severity_counts[report.severity] += 1

        code_counts: dict = {}
        for issue in issues:
            code_counts[issue.code] = code_counts.get(issue.code, 0) + 1

        worst = sorted(flagged, key=lambda report: (-SEVERITY_RANK[report.severity], -report.stats["zero_frac"]))

        return {
            "run_directory"   : str(run_dir),
            "layers"          : len(reports),
            "hooked_layers"   : len(reports) + len(silent_layers),
            "batches"         : n_batches,
            "flagged_layers"  : len(flagged),
            "issues"          : len(issues),
            "severity_counts" : severity_counts,
            "issue_codes"     : dict(sorted(code_counts.items(), key=lambda item: item[1], reverse=True)),
            "worst_layers"    : [report.name for report in worst[:5]],
            "reused_modules"  : dict(reused_modules),
            "silent_layers"   : dict(silent_layers),
            "verdict"         : self._verdict(severity_counts),
        }

    def _verdict(self, severity_counts: dict) -> str:
        """Returns the verdict phrase implied by the worst populated severity level."""
        if severity_counts["critical"] > 0:
            return "critical issues detected"
        if severity_counts["warning"] > 0:
            return "warnings detected"
        if severity_counts["info"] > 0:
            return "minor observations only"
        return "clean"
