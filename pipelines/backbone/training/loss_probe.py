"""Pre-training probe measuring the raw magnitude of each loss term.

Runs a few batches with every selected term weighted at 1.0, then reports the
scale factors that would bring the terms onto a common magnitude.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import defaultdict

import torch

from pipelines.backbone.training.loss       import Loss
from pipelines.backbone.training.loss_terms import LOSS_TERMS
from tools.monitoring.logger                import NullLogger
from tools.monitoring.tracker               import NullTracker


class LossScaleProbe:
    """Measures each loss term at unit weight and suggests balancing weights.

    Attributes:
        probe_cfg    : Probe configuration (batch count, reference term, enabled flags, exit behaviour).
        gaussian_cfg : Gaussian head configuration.
        norm_stats   : Normalizer for the output parameter space.
        logger       : Logger receiving the probe report.
        loss_cfg     : Copy of the loss stage with every probed term enabled at weight 1.0.
    """

    SHAPING_NOTE = "Raw values are shaped by the initial-stage loss config (matching, active normalization, presence balance, amp_zero_thr)."

    def __init__(self, probe_cfg, loss_cfg, gaussian_cfg, norm_stats=None, logger=None):
        """Builds a loss configuration with the probed terms enabled at unit weight.

        Args:
            probe_cfg    : Probe configuration; its `enabled_losses` overrides the default on/off state per term.
            loss_cfg     : Loss stage the probe copies its remaining settings from.
            gaussian_cfg : Gaussian head configuration.
            norm_stats   : Normalizer for the output parameter space.
            logger       : Logger receiving the probe report.

        Raises:
            ValueError : If `enabled_losses` names a flag no loss term declares.
        """
        self.probe_cfg    = probe_cfg
        self.gaussian_cfg = gaussian_cfg
        self.norm_stats   = norm_stats
        self.logger       = logger
        self.loss_cfg     = copy.deepcopy(loss_cfg)

        known   = {term.use_flag for term in LOSS_TERMS}
        unknown = sorted(set(probe_cfg.enabled_losses) - known)
        if unknown:
            raise ValueError(f"Unknown probe enabled_losses keys {unknown}; valid flags: {sorted(known)}")

        for term in LOSS_TERMS:
            override = probe_cfg.enabled_losses.get(term.use_flag)
            setattr(self.loss_cfg, term.use_flag, override if override is not None else True)

        for term in LOSS_TERMS:
            if getattr(self.loss_cfg, term.use_flag):
                setattr(self.loss_cfg, term.weight_key, 1.0)

    def _build_criterion(self, z_axis: torch.Tensor):
        """Builds a silent Loss over the probe configuration.

        Args:
            z_axis : Z sample positions in metres, of shape (profile_length,).

        Returns:
            A Loss instance logging to null sinks so the probe output stays clean.
        """
        return Loss(
            z_axis       = z_axis,
            logger       = NullLogger(),
            tracker      = NullTracker(),
            gaussian_cfg = self.gaussian_cfg,
            loss_cfg     = self.loss_cfg,
            norm_stats   = self.norm_stats,
        )

    def _collect(self, criterion, train_loader, model, device) -> dict[str, list[float]]:
        """Records the per-batch value of every loss component without gradients.

        Args:
            criterion    : The unit-weight loss to evaluate.
            train_loader : Loader yielding (inputs, targets) batches.
            model        : Model under evaluation; left in train mode afterwards.
            device       : Device the batches are moved to.

        Returns:
            Mapping of component name to the list of per-batch values.
        """
        model.eval()
        accum: dict[str, list[float]] = defaultdict(list)

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= self.probe_cfg.n_batches:
                    break

                inputs  = batch[0].to(device)
                targets = batch[1].to(device)
                result  = self._criterion_result(criterion, model(inputs), targets)

                for name, val in result["components"].items():
                    if "/" not in name:
                        accum[name].append(float(val))

        model.train()

        return accum

    @staticmethod
    def _criterion_result(criterion, prediction, targets) -> dict:
        """Evaluates the criterion on one probe batch.

        Args:
            criterion  : The unit-weight loss under measurement.
            prediction : Model output for the batch.
            targets    : Normalized ground-truth parameter batch.

        Returns:
            The criterion's loss dict.
        """
        return criterion(prediction, targets)

    def _suggest(self, accum: dict[str, list[float]]) -> dict:
        """Derives balancing weights from the outlier-filtered mean of each term.

        Weights are the reference term's mean divided by the term's own mean, or
        the reciprocal of the mean when no reference term is configured.

        Args:
            accum : Per-batch values of each loss component.

        Returns:
            Dict holding the raw and filtered values, their means, the reference
            term and its mean, and the suggested weight per term (NaN where the
            raw mean is not positive).
        """
        filtered = {k: self._iqr_filter(v) for k, v in accum.items()}
        means    = {k: sum(v) / len(v) for k, v in filtered.items()}

        ref     = self.probe_cfg.reference
        ref_val = means.get(ref) if ref is not None else None

        suggested = {
            name: ((ref_val / raw) if ref_val is not None else (1.0 / raw)) if raw > 0 else float("nan")
            for name, raw in means.items()
        }

        return {
            "accum"     : accum,
            "filtered"  : filtered,
            "means"     : means,
            "ref"       : ref,
            "ref_val"   : ref_val,
            "suggested" : suggested,
        }

    @staticmethod
    def _iqr_filter(values: list[float], k: float = 3.0) -> list[float]:
        """Drops values further than `k` interquartile ranges from the quartiles.

        Args:
            values : Measured values of one loss term.
            k      : Interquartile-range multiplier bounding the kept interval.

        Returns:
            The filtered values, or the originals when fewer than four values
            were given or fewer than three would survive.
        """
        if len(values) < 4:
            return values
        sorted_v = sorted(values)
        n        = len(sorted_v)
        q1       = sorted_v[n // 4]
        q3       = sorted_v[(3 * n) // 4]
        iqr      = q3 - q1
        lo, hi   = q1 - k * iqr, q3 + k * iqr
        filtered = [v for v in values if lo <= v <= hi]
        return filtered if len(filtered) >= 3 else values

    def _emit_report(self, report: dict) -> None:
        """Logs the per-term raw values, kept-sample counts and suggested weights.

        Args:
            report : The dict produced by `_suggest`.
        """
        accum     = report["accum"]
        filtered  = report["filtered"]
        means     = report["means"]
        ref       = report["ref"]
        ref_val   = report["ref_val"]
        suggested = report["suggested"]

        skipped = [name for name, weight in suggested.items() if not math.isfinite(weight)]

        for name in sorted(skipped):
            self.logger.warning(f"Probe term '{name}' has raw mean <= 0 (raw={means[name]:.6f}); no usable suggested weight.")

        rows = [
            {
                "Term"             : name,
                "Raw value"        : f"{means[name]:.6f}",
                "Kept"             : f"{len(filtered[name])}/{len(accum[name])}",
                "Suggested weight" : self._weight_cell(name, suggested[name], ref),
            }
            for name in sorted(means)
        ]
        self.logger.metrics_table(rows, columns=["Term", "Raw value", "Kept", "Suggested weight"], title="Probe Results")

        formula = f"raw({ref}) / raw(i)" if ref_val is not None else "1 / raw(i)"
        self.logger.subsection(f"Formula : suggested_weight(i) = {formula}")
        self.logger.subsection("Fold these scale factors into the relevant weight_* terms to bring them to a comparable magnitude.")

        if ref is not None and ref not in means:
            self.logger.warning(f"Reference term '{ref}' not found — each term normalised to 1.0 independently.")

    @staticmethod
    def _weight_cell(name: str, weight: float, ref) -> str:
        """Returns the table cell for a suggested weight, marking the reference term."""
        if not math.isfinite(weight):
            return "skipped: raw<=0"

        return f"{weight:.6f}" + ("  (reference)" if name == ref else "")

    def run(self, train_loader, model, device, z_axis: torch.Tensor) -> dict[str, float]:
        """Measures the loss terms over a few batches and reports balancing weights.

        Args:
            train_loader : Loader supplying the probe batches.
            model        : Model whose predictions the terms are evaluated on.
            device       : Device the batches are moved to.
            z_axis       : Z sample positions in metres, of shape (profile_length,).

        Returns:
            The suggested weight per term, or an empty dict when the probe is
            disabled or no component was recorded.

        Raises:
            SystemExit : If `probe_cfg.exit_after` asks the process to stop once the report is written.
        """
        if not self.probe_cfg.enabled:
            return {}

        criterion = self._build_criterion(z_axis)

        self.logger.section("[Loss Scale Probe]")
        self.logger.subsection(f"Averaging over {self.probe_cfg.n_batches} batches; all enabled term weights forced to 1.0.")
        self.logger.subsection(self.SHAPING_NOTE)

        accum = self._collect(criterion, train_loader, model, device)

        if not accum:
            self.logger.warning("No loss components recorded — check model outputs.")
            return {}

        report = self._suggest(accum)
        self._emit_report(report)

        if self.probe_cfg.exit_after:
            sys.exit(0)

        return report["suggested"]
