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

    DEFAULT_OFF = ("use_param_legacy",)

    def __init__(self, probe_cfg, loss_cfg, gaussian_cfg, geometry_cfg, norm_stats=None, logger=None):
        self.probe_cfg    = probe_cfg
        self.gaussian_cfg = gaussian_cfg
        self.geometry_cfg = geometry_cfg
        self.norm_stats   = norm_stats
        self.logger       = logger
        self.loss_cfg     = copy.deepcopy(loss_cfg)

        known   = {term.use_flag for term in LOSS_TERMS}
        unknown = sorted(set(probe_cfg.enabled_losses) - known)
        if unknown:
            raise ValueError(f"Unknown probe enabled_losses keys {unknown}; valid flags: {sorted(known)}")

        for term in LOSS_TERMS:
            override = probe_cfg.enabled_losses.get(term.use_flag)
            default  = term.use_flag not in self.DEFAULT_OFF
            setattr(self.loss_cfg, term.use_flag, override if override is not None else default)

        for term in LOSS_TERMS:
            if getattr(self.loss_cfg, term.use_flag):
                setattr(self.loss_cfg, term.weight_key, 1.0)

    def _build_criterion(self, x_axis: torch.Tensor):
        return Loss(
            x_axis       = x_axis,
            logger       = NullLogger(),
            tracker      = NullTracker(),
            gaussian_cfg = self.gaussian_cfg,
            loss_cfg     = self.loss_cfg,
            norm_stats   = self.norm_stats,
            geometry_cfg = self.geometry_cfg,
        )

    def _collect(self, criterion, train_loader, model, device) -> dict[str, list[float]]:
        model.eval()
        accum: dict[str, list[float]] = defaultdict(list)

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= self.probe_cfg.n_batches:
                    break

                inputs  = batch[0].to(device)
                targets = batch[1].to(device)
                kz_map  = batch[2].to(device) if len(batch) > 2 and batch[2] is not None else None
                result  = criterion(model(inputs), targets, kz_map)

                for name, val in result["components"].items():
                    if "/" not in name:
                        accum[name].append(float(val))

        model.train()

        return accum

    def _suggest(self, accum: dict[str, list[float]]) -> dict:
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
        if not math.isfinite(weight):
            return "skipped: raw<=0"

        return f"{weight:.6f}" + ("  (reference)" if name == ref else "")

    def run(self, train_loader, model, device, x_axis: torch.Tensor) -> dict[str, float]:
        if not self.probe_cfg.enabled:
            return {}

        criterion = self._build_criterion(x_axis)

        self.logger.section("[Loss Scale Probe]")
        self.logger.subsection(f"Averaging over {self.probe_cfg.n_batches} batches; all enabled term weights forced to 1.0 (use_param_legacy forced off unless explicitly enabled).")
        self.logger.subsection("Raw values are shaped by the initial-stage loss config (matching, active normalization, presence balance, amp_zero_thr).")

        accum = self._collect(criterion, train_loader, model, device)

        if not accum:
            self.logger.warning("No loss components recorded — check model outputs.")
            return {}

        report = self._suggest(accum)
        self._emit_report(report)

        if self.probe_cfg.exit_after:
            sys.exit(0)

        return report["suggested"]
