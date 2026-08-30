"""Global gradient-norm monitoring and fixed or adaptive gradient clipping."""

from __future__ import annotations

import math

import numpy as np
import torch


class GradientClipper:
    """Measures the global gradient norm each step and rescales it against a threshold.

    The threshold is either fixed or derived from a rolling window of recorded
    norms, as a percentile or a mean plus k standard deviations. Adaptive modes do
    not clip until the window is full.

    Attributes:
        logger                  : Logger for warnings and mode reporting.
        tracker                 : Experiment tracker receiving norm scalars and histograms.
        param_groups            : Optimizer parameter groups whose norms are logged separately, or None.
        mode                    : One of "disabled", "fixed", "adaptive_percentile", "adaptive_mean_std".
        threshold               : Fixed clipping threshold, or None in adaptive modes.
        window                  : Number of recorded steps in the adaptive window.
        percentile              : Percentile of the window used as adaptive threshold.
        mean_std_k              : Standard-deviation multiplier of the mean+k*std threshold.
        epsilon                 : Denominator guard used when computing the rescaling factor.
        hist_freq               : Number of recorded steps between gradient-norm histograms.
        history                 : Recorded gradient norms, most recent last.
        n_recorded              : Total number of recorded gradient norms.
        EXPLODING_NORM_SENTINEL : Norm above which an alert is logged, independent of the clipping threshold.
    """

    EXPLODING_NORM_SENTINEL = 100.0

    def __init__(self, config, logger, tracker, param_groups=None):
        """Reads the clipper settings from the config and reports them.

        Args:
            config       : Training configuration exposing a gradient_clipper block.
            logger       : Logger for the configuration table and warnings.
            tracker      : Experiment tracker for scalars and histograms.
            param_groups : Optimizer parameter groups for per-group norm logging, or None to log only the global norm.
        """
        self.logger       = logger
        self.tracker      = tracker
        self.param_groups = param_groups

        self.mode       = config.gradient_clipper.clip_mode
        self.threshold  = config.gradient_clipper.max_grad_norm if self.mode == "fixed" else None
        self.window     = config.gradient_clipper.adaptive_window
        self.percentile = config.gradient_clipper.adaptive_percentile
        self.mean_std_k = config.gradient_clipper.adaptive_mean_std_k
        self.epsilon    = config.gradient_clipper.clip_epsilon
        self.hist_freq  = config.gradient_clipper.log_histogram_freq

        self.history: list[float] = []
        self.n_recorded           = 0

        self._adaptive_active_logged = False

        fields = {"Mode": self.mode}

        if self.mode == "fixed":
            fields["Threshold"] = self.threshold

        elif self.mode == "adaptive_percentile":
            fields["Window"]     = self.window
            fields["Percentile"] = self.percentile
            fields["Warm-up"]    = f"no clipping until {self.window} optimizer steps are recorded"

        elif self.mode == "adaptive_mean_std":
            fields["Window"]       = self.window
            fields["Mean+k*Std k"] = self.mean_std_k
            fields["Warm-up"]      = f"no clipping until {self.window} optimizer steps are recorded"

        fields["Histogram every"] = f"{self.hist_freq} optimizer steps"
        fields["NaN grad scan"]   = "on (log_debug)" if self.tracker.debug else "off (enable with training.log_debug)"

        self.logger.section("[Gradient Clipper]")
        self.logger.kv_table(fields)

    @staticmethod
    def global_norm(model: torch.nn.Module) -> float:
        """Returns the L2 norm over all parameter gradients of a model, or 0.0 when none exist."""
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]

        if not grads:
            return 0.0

        per_param_norms = torch._foreach_norm(grads, 2)
        total_norm      = torch.norm(torch.stack(per_param_norms), 2)

        return total_norm.item()

    def _log_group_norms(self, global_step: int) -> None:
        """Logs the L2 gradient norm of each parameter group when several groups exist.

        Args:
            global_step : Optimizer step index the scalars are logged at.
        """
        if self.param_groups is None or len(self.param_groups) <= 1:
            return

        for i, group in enumerate(self.param_groups):
            grads = [p.grad.detach() for p in group["params"] if p.grad is not None]

            if not grads:
                continue

            group_norm = torch.norm(torch.stack(torch._foreach_norm(grads, 2)), 2).item()
            name       = group.get("name", str(i))
            self.tracker.log_scalar(f"optim/grad_norm/{name}", group_norm, global_step)

    def _clip(self, model: torch.nn.Module, norm: float, max_norm: float) -> tuple[float, float]:
        """Rescales the gradients in place so their norm does not exceed a threshold.

        Args:
            model    : Model whose gradients are scaled.
            norm     : Current global gradient norm.
            max_norm : Threshold the norm is scaled down to.

        Returns:
            Tuple of the norm before and after scaling.
        """
        scale = min(1.0, max_norm / (norm + self.epsilon))
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]

        if scale < 1.0 and grads:
            torch._foreach_mul_(grads, scale)

        norm_after = norm * scale
        return norm, norm_after

    def _compute_adaptive_threshold(self) -> float | None:
        """Returns the threshold implied by the rolling norm window, or None until it is full."""
        if len(self.history) < self.window:
            return None

        window_data = np.asarray(self.history[-self.window:], dtype=np.float32)

        if self.mode == "adaptive_percentile":
            return float(np.percentile(window_data, self.percentile))

        else:
            return float(window_data.mean() + self.mean_std_k * window_data.std())

    def check_gradients(self, model: torch.nn.Module, global_step: int) -> bool:
        """Returns whether any parameter gradient holds a NaN or Inf, warning on the first hit.

        Args:
            model       : Model whose named parameters are scanned.
            global_step : Optimizer step index quoted in the warning.

        Returns:
            True when a non-finite gradient was found.
        """
        has_invalid = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning(f"NaN/Inf gradient detected in {name} at step {global_step}!")
                    has_invalid = True
                    break

        return has_invalid

    def maybe_clip(self, model: torch.nn.Module, global_step: int):
        """Measures the gradient norm, logs it and clips it when a threshold applies.

        Args:
            model       : Model whose gradients are measured and possibly rescaled.
            global_step : Optimizer step index used for logging.

        Returns:
            The gradient norm before any clipping.

        Raises:
            FloatingPointError : If the gradient norm is non-finite, since rescaling cannot contain a NaN or Inf gradient.
        """
        has_invalid = self.check_gradients(model, global_step) if self.tracker.debug else False
        norm        = GradientClipper.global_norm(model)

        if has_invalid or not math.isfinite(norm):
            raise FloatingPointError(f"Gradient norm is non-finite ({norm}) at step {global_step}; rescaling cannot contain a NaN/Inf gradient and the next optimizer step would corrupt every parameter.")

        if norm > self.EXPLODING_NORM_SENTINEL:
            self.logger.warning(f"Gradient norm {norm:.2f} exceeded the fixed alert sentinel {self.EXPLODING_NORM_SENTINEL} at step {global_step} (alert only, independent of the clip threshold).")

        self.tracker.log_scalar("optim/grad_norm", norm, global_step)
        self._log_group_norms(global_step)

        if self.mode == "disabled":
            return norm

        if self.mode == "fixed":
            threshold = self.threshold
        else:
            threshold = self._compute_adaptive_threshold()
            if threshold is not None and not self._adaptive_active_logged:
                self._adaptive_active_logged = True
                self.logger.info(f"Adaptive gradient clipping active from step {global_step}: window of {self.window} optimizer steps filled, initial threshold {threshold:.4f}.")

        if threshold is None:
            return norm

        norm_before, norm_after = self._clip(model, norm, threshold)
        clip_ratio = norm_after / (norm_before + self.epsilon)

        self.tracker.log_scalar("optim/grad_clip_ratio", clip_ratio, global_step)

        if self.mode != "fixed":
            self.tracker.log_scalar("optim/grad_clip_threshold", threshold, global_step)

        return norm_before

    def record(self, grad_norm_value: float, global_step: int):
        """Appends a gradient norm to the adaptive window and logs a histogram periodically.

        Args:
            grad_norm_value : Gradient norm of the step just taken.
            global_step     : Optimizer step index the histogram is logged at.
        """
        self.history.append(float(grad_norm_value))
        self.n_recorded += 1

        if self.n_recorded % self.hist_freq == 0:
            self.tracker.log_histogram("optim/grad_norm_hist", np.asarray(self.history[-self.hist_freq:], dtype=np.float32), global_step)

    def state_dict(self) -> dict:
        """Returns the trailing gradient-norm history and the recorded-step count."""
        keep = max(self.window, self.hist_freq)
        return {"history": [float(v) for v in self.history[-keep:]], "n_recorded": int(self.n_recorded)}

    def load_state_dict(self, state: dict) -> None:
        """Restores the gradient-norm history after checking it is finite.

        Args:
            state : State dictionary produced by state_dict().

        Raises:
            FloatingPointError : If the saved history holds non-finite values, which would disable adaptive clipping for the rest of the run.
        """
        history = [float(v) for v in state["history"]]

        if not all(math.isfinite(value) for value in history):
            raise FloatingPointError("The saved gradient-norm history holds non-finite values; that trainer state predates the non-finite gradient guard and would disable adaptive clipping for the rest of the run.")

        self.history    = history
        self.n_recorded = int(state["n_recorded"])
