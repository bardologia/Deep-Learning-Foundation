from __future__ import annotations

import math

import numpy as np
import torch


class GradientClipper:
    EXPLODING_NORM_SENTINEL = 100.0

    def __init__(self, config, logger, tracker, param_groups=None):
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
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]

        if not grads:
            return 0.0

        per_param_norms = torch._foreach_norm(grads, 2)
        total_norm      = torch.norm(torch.stack(per_param_norms), 2)

        return total_norm.item()

    def _log_group_norms(self, global_step: int) -> None:
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
        scale = min(1.0, max_norm / (norm + self.epsilon))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(scale)

        norm_after = norm * scale
        return norm, norm_after

    def _compute_adaptive_threshold(self) -> float | None:
        if len(self.history) < self.window:
            return None

        window_data = np.asarray(self.history[-self.window:], dtype=np.float32)

        if self.mode == "adaptive_percentile":
            return float(np.percentile(window_data, self.percentile))

        else:
            return float(window_data.mean() + self.mean_std_k * window_data.std())

    def check_gradients(self, model: torch.nn.Module, global_step: int) -> bool:
        has_invalid = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning(f"NaN/Inf gradient detected in {name} at step {global_step}!")
                    has_invalid = True
                    break

        return has_invalid

    def maybe_clip(self, model: torch.nn.Module, global_step: int):
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
        self.history.append(float(grad_norm_value))

        if len(self.history) % self.hist_freq == 0:
            self.tracker.log_histogram("optim/grad_norm_hist", np.asarray(self.history[-self.hist_freq:], dtype=np.float32), global_step)

    def state_dict(self) -> dict:
        keep = max(self.window, self.hist_freq)
        return {"history": [float(v) for v in self.history[-keep:]]}

    def load_state_dict(self, state: dict) -> None:
        history = [float(v) for v in state["history"]]

        if not all(math.isfinite(value) for value in history):
            raise FloatingPointError("The saved gradient-norm history holds non-finite values; that trainer state predates the non-finite gradient guard and would disable adaptive clipping for the rest of the run.")

        self.history = history
