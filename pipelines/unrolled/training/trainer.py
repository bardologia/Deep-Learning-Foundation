from __future__ import annotations

import torch

from pipelines.unrolled.synthesis     import MeasurementSynthesiser
from pipelines.unrolled.training.loss import Loss
from tools.training                   import BaseTrainer
from tools.training.aggregation       import MetricAggregator


class UnrolledTrainer(BaseTrainer):
    stage_name    = "Unrolled"
    section_title = "[Unrolled Physics Training]"

    def __init__(self, model, model_cfg, x_axis, config, run_dir, logger, norm_stats):
        self.model_cfg    = model_cfg
        self.norm_stats   = norm_stats
        self.peak_error_m = {}

        super().__init__(model, config, run_dir, logger, x_axis)

        self.synthesiser = MeasurementSynthesiser(self.x_axis, config.params_per_gaussian, config.power_floor, config.measurement_noise_std)

    def _build_param_groups(self) -> list[dict]:
        return self.model_cfg.get_param_groups(self.model)

    def _build_criterion(self) -> Loss:
        config = self.config

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Curve loss"     : config.curve_loss,
            "Noise std"      : f"{config.measurement_noise_std} (fresh noise every epoch, also for val/test targets)",
            "Power floor"    : config.power_floor,
            "Elevation axis" : f"[{float(self.x_axis[0]):g}, {float(self.x_axis[-1]):g}] m, {int(self.x_axis.shape[0])} samples",
        })

        return Loss(config.curve_loss, self.x_axis)

    def _unpack(self, batch) -> tuple:
        if len(batch) < 3 or batch[2] is None:
            raise RuntimeError("Unrolled training requires the per-pixel kz geometry field; the dataset pipeline must run with build_geometry_field=True")

        gt_params = batch[1].to(self.device, non_blocking=True)
        kz_map    = batch[2].to(self.device, non_blocking=True).float()

        return gt_params, kz_map

    @torch.no_grad()
    def _synthesise_measurements(self, gt_params: torch.Tensor, kz_map: torch.Tensor) -> tuple:
        return self.synthesiser.synthesise(self.norm_stats.denormalize_output(gt_params.float()), kz_map)

    def _compute_loss(self, batch) -> dict:
        gt_params, kz_map          = self._unpack(batch)
        measurements, target, mask = self._synthesise_measurements(gt_params, kz_map)

        prediction = self.model(measurements, kz_map, self.x_axis)

        return self.criterion(prediction, target, mask)

    def _log_epoch_metrics(self, stage: str, aggregator: MetricAggregator, epoch: int) -> None:
        super()._log_epoch_metrics(stage, aggregator, epoch)

        physical = aggregator.reduce_physical()
        if not physical:
            return

        self.peak_error_m[stage] = physical["peak_mae_m"]
        self.logger.subsection(f"Peak MAE ({stage}) : {physical['peak_mae_m']:.3f} m")
