from __future__ import annotations

from tools.training import BaseTrainer


class AutoencoderTrainer(BaseTrainer):
    def __init__(self, model, model_cfg, x_axis, config, run_dir, logger):
        self.model_cfg = model_cfg
        super().__init__(model, config, run_dir, logger, x_axis)

    def _build_param_groups(self):
        return self.model_cfg.get_param_groups(self.model)
