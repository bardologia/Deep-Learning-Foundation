from __future__ import annotations

from pathlib import Path

from configuration.training import LossScaleProbeConfig


class WorkerBase:
    def __init__(self, config, run_tag: str) -> None:
        self.config  = config
        self.run_tag = run_tag
        self.run_dir = Path(config.paths.log_base_dir) / run_tag

    def _probe_config(self) -> LossScaleProbeConfig:
        return LossScaleProbeConfig(
            enabled        = False,
            n_batches      = 100,
            reference      = "param_l1",
            exit_after     = True,
            enabled_losses = {},
        )
