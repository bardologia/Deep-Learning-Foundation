"""Trainer base for the autoencoder families.

Extends the shared ``BaseTrainer`` with parameter groups taken from the model
architecture configuration, so per-module learning rates follow the same rules
as the backbone families.
"""

from __future__ import annotations

from tools.training import BaseTrainer


class AutoencoderTrainer(BaseTrainer):
    """Trainer whose optimizer parameter groups come from the model config.

    Attributes:
        model_cfg : Architecture configuration that defines the parameter groups.
    """

    def __init__(self, model, model_cfg, z_axis, config, run_dir, logger, writer=None):
        """Binds the architecture config before initialising the base trainer.

        Args:
            model     : Autoencoder being trained.
            model_cfg : Architecture configuration supplying the parameter groups.
            z_axis    : Z axis in metres of shape (profile_length,).
            config    : Trainer configuration.
            run_dir   : Directory the run writes checkpoints and logs into.
            logger    : Logger for the run.
            writer    : Optional TensorBoard writer.
        """
        self.model_cfg = model_cfg
        super().__init__(model, config, run_dir, logger, z_axis, writer)

    def _build_param_groups(self):
        """Returns the optimizer parameter groups declared by the architecture config."""
        return self.model_cfg.get_param_groups(self.model)
