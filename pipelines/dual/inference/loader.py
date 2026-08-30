"""Run loading for dual-trunk inference.

Overrides the backbone run loader so the model is rebuilt from the dual model
configuration persisted alongside the checkpoint.
"""

from __future__ import annotations

from models.dual                                import get_dual
from pipelines.backbone.inference.loader        import RunLoader
from pipelines.shared.config.config_persistence import DualModelConfigIO


class DualRunLoader(RunLoader):
    """Loads a dual-trunk run and rebuilds its model from the persisted config."""

    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size):
        """Rebuilds the dual model and records its head.

        Args:
            backbone_name : Registry key of the dual model family.
            in_channels   : Number of input representation channels.
            out_channels  : Number of Gaussian head output channels.
            patch_size    : Patch size as (height, width) in pixels, unused because the persisted dual config already fixes it.

        Returns:
            The instantiated dual model.
        """
        model_config, _ = DualModelConfigIO.load(self.meta_directory)
        self.model_head = model_config.head

        model, _ = get_dual(backbone_name, config=model_config, in_channels=in_channels, out_channels=out_channels)

        return model
