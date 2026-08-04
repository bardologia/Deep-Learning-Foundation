from __future__ import annotations

from models.dual                                import get_dual
from pipelines.backbone.inference.loader        import RunLoader
from pipelines.shared.config.config_persistence import DualModelConfigIO


class DualRunLoader(RunLoader):
    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size):
        model_config, _ = DualModelConfigIO.load(self.meta_directory)
        self.model_head = model_config.head

        model, _ = get_dual(backbone_name, config=model_config, in_channels=in_channels, out_channels=out_channels)

        return model
