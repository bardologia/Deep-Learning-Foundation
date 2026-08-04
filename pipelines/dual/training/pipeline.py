from __future__ import annotations

from configuration.architectures                import DualResUNetConfig
from models.dual                                import get_dual
from pipelines.backbone.training.pipeline       import TrainingPipeline
from pipelines.shared.config.config_persistence import DualModelConfigIO
from pipelines.shared.model.model_builder       import ModelBuilder
from pipelines.shared.training.run_naming       import RunNaming


class TrunkChannelMap:

    MAX_TRACKS = 64
    GROUPS     = ("pass", "ifg")

    @classmethod
    def resolve(cls, input_config, in_channels: int, groups: tuple) -> tuple[int, ...]:
        unknown = [group for group in groups if group not in cls.GROUPS]
        if unknown:
            raise ValueError(f"Unknown input groups {unknown}. Available: {list(cls.GROUPS)}")

        keys    = cls._channel_keys(input_config, in_channels)
        indices = tuple(index for index, key in enumerate(keys) if key.split("/")[0] in groups)

        if not indices:
            raise ValueError(f"Input groups {tuple(groups)} select no channels under the active input configuration")

        return indices

    @classmethod
    def _channel_keys(cls, input_config, in_channels: int) -> list[str]:
        counts = [n for n in range(cls.MAX_TRACKS + 1) if input_config.total_channels(n, n) == in_channels]
        if len(counts) != 1:
            raise ValueError(f"Could not infer a unique track count from in_channels={in_channels} under the active input configuration")

        return input_config.channel_group_keys(counts[0], counts[0])

    @classmethod
    def group_label(cls, groups: tuple) -> str:
        unknown = [group for group in groups if group not in cls.GROUPS]
        if unknown:
            raise ValueError(f"Unknown input groups {unknown}. Available: {list(cls.GROUPS)}")

        ordered = tuple(group for group in cls.GROUPS if group in groups)
        if not ordered:
            raise ValueError("Cannot label an empty trunk input selection")

        return "full" if ordered == cls.GROUPS else ordered[0]


class DualTrainingPipeline(TrainingPipeline):

    MODEL_CONFIG_IO = DualModelConfigIO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.dataset_config.input_config.use_dem:
            raise ValueError("The dual model never uses the DEM channel; disable input.use_dem")

    def _build_model(self, in_channels: int, out_channels: int):
        model, model_cfg = super()._build_model(in_channels, out_channels)

        params_arm    = sum(p.numel() for name, p in model.named_parameters() if name.startswith(("trunk_params.", "gaussian_heads.")))
        existence_arm = sum(p.numel() for name, p in model.named_parameters() if not name.startswith(("trunk_params.", "gaussian_heads.")))

        params_features    = list(model_cfg.params_features)    or "backbone default"
        existence_features = list(model_cfg.existence_features) or "backbone default"

        self.logger.subsection(f"Params Trunk : {model_cfg.params_backbone} | input {'+'.join(model_cfg.params_input)} | channels {list(model_cfg.params_channels)} | features {params_features} | {params_arm:,} params")
        self.logger.subsection(f"Exist Trunk  : {model_cfg.existence_backbone} | input {'+'.join(model_cfg.existence_input)} | channels {list(model_cfg.existence_channels)} | features {existence_features} | {existence_arm:,} params")
        return model, model_cfg

    def _architecture_label(self) -> str:
        model_config = self.model_config if self.model_config is not None else DualResUNetConfig()
        return RunNaming.dual_model_tag(model_config.params_backbone, model_config.existence_backbone)

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        model_config = self.model_config if self.model_config is not None else DualResUNetConfig()
        input_config = self.dataset_config.input_config

        overrides                        = super()._model_overrides(in_channels, out_channels)
        overrides["params_channels"]     = TrunkChannelMap.resolve(input_config, in_channels, model_config.params_input)
        overrides["existence_channels"]  = TrunkChannelMap.resolve(input_config, in_channels, model_config.existence_input)
        overrides["params_overrides"]    = {**model_config.params_overrides,    **ModelBuilder.image_size_override(model_config.params_backbone,    self.patch_size)}
        overrides["existence_overrides"] = {**model_config.existence_overrides, **ModelBuilder.image_size_override(model_config.existence_backbone, self.patch_size)}

        return overrides

    def _model_factory(self):
        return get_dual
