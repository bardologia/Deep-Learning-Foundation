"""Training pipeline of the dual-trunk model and its input-routing channel map.

The channel map turns named input groups (primary and secondary channels,
derived channels, the auxiliary channel) into the channel indices
each trunk reads, and the pipeline builds the two-trunk model on top of the
backbone training pipeline.
"""

from __future__ import annotations

from configuration.architectures                import DualResUNetConfig
from models.dual                                import get_dual
from pipelines.backbone.training.pipeline       import TrainingPipeline
from pipelines.shared.config.config_persistence import DualModelConfigIO
from pipelines.shared.model.model_builder       import ModelBuilder
from pipelines.shared.training.run_naming       import RunNaming


class TrunkChannelMap:
    """Maps named input channel groups onto the channel indices a trunk reads.

    The groups are the sources of the assembled input stack: `chan` for the
    primary and secondary channels, `derived` for the derived channels, `aux`
    for the auxiliary channel. A selected group that carries no channels under
    the active input configuration contributes nothing, so `aux` may stay
    selected while `input.use_aux` is off.
    """

    MAX_TRACKS = 64
    GROUPS     = ("chan", "derived", "aux")

    @classmethod
    def resolve(cls, input_config, in_channels: int, groups: tuple) -> tuple[int, ...]:
        """Returns the indices of the input channels belonging to the given groups.

        Args:
            input_config : Dataset input configuration describing the stack layout.
            in_channels  : Width of the assembled input stack.
            groups       : Channel group names routed into the trunk.

        Returns:
            Channel indices into the input stack, in stack order.

        Raises:
            ValueError : If a group name is unknown or the selection is empty under the active input configuration.
        """
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
        """Returns the per-channel `source/slot_kind` keys of a stack of the given width.

        Args:
            input_config : Dataset input configuration describing the stack layout.
            in_channels  : Width of the assembled input stack.

        Raises:
            ValueError : If no track count, or more than one, reproduces `in_channels` under the active input configuration.
        """
        counts = [n for n in range(cls.MAX_TRACKS + 1) if input_config.total_channels(n, n) == in_channels]
        if len(counts) != 1:
            raise ValueError(f"Could not infer a unique track count from in_channels={in_channels} under the active input configuration")

        return input_config.channel_group_keys(counts[0], counts[0])

    @classmethod
    def group_label(cls, groups: tuple) -> str:
        """Returns the run-name label of a trunk input selection.

        Args:
            groups : Channel group names routed into the trunk.

        Returns:
            `full` when every group is selected, otherwise the selected groups
            joined with `+` in canonical order.

        Raises:
            ValueError : If a group name is unknown or the selection is empty.
        """
        unknown = [group for group in groups if group not in cls.GROUPS]
        if unknown:
            raise ValueError(f"Unknown input groups {unknown}. Available: {list(cls.GROUPS)}")

        ordered = tuple(group for group in cls.GROUPS if group in groups)
        if not ordered:
            raise ValueError("Cannot label an empty trunk input selection")

        return "full" if ordered == cls.GROUPS else "+".join(ordered)


class DualTrainingPipeline(TrainingPipeline):
    """Backbone training pipeline specialised for the two-trunk dual model."""

    MODEL_CONFIG_IO = DualModelConfigIO

    def _square_patch_backbones(self) -> list[str]:
        """Returns both trunk backbone names, which decide whether patches must be squared."""
        model_config = self.model_config if self.model_config is not None else DualResUNetConfig()

        return [model_config.params_backbone, model_config.existence_backbone]

    def _build_model(self, in_channels: int, out_channels: int):
        """Builds the dual model and logs the composition and parameter count of each arm.

        Args:
            in_channels  : Width of the assembled input stack.
            out_channels : Width of the Gaussian-parameter output.

        Returns:
            The instantiated dual model and its resolved model configuration.
        """
        model, model_cfg = super()._build_model(in_channels, out_channels)

        params_arm    = sum(p.numel() for name, p in model.named_parameters() if name.startswith(("trunk_params.", "gaussian_heads.")))
        existence_arm = sum(p.numel() for name, p in model.named_parameters() if not name.startswith(("trunk_params.", "gaussian_heads.")))

        params_features    = list(model_cfg.params_features)    or "backbone default"
        existence_features = list(model_cfg.existence_features) or "backbone default"

        self.logger.section("[Dual Trunk Configuration]")
        self.logger.kv_table({
            "Params Backbone"   : model_cfg.params_backbone,
            "Params Input"      : "+".join(model_cfg.params_input),
            "Params Channels"   : str(list(model_cfg.params_channels)),
            "Params Features"   : str(params_features),
            "Params Parameters" : f"{params_arm:,}",
            "Exist Backbone"    : model_cfg.existence_backbone,
            "Exist Input"       : "+".join(model_cfg.existence_input),
            "Exist Channels"    : str(list(model_cfg.existence_channels)),
            "Exist Features"    : str(existence_features),
            "Exist Parameters"  : f"{existence_arm:,}",
        })
        return model, model_cfg

    def _architecture_label(self) -> str:
        """Returns the run-name fragment combining both trunk backbones."""
        model_config = self.model_config if self.model_config is not None else DualResUNetConfig()
        return RunNaming.dual_model_tag(model_config.params_backbone, model_config.existence_backbone)

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        """Returns the model constructor overrides, adding per-trunk channel routing and image sizing.

        Args:
            in_channels  : Width of the assembled input stack.
            out_channels : Width of the Gaussian-parameter output.
        """
        model_config = self.model_config if self.model_config is not None else DualResUNetConfig()
        input_config = self.dataset_config.input_config

        overrides                        = super()._model_overrides(in_channels, out_channels)
        overrides["params_channels"]     = TrunkChannelMap.resolve(input_config, in_channels, model_config.params_input)
        overrides["existence_channels"]  = TrunkChannelMap.resolve(input_config, in_channels, model_config.existence_input)
        overrides["params_overrides"]    = {**model_config.params_overrides,    **ModelBuilder.image_size_override(model_config.params_backbone,    self.patch_size)}
        overrides["existence_overrides"] = {**model_config.existence_overrides, **ModelBuilder.image_size_override(model_config.existence_backbone, self.patch_size)}

        return overrides

    def _model_factory(self):
        """Returns the dual model factory used to instantiate the architecture."""
        return get_dual
