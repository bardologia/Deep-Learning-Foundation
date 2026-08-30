"""Dual-trunk model that separates scatterer parameters from scatterer existence.

One backbone reads the input channels chosen for parameter regression, a second reads
the channels chosen for the presence decision, and the set-prediction head gates each
slot's amplitude with the second trunk's embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configuration.architectures import DualResUNetConfig
from ..backbone                  import BACKBONE_CONFIG_REGISTRY, BACKBONE_MODEL_REGISTRY
from ..blocks                    import LayerFactory, OutputHeadsMixin, PixelMLP


class DualResUNet(nn.Module, OutputHeadsMixin):
    """Two independent backbones feeding a shared set-prediction head.

    Each trunk may be any registered backbone and sees its own subset of the input
    channels, selected by index. Only the ``set_pred`` head is supported, since the
    split into parameters and existence is what the architecture exists for.

    Attributes:
        RESERVED_TRUNK_FIELDS : Trunk config fields the dual model sets itself and that per-trunk overrides may not touch.
        config                : Architecture configuration for both trunks and the head.
        trunk_params          : Backbone producing the parameter embedding.
        trunk_existence       : Backbone producing the existence embedding.
        embedding_channels    : Channel width of the parameter trunk embedding.
        params_index          : Buffer of input channel indices routed to the parameter trunk.
        existence_index       : Buffer of input channel indices routed to the existence trunk.
    """

    RESERVED_TRUNK_FIELDS = ("in_channels", "out_channels", "params_per_gaussian", "head", "init_mode", "features")

    def __init__(self, config: DualResUNetConfig | None = None):
        """Builds both trunks, the set-prediction head and the channel routing buffers.

        Args:
            config : Architecture configuration; defaults to ``DualResUNetConfig()``.
        """
        super().__init__()
        self.config = config if config is not None else DualResUNetConfig()

        self._validate()

        self.trunk_params    = self._build_trunk("params_backbone",    self.config.params_backbone,    self.config.params_channels,    self.config.params_features,    self.config.params_overrides)
        self.trunk_existence = self._build_trunk("existence_backbone", self.config.existence_backbone, self.config.existence_channels, self.config.existence_features, self.config.existence_overrides)

        self.embedding_channels = self.trunk_params.embedding_channels
        self._build_output_head()

        self.register_buffer("params_index",    torch.tensor(list(self.config.params_channels),    dtype=torch.long), persistent=False)
        self.register_buffer("existence_index", torch.tensor(list(self.config.existence_channels), dtype=torch.long), persistent=False)

        LayerFactory.initialize_weights(module=self, mode=self.config.init_mode)

    def _validate(self) -> None:
        """Checks the head choice, the trunk names and the channel index ranges.

        Raises:
            ValueError : If the head is not ``set_pred``, if either backbone name is unregistered, or if either channel index list is empty or reaches outside ``in_channels``.
        """
        config = self.config

        if config.head != "set_pred":
            raise ValueError(f"DualResUNet only supports the set_pred head; got '{config.head}'")

        for name, backbone in (("params_backbone", config.params_backbone), ("existence_backbone", config.existence_backbone)):
            if backbone not in BACKBONE_MODEL_REGISTRY:
                raise ValueError(f"Unknown {name} '{backbone}'. Available: {list(BACKBONE_MODEL_REGISTRY)}")

        for name, channels in (("params_channels", config.params_channels), ("existence_channels", config.existence_channels)):
            if len(channels) == 0:
                raise ValueError(f"{name} must list the input channel indices that feed the trunk")
            if any(index < 0 or index >= config.in_channels for index in channels):
                raise ValueError(f"{name} {tuple(channels)} out of range for in_channels={config.in_channels}")

    def _build_trunk(self, arm: str, backbone: str, channels: tuple, features: list[int], overrides: dict):
        """Instantiates one headless backbone from the dual configuration.

        Args:
            arm       : Name of the dual config field naming this trunk, used in error messages.
            backbone  : Registered backbone key.
            channels  : Input channel indices routed to this trunk; its length becomes the trunk's ``in_channels``.
            features  : Channel ladder overriding the backbone default, or empty to keep it.
            overrides : Extra trunk config fields to set.

        Returns:
            The constructed backbone, built with ``head="none"`` so it returns embeddings.

        Raises:
            ValueError     : If ``features`` is given for a backbone with no ``features`` ladder, or if an override names a field the dual model owns.
            AttributeError : If an override names a field the trunk config does not have.
        """
        trunk_config = BACKBONE_CONFIG_REGISTRY[backbone](
            in_channels         = len(channels),
            out_channels        = self.config.out_channels,
            params_per_gaussian = self.config.params_per_gaussian,
            head                = "none",
            init_mode           = self.config.init_mode,
        )

        if features:
            if not hasattr(trunk_config, "features"):
                raise ValueError(f"{arm} '{backbone}' has no 'features' ladder; clear the {arm.replace('_backbone', '_features')} list ([]) and size the trunk via its overrides instead")
            trunk_config.features = list(features)

        for attribute, value in overrides.items():
            if attribute in self.RESERVED_TRUNK_FIELDS:
                raise ValueError(f"Trunk override '{attribute}' for {arm} is set by the dual model itself; use the dedicated dual config fields")
            if not hasattr(trunk_config, attribute):
                raise AttributeError(f"Unknown trunk override '{attribute}' for {type(trunk_config).__name__}")
            setattr(trunk_config, attribute, value)

        return BACKBONE_MODEL_REGISTRY[backbone](trunk_config)

    def _head_activation(self) -> str:
        """Returns the activation name used inside the output head."""
        return self.config.head_activation

    def _build_set_prediction_heads(self) -> None:
        """Builds the per-slot parameter heads and sizes the gate to the existence trunk."""
        self._build_per_gaussian_heads()

        existence_channels  = self.trunk_existence.embedding_channels
        existence_hidden    = max(existence_channels // 2, 16)
        self.existence_head = PixelMLP(existence_channels, existence_hidden, self.n_gaussians, self._head_activation())
        self.amp_off        = nn.Parameter(torch.zeros(self.n_gaussians))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs both trunks on their own channel subsets and gates the slot amplitudes.

        Args:
            x : Input feature stack of shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, out_channels, height, width) holding the
            interleaved Gaussian parameters with gated amplitudes.
        """
        embedding_params    = self.trunk_params.encode_decode(x.index_select(1, self.params_index))
        embedding_existence = self.trunk_existence.encode_decode(x.index_select(1, self.existence_index))

        return self._set_prediction_forward(embedding_params, embedding_existence)
