"""Architecture configuration dataclasses for the 2D image autoencoder family.

Image autoencoders compress a volumetric image patch to a per-pixel embedding
and reconstruct it; the base class holds the fields shared by every family and
each subclass adds the fields specific to its encoder-decoder topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from configuration.architectures.param_groups import ParamGroups


@dataclass
class ImageAutoencoderBaseConfig:
    """Fields shared by every 2D image autoencoder configuration.

    Attributes:
        in_channels       : Channels of the input image patch.
        embedding_dim     : Width of the latent representation.
        embedding_norm    : Normalisation applied to the latent representation.
        downsample_factor : Spatial reduction between input and latent grid.
        base_channels     : Channel width of the first encoder stage.
        depth             : Number of encoder stages, mirrored in the decoder.
        activation        : Activation function name used throughout the network.
        normalization     : Normalisation layer name used throughout the network.
        dropout           : Dropout probability inside the blocks.
        init_mode         : Weight initialisation scheme name.
        encoder_lr        : Learning rate of the encoder.
        decoder_lr        : Learning rate of the decoder.
        encoder_wd        : Weight decay of the encoder.
        decoder_wd        : Weight decay of the decoder.
    """

    in_channels: int = 1

    embedding_dim  : int = 24
    embedding_norm : str = "none"

    downsample_factor : int = 1
    base_channels     : int = 32
    depth             : int = 2

    activation    : str   = "gelu"
    normalization : str   = "batch"
    dropout       : float = 0.0
    init_mode     : str   = "default"

    encoder_lr : float = 3e-4
    decoder_lr : float = 3e-4

    encoder_wd : float = 1e-4
    decoder_wd : float = 1e-4

    @classmethod
    def tunable_lr_params(cls) -> dict:
        """Returns the tuner search space for encoder and decoder learning rates and decays.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            "encoder_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the tuner search space shared by all image autoencoder families.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            "embedding_dim" : {"type": "categorical", "choices": [16, 24, 32, 48]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        """Returns optimiser parameter groups for the encoder and the decoder.

        Args:
            model : Instantiated image autoencoder exposing ``encoder`` and ``decoder``.

        Returns:
            Verified list of optimiser parameter-group dicts covering every
            trainable parameter of the model exactly once.
        """
        groups = [
            {"params": list(model.encoder.parameters()), "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "image_ae_encoder"},
            {"params": list(model.decoder.parameters()), "lr": self.decoder_lr, "weight_decay": self.decoder_wd, "name": "image_ae_decoder"},
        ]

        return ParamGroups.verified(model, groups)


@dataclass
class Conv2dImageAutoencoderConfig(ImageAutoencoderBaseConfig):
    """Configuration of the plain strided-convolution image autoencoder.

    Attributes:
        upsample_mode : Decoder upsampling operator.
    """

    upsample_mode: str = "convtranspose"

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "base_channels"     : {"type": "categorical", "choices": [16, 32, 64]},
            "depth"             : {"type": "categorical", "choices": [1, 2, 3]},
            "downsample_factor" : {"type": "categorical", "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical", "choices": ["batch", "instance", "group"]},
            "dropout"           : {"type": "float",       "low": 0.0, "high": 0.3},
        }


@dataclass
class ResNet2dImageAutoencoderConfig(ImageAutoencoderBaseConfig):
    """Configuration of the residual-block image autoencoder.

    Attributes:
        downsample_factor : Spatial reduction between input and latent grid.
        upsample_mode     : Decoder upsampling operator.
    """

    downsample_factor : int = 2
    upsample_mode     : str = "convtranspose"

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "base_channels"     : {"type": "categorical", "choices": [16, 32, 64]},
            "depth"             : {"type": "categorical", "choices": [1, 2, 3]},
            "downsample_factor" : {"type": "categorical", "choices": [2, 4, 8]},
            "activation"        : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical", "choices": ["batch", "instance", "group"]},
            "dropout"           : {"type": "float",       "low": 0.0, "high": 0.3},
        }


@dataclass
class ConvNeXt2dImageAutoencoderConfig(ImageAutoencoderBaseConfig):
    """Configuration of the ConvNeXt-block image autoencoder.

    Attributes:
        normalization         : Normalisation layer name, layer norm by default.
        downsample_factor     : Spatial reduction between input and latent grid.
        upsample_mode         : Decoder upsampling operator.
        stochastic_depth_rate : Maximum drop-path probability across blocks.
    """

    normalization: str = "layernorm"

    downsample_factor     : int   = 2
    upsample_mode         : str   = "convtranspose"
    stochastic_depth_rate : float = 0.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "base_channels"         : {"type": "categorical", "choices": [16, 32, 64]},
            "depth"                 : {"type": "categorical", "choices": [1, 2, 3]},
            "downsample_factor"     : {"type": "categorical", "choices": [2, 4, 8]},
            "activation"            : {"type": "categorical", "choices": ["gelu", "silu"]},
            "stochastic_depth_rate" : {"type": "float",       "low": 0.0, "high": 0.3},
        }


@dataclass
class DilatedConv2dImageAutoencoderConfig(ImageAutoencoderBaseConfig):
    """Configuration of the dilated-convolution image autoencoder that keeps full resolution.

    Attributes:
        downsample_factor : Spatial reduction between input and latent grid.
        dilation_depth    : Number of stacked layers with growing dilation.
    """

    downsample_factor : int = 1
    dilation_depth    : int = 3

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "base_channels"  : {"type": "categorical", "choices": [16, 32, 64]},
            "dilation_depth" : {"type": "categorical", "choices": [2, 3, 4]},
            "activation"     : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "normalization"  : {"type": "categorical", "choices": ["batch", "instance", "group"]},
            "dropout"        : {"type": "float",       "low": 0.0, "high": 0.3},
        }


@dataclass
class ViTImageAutoencoderConfig(ImageAutoencoderBaseConfig):
    """Configuration of the vision-transformer image autoencoder.

    Attributes:
        patch_size : Side length of the tokens produced by patch embedding.
        hidden_dim : Token width of the transformer blocks.
        depth      : Number of transformer blocks in the encoder.
        num_heads  : Attention heads per transformer block.
        mlp_ratio  : Hidden-to-token width ratio of the feed-forward blocks.
    """

    activation    : str = "gelu"
    normalization : str = "layernorm"

    patch_size : int   = 8
    hidden_dim : int   = 192
    depth      : int   = 4
    num_heads  : int   = 6
    mlp_ratio  : float = 4.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "patch_size" : {"type": "categorical", "choices": [4, 8, 16]},
            "hidden_dim" : {"type": "categorical", "choices": [96, 192, 384]},
            "depth"      : {"type": "categorical", "choices": [2, 4, 6]},
            "num_heads"  : {"type": "categorical", "choices": [3, 6, 12]},
            "dropout"    : {"type": "float",       "low": 0.0, "high": 0.3},
        }
