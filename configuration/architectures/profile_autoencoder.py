"""Architecture configuration dataclasses for the 1D profile autoencoder family.

Profile autoencoders compress a single reflectivity profile along the z
axis to a low-dimensional embedding and reconstruct it. The base class holds
the fields shared by every family and each subclass adds the fields specific to
its encoder-decoder topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from configuration.architectures.param_groups import ParamGroups


@dataclass
class ProfileAutoencoderBaseConfig:
    """Fields shared by every 1D profile autoencoder configuration.

    Attributes:
        profile_length : Number of z samples in an input profile.
        embedding_dim  : Width of the latent representation.
        embedding_norm : Normalisation applied to the latent representation.
        activation     : Activation function name used throughout the network.
        init_mode      : Weight initialisation scheme name.
        encoder_lr     : Learning rate of the encoder.
        decoder_lr     : Learning rate of the decoder.
        encoder_wd     : Weight decay of the encoder.
        decoder_wd     : Weight decay of the decoder.
    """

    profile_length: int = 256

    embedding_dim  : int = 24
    embedding_norm : str = "layernorm"

    activation : str = "gelu"
    init_mode  : str = "default"

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
        """Returns the tuner search space shared by all profile autoencoder families.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            "embedding_dim" : {"type": "categorical", "choices": [16, 24, 32, 48]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        """Returns optimiser parameter groups for the encoder, embedding norm and decoder.

        Args:
            model : Instantiated profile autoencoder exposing ``encoder``, ``decoder`` and an optional ``embedding_layernorm``.

        Returns:
            Verified list of optimiser parameter-group dicts covering every
            trainable parameter of the model exactly once.
        """
        embedding_params = list(model.embedding_layernorm.parameters()) if model.embedding_layernorm is not None else []

        groups = [
            {"params": list(model.encoder.parameters()), "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "ae_encoder"},
            {"params": embedding_params,                 "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "ae_embedding_norm"},
            {"params": list(model.decoder.parameters()), "lr": self.decoder_lr, "weight_decay": self.decoder_wd, "name": "ae_decoder"},
        ]

        return ParamGroups.verified(model, groups)


@dataclass
class MlpAutoencoderConfig(ProfileAutoencoderBaseConfig):
    """Configuration of the fully connected profile autoencoder.

    Attributes:
        hidden_dim    : Width of each hidden layer.
        depth         : Number of hidden layers per encoder and decoder.
        dropout       : Dropout probability between hidden layers.
        normalization : Per-pixel normalization after each hidden layer, one of none, batchnorm or layernorm.
        hidden_dims   : Explicit per-layer hidden widths; None uses depth copies of hidden_dim.
    """

    hidden_dim    : int              = 512
    depth         : int              = 4
    dropout       : float            = 0.0
    normalization : str              = "none"
    hidden_dims   : list[int] | None = None

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "hidden_dim"    : {"type": "categorical", "choices": [256, 512, 768, 1024]},
            "depth"         : {"type": "categorical", "choices": [3, 4, 6, 8]},
            "activation"    : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "dropout"       : {"type": "float",       "low": 0.0, "high": 0.3},
            "normalization" : {"type": "categorical", "choices": ["none", "batchnorm", "layernorm"]},
        }


@dataclass
class Conv1dAutoencoderConfig(ProfileAutoencoderBaseConfig):
    """Configuration of the 1D convolutional profile autoencoder.

    Attributes:
        seq_channels      : Channel width of the convolutional stack.
        seq_kernel_size   : Kernel size along the z axis.
        depth             : Number of convolution layers per encoder and decoder body.
        dropout           : Dropout probability between convolution layers.
        normalization     : Sequence normalization after each convolution, one of none or batchnorm.
        pooling           : Length-axis pooling to the latent code, one of avg, max or attention.
        seq_channels_list : Explicit per-layer channel widths of the encoder body, mirrored in reverse by the decoder; None uses depth copies of seq_channels.
    """

    seq_channels      : int              = 192
    seq_kernel_size   : int              = 5
    depth             : int              = 2
    dropout           : float            = 0.0
    normalization     : str              = "none"
    pooling           : str              = "avg"
    seq_channels_list : list[int] | None = None

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "seq_channels"    : {"type": "categorical", "choices": [128, 192, 256]},
            "seq_kernel_size" : {"type": "categorical", "choices": [3, 5, 7]},
            "depth"           : {"type": "categorical", "choices": [2, 3, 4]},
            "activation"      : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "normalization"   : {"type": "categorical", "choices": ["none", "batchnorm"]},
            "pooling"         : {"type": "categorical", "choices": ["avg", "max", "attention"]},
        }


@dataclass
class Transformer1dAutoencoderConfig(ProfileAutoencoderBaseConfig):
    """Configuration of the transformer profile autoencoder.

    Attributes:
        hidden_dim : Token width of the transformer blocks.
        depth      : Number of transformer blocks per encoder and decoder.
        num_heads  : Attention heads per transformer block.
        dropout    : Dropout probability inside the blocks.
        ffn_ratio  : Feed-forward width as a multiple of hidden_dim.
    """

    hidden_dim : int   = 192
    depth      : int   = 3
    num_heads  : int   = 4
    dropout    : float = 0.0
    ffn_ratio  : float = 2.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "hidden_dim" : {"type": "categorical", "choices": [128, 192, 256]},
            "depth"      : {"type": "categorical", "choices": [2, 3, 4]},
            "dropout"    : {"type": "float",       "low": 0.0, "high": 0.3},
            "activation" : {"type": "categorical", "choices": ["relu", "gelu"]},
        }


@dataclass
class ResMlpAutoencoderConfig(ProfileAutoencoderBaseConfig):
    """Configuration of the residual-MLP profile autoencoder.

    Attributes:
        hidden_dim    : Width of each residual block.
        depth         : Number of residual blocks per encoder and decoder.
        dropout       : Dropout probability inside the blocks.
        normalization : Pre-norm layer of each residual block, one of layernorm or none.
    """

    hidden_dim    : int   = 384
    depth         : int   = 3
    dropout       : float = 0.0
    normalization : str   = "layernorm"

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "hidden_dim"    : {"type": "categorical", "choices": [256, 384, 512]},
            "depth"         : {"type": "categorical", "choices": [2, 3, 4, 6]},
            "activation"    : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "dropout"       : {"type": "float",       "low": 0.0, "high": 0.3},
            "normalization" : {"type": "categorical", "choices": ["layernorm", "none"]},
        }


@dataclass
class TcnAutoencoderConfig(ProfileAutoencoderBaseConfig):
    """Configuration of the temporal convolutional profile autoencoder.

    Attributes:
        seq_channels    : Channel width of the convolutional stack.
        seq_kernel_size : Kernel size along the z axis.
        depth           : Number of dilated convolution levels.
        dropout         : Dropout probability inside the blocks.
        normalization   : Sequence normalization between the dilated convolutions, one of none or batchnorm.
        pooling         : Length-axis pooling to the latent code, one of avg, max or attention.
    """

    seq_channels    : int   = 128
    seq_kernel_size : int   = 3
    depth           : int   = 4
    dropout         : float = 0.0
    normalization   : str   = "none"
    pooling         : str   = "avg"

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "seq_channels"    : {"type": "categorical", "choices": [64, 96, 128]},
            "seq_kernel_size" : {"type": "categorical", "choices": [3, 5, 7]},
            "depth"           : {"type": "categorical", "choices": [3, 4, 5]},
            "activation"      : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
            "dropout"         : {"type": "float",       "low": 0.0, "high": 0.3},
            "normalization"   : {"type": "categorical", "choices": ["none", "batchnorm"]},
            "pooling"         : {"type": "categorical", "choices": ["avg", "max", "attention"]},
        }


@dataclass
class GruAutoencoderConfig(ProfileAutoencoderBaseConfig):
    """Configuration of the recurrent GRU profile autoencoder.

    Attributes:
        hidden_dim    : Hidden state width of the GRU.
        depth         : Number of stacked GRU layers.
        bidirectional : Whether the encoder GRU reads the profile in both directions.
        dropout       : Dropout probability between GRU layers.
    """

    hidden_dim    : int   = 224
    depth         : int   = 2
    bidirectional : bool  = True
    dropout       : float = 0.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "hidden_dim"    : {"type": "categorical", "choices": [128, 224, 320]},
            "depth"         : {"type": "categorical", "choices": [1, 2, 3]},
            "bidirectional" : {"type": "categorical", "choices": [True, False]},
            "dropout"       : {"type": "float",       "low": 0.0, "high": 0.3},
        }


@dataclass
class CnnAttnAutoencoderConfig(ProfileAutoencoderBaseConfig):
    """Configuration of the profile autoencoder mixing a convolutional stem with attention.

    Attributes:
        seq_channels    : Channel width of the convolutional stem.
        seq_kernel_size : Kernel size along the z axis.
        patch_size      : Number of z samples pooled into one token.
        hidden_dim      : Token width of the attention blocks.
        depth           : Number of attention blocks.
        num_heads       : Attention heads per block.
        dropout         : Dropout probability inside the blocks.
        ffn_ratio       : Feed-forward width as a multiple of hidden_dim.
        pooling         : Token pooling to the latent code, one of avg, max or attention.
        pos_embedding   : Positional embedding of the tokens, one of learned, sinusoidal or none.
    """

    seq_channels    : int   = 32
    seq_kernel_size : int   = 5
    patch_size      : int   = 8
    hidden_dim      : int   = 192
    depth           : int   = 2
    num_heads       : int   = 4
    dropout         : float = 0.0
    ffn_ratio       : float = 4.0
    pooling         : str   = "avg"
    pos_embedding   : str   = "learned"

    @classmethod
    def tunable_arch_params(cls) -> dict:
        """Returns the base search space extended with this family's structural fields.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            **super().tunable_arch_params(),
            "seq_channels"    : {"type": "categorical", "choices": [16, 32, 64]},
            "seq_kernel_size" : {"type": "categorical", "choices": [3, 5, 7]},
            "patch_size"      : {"type": "categorical", "choices": [4, 8, 16]},
            "hidden_dim"      : {"type": "categorical", "choices": [128, 192, 256]},
            "depth"           : {"type": "categorical", "choices": [2, 3, 4]},
            "num_heads"       : {"type": "categorical", "choices": [2, 4, 8]},
            "dropout"         : {"type": "float",       "low": 0.0, "high": 0.3},
            "pooling"         : {"type": "categorical", "choices": ["avg", "max", "attention"]},
            "pos_embedding"   : {"type": "categorical", "choices": ["learned", "sinusoidal", "none"]},
        }
