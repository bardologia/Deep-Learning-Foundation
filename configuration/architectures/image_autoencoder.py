from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ImageAutoencoderBaseConfig:
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
        return {
            "encoder_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "embedding_dim" : {"type": "categorical", "choices": [16, 24, 32, 48]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        groups = [
            {"params": list(model.encoder.parameters()), "lr": self.encoder_lr, "weight_decay": self.encoder_wd, "name": "image_ae_encoder"},
            {"params": list(model.decoder.parameters()), "lr": self.decoder_lr, "weight_decay": self.decoder_wd, "name": "image_ae_decoder"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


@dataclass
class Conv2dImageAutoencoderConfig(ImageAutoencoderBaseConfig):
    upsample_mode: str = "convtranspose"

    @classmethod
    def tunable_arch_params(cls) -> dict:
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
    downsample_factor : int = 2
    upsample_mode     : str = "convtranspose"

    @classmethod
    def tunable_arch_params(cls) -> dict:
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
    normalization: str = "layernorm"

    downsample_factor     : int   = 2
    upsample_mode         : str   = "convtranspose"
    stochastic_depth_rate : float = 0.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
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
    downsample_factor : int = 1
    dilation_depth    : int = 3

    @classmethod
    def tunable_arch_params(cls) -> dict:
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
    activation    : str = "gelu"
    normalization : str = "layernorm"

    patch_size : int   = 8
    hidden_dim : int   = 192
    depth      : int   = 4
    num_heads  : int   = 6
    mlp_ratio  : float = 4.0

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            **super().tunable_arch_params(),
            "patch_size" : {"type": "categorical", "choices": [4, 8, 16]},
            "hidden_dim" : {"type": "categorical", "choices": [96, 192, 384]},
            "depth"      : {"type": "categorical", "choices": [2, 4, 6]},
            "num_heads"  : {"type": "categorical", "choices": [3, 6, 12]},
            "dropout"    : {"type": "float",       "low": 0.0, "high": 0.3},
        }
