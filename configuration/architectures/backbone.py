from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class UNetConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384], [64, 128, 256, 512, 1024]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.encoder.parameters()),     'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': list(model.decoder.parameters()),     'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class ResUNetConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.encoder_blocks.parameters())
        decoder_params = list(model.upsample_layers.parameters()) + list(model.decoder_blocks.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class UNetSkipConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.encoder_blocks.parameters()) + list(model.downsample_layers.parameters())
        decoder_params = list(model.upsample_layers.parameters()) + list(model.decoder_blocks.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class AttentionUNetConfig:
    in_channels                  : int       = 1
    out_channels                 : int       = 6
    params_per_gaussian          : int       = 3
    head                         : str       = "conv"
    features                     : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor            : int       = 2
    dropout                      : float     = 0.15
    attention_intermediate_ratio : float     = 0.5
    activation                   : str       = "relu"
    normalization                : str       = "batch"
    upsample_mode                : str       = "convtranspose"
    conv_bias                    : bool      = False
    init_mode                    : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU, nn.Sigmoid,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"                     : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor"            : {"type": "categorical",         "choices": [1, 2, 4]},
            "attention_intermediate_ratio" : {"type": "float",              "low": 0.25, "high": 0.75},
            "activation"                   : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"                : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"                : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
     
        encoder_params = (
            list(model.encoder_blocks.parameters()) +
            list(model.downsample_layers.parameters())
        )
     
        decoder_params = (
            list(model.upsample_layers.parameters()) +
            list(model.attention_gates.parameters()) +
            list(model.decoder_blocks.parameters())
        )
     
        return [g for g in [
            {'params': encoder_params,                        'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),   'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                        'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),               'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class UNetPlusPlusConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [56, 112, 216, 440])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Upsample, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2, 4]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        head_ids = {id(p) for p in model.head_parameters()}

        encoder_params, dense_params, upsample_params = [], [], []
        for name, param in model.named_parameters():
            if id(param) in head_ids:
                continue
            if name.startswith("encoder_") or name.startswith("pool."):
                encoder_params.append(param)
            elif name.startswith("dense_"):
                dense_params.append(param)
            elif name.startswith("upsample.") or name.startswith("upsample_modules."):
                upsample_params.append(param)

        return [g for g in [
            {'params': encoder_params,          'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': dense_params,            'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'dense_blocks'},
            {'params': upsample_params,         'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'upsample'},
            {'params': model.head_parameters(), 'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'}
        ] if len(g['params']) > 0]


@dataclass
class LinkNetConfig:
    in_channels              : int       = 1
    out_channels             : int       = 6
    params_per_gaussian      : int       = 3
    head                     : str       = "conv"
    features                 : list[int] = field(default_factory=lambda: [152, 312, 624, 1248])
    dropout                  : float     = 0.15
    initial_kernel_size      : int       = 7
    decoder_bottleneck_ratio : int       = 4
    activation               : str       = "relu"
    normalization            : str       = "batch"
    conv_bias                : bool      = False
    init_mode                : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"                 : {"type": "indexed_categorical", "choices": [[64, 128, 256, 512], [152, 312, 624, 1248], [96, 192, 384, 768]]},
            "initial_kernel_size"      : {"type": "categorical",         "choices": [3, 5, 7]},
            "decoder_bottleneck_ratio" : {"type": "categorical",         "choices": [2, 4, 8]},
            "activation"               : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"            : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.initial_conv.parameters()),   'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'initial_conv'},
            {'params': list(model.encoder_stages.parameters()), 'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.decoder_stages.parameters()), 'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),                 'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class SwinUNetConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    head                  : str       = "conv"
    image_size            : int       = 256
    patch_size            : int       = 4
    embedding_dim         : int       = 80
    depths                : list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads             : list[int] = field(default_factory=lambda: [2, 5, 10, 20])
    window_size           : int       = 7
    mlp_ratio             : float     = 4.0
    dropout               : float     = 0.30
    attention_dropout     : float     = 0.10
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.10
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-2
    decoder_wd     : float = 1e-2
    output_head_wd : float = 1e-2
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm, nn.GELU, nn.Dropout,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "embedding_dim"         : {"type": "categorical", "choices": [40, 80, 120, 160]},
            "attention_dropout"     : {"type": "float",       "low": 0.0,  "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",       "low": 0.0,  "high": 0.3},
            "ffn_activation"        : {"type": "categorical", "choices": ["gelu", "relu", "silu"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
     
        patch_embed_params = (
            list(model.patch_embed.parameters()) +
            list(model.patch_norm.parameters())
        )
     
        encoder_params = (
            list(model.encoder_stages.parameters()) +
            list(model.downsample_layers.parameters())
        )
     
        decoder_params = (
            list(model.upsample_layers.parameters()) +
            list(model.skip_projections.parameters()) +
            list(model.decoder_stages.parameters())
        )
     
        return [g for g in [
            {'params': patch_embed_params,                        'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'patch_embed'},
            {'params': encoder_params,                            'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck_norm.parameters()),  'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'bottleneck'},
            {'params': decoder_params,                            'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': list(model.final_upsample.parameters()),   'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'final_upsample'},
            {'params': model.head_parameters(),                   'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class TransUNetConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    head                  : str       = "conv"
    image_size            : int       = 256
    cnn_features          : list[int] = field(default_factory=lambda: [32, 72, 136, 272])
    bottleneck_factor     : int       = 2
    transformer_layers    : int       = 6
    transformer_heads     : int       = 4
    transformer_mlp_ratio : float     = 4.0
    patch_size            : int       = 1
    dropout               : float     = 0.15
    activation            : str       = "relu"
    normalization         : str       = "batch"
    upsample_mode         : str       = "convtranspose"
    conv_bias             : bool      = False
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.0
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 5e-3
    decoder_wd     : float = 5e-3
    output_head_wd : float = 5e-3
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Linear, nn.LayerNorm,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
        nn.Dropout, nn.Dropout2d,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.4},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "cnn_features"          : {"type": "indexed_categorical", "choices": [[16, 32, 64, 128], [32, 72, 136, 272], [32, 64, 128, 256]]},
            "transformer_layers"    : {"type": "categorical",         "choices": [2, 4, 6, 8]},
            "transformer_heads"     : {"type": "categorical",         "choices": [2, 4, 8]},
            "attention_dropout"     : {"type": "float",               "low": 0.0, "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.2},
            "activation"            : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"         : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        
        cnn_encoder_params = (
            list(model.encoder_blocks.parameters()) +
            list(model.downsample_layers.parameters()) +
            list(model.pre_transformer_conv.parameters())
        )
        
        patch_embed_params = (
            list(model.patch_embedding.parameters()) +
            [model.positional_embedding]
        )
        
        transformer_params = (
            list(model.transformer_blocks.parameters()) +
            list(model.transformer_norm.parameters())
        )
        
        decoder_params = (
            list(model.upsample_layers.parameters()) +
            list(model.decoder_blocks.parameters())
        )
        
        return [g for g in [
            {'params': cnn_encoder_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'cnn_encoder'},
            {'params': patch_embed_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'patch_embed'},
            {'params': transformer_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'transformer'},
            {'params': decoder_params,                        'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),               'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class UNETRConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    head                  : str       = "conv"
    image_size            : int       = 256
    patch_size            : int       = 16
    embedding_dim         : int       = 544
    transformer_layers    : int       = 8
    transformer_heads     : int       = 8
    transformer_mlp_ratio : float     = 4.0
    decoder_features      : list[int] = field(default_factory=lambda: [360, 184, 88, 48])
    dropout               : float     = 0.15
    activation            : str       = "relu"
    normalization         : str       = "batch"
    conv_bias             : bool      = False
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.0
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 5e-3
    decoder_wd     : float = 5e-3
    output_head_wd : float = 5e-3
    
    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.GELU, nn.SiLU,
        nn.Dropout, nn.Dropout2d,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.4},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "embedding_dim"         : {"type": "categorical",         "choices": [256, 384, 544, 768]},
            "transformer_layers"    : {"type": "categorical",         "choices": [4, 6, 8, 12]},
            "transformer_heads"     : {"type": "categorical",         "choices": [4, 8, 16]},
            "decoder_features"      : {"type": "indexed_categorical", "choices": [[180, 96, 48, 24], [360, 184, 88, 48], [256, 128, 64, 32]]},
            "attention_dropout"     : {"type": "float",               "low": 0.0, "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.2},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
       
        patch_embed_params = (
            list(model.patch_embedding.parameters()) +
            [model.positional_embedding]
        )
       
        transformer_params = (
            list(model.transformer_blocks.parameters()) +
            list(model.transformer_norm.parameters())
        )
       
        decoder_params = (
            list(model.transformer_skip_heads.parameters()) +
            list(model.bottleneck_projection.parameters()) +
            list(model.input_skip_conv.parameters()) +
            list(model.upsample_layers.parameters()) +
            list(model.decoder_blocks.parameters()) +
            list(model.final_upsample.parameters())
        )
       
        return [g for g in [
            {'params': patch_embed_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'patch_embed'},
            {'params': transformer_params,                    'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'transformer'},
            {'params': decoder_params,                        'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),               'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class DeepLabV3PlusConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    atrous_rates        : tuple     = (1, 2, 4)
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    aspp_lr        : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    aspp_wd        : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "aspp_lr"        : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "aspp_wd"        : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"      : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "atrous_rates"  : {"type": "indexed_categorical", "choices": [(1, 2, 4), (2, 4, 8), (1, 2, 4, 8)]},
            "activation"    : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization" : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.stem.parameters()) + list(model.encoder_stages.parameters())
        decoder_params = list(model.low_level_projection.parameters()) + list(model.decoder_blocks.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.aspp.parameters()),        'lr': self.aspp_lr,        'weight_decay': self.aspp_wd,        'name': 'aspp'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class SegFormerLiteConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    head                  : str       = "conv"
    embedding_dims        : list[int] = field(default_factory=lambda: [40, 80, 192, 320])
    depths                : list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads             : tuple     = (1, 2, 4, 8)
    sr_ratios             : tuple     = (4, 2, 2, 1)
    mlp_ratio             : float     = 4.0
    decoder_channels      : int       = 256
    dropout               : float     = 0.10
    attention_dropout     : float     = 0.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.10
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-2
    decoder_wd     : float = 1e-2
    output_head_wd : float = 1e-2

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d,
        nn.GELU, nn.ReLU, nn.SiLU, nn.Dropout, nn.Dropout2d,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.4},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "embedding_dims"        : {"type": "indexed_categorical", "choices": [[32, 64, 160, 256], [40, 80, 192, 320], [64, 128, 320, 512]]},
            "depths"                : {"type": "indexed_categorical", "choices": [[2, 2, 2, 2], [2, 2, 4, 2], [3, 3, 6, 3]]},
            "decoder_channels"      : {"type": "categorical",         "choices": [128, 256, 384]},
            "attention_dropout"     : {"type": "float",               "low": 0.0, "high": 0.3},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.2},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.patch_embeddings.parameters()) + list(model.encoder_stages.parameters())
        decoder_params = list(model.decode_projections.parameters()) + list(model.fuse.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class ConvNeXtUNetConfig:
    in_channels           : int       = 1
    out_channels          : int       = 6
    params_per_gaussian   : int       = 3
    head                  : str       = "conv"
    features              : list[int] = field(default_factory=lambda: [48, 96, 192, 384])
    bottleneck_factor     : int       = 2
    blocks_per_stage      : int       = 2
    ffn_ratio             : float     = 4.0
    ffn_activation        : str       = "gelu"
    stochastic_depth_rate : float     = 0.10
    layer_scale_init      : float     = 1e-6
    conv_bias             : bool      = False
    init_mode             : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 5e-3
    bottleneck_wd  : float = 5e-3
    decoder_wd     : float = 5e-3
    output_head_wd : float = 5e-3

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LayerNorm,
        nn.GELU, nn.ReLU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"              : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [48, 96, 192, 384], [64, 128, 256, 512]]},
            "blocks_per_stage"      : {"type": "categorical",         "choices": [1, 2, 3]},
            "bottleneck_factor"     : {"type": "categorical",         "choices": [1, 2]},
            "ffn_ratio"             : {"type": "categorical",         "choices": [2.0, 4.0]},
            "stochastic_depth_rate" : {"type": "float",               "low": 0.0, "high": 0.3},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.stem.parameters()) + list(model.encoder_stages.parameters()) + list(model.downsample_layers.parameters())
        decoder_params = list(model.upsample_layers.parameters()) + list(model.decoder_stages.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class DenseUNetConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    growth_rate         : int       = 16
    block_layers        : list[int] = field(default_factory=lambda: [4, 4, 4])
    bottleneck_layers   : int       = 4
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "growth_rate"       : {"type": "categorical",         "choices": [12, 16, 24, 32]},
            "block_layers"      : {"type": "indexed_categorical", "choices": [[4, 4, 4], [4, 5, 7], [5, 5, 5, 5]]},
            "bottleneck_layers" : {"type": "categorical",         "choices": [4, 6, 8]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.stem.parameters()) + list(model.dense_down.parameters()) + list(model.trans_down.parameters())
        decoder_params = list(model.trans_up.parameters()) + list(model.dense_up.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class HRNetLiteConfig:
    in_channels         : int   = 1
    out_channels        : int   = 6
    params_per_gaussian : int   = 3
    head                : str   = "conv"
    base_channels       : int   = 48
    n_branches          : int   = 3
    blocks_per_stage    : int   = 2
    dropout             : float = 0.15
    activation          : str   = "relu"
    normalization       : str   = "batch"
    conv_bias           : bool  = False
    init_mode           : str   = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "base_channels"    : {"type": "categorical", "choices": [32, 48, 64]},
            "n_branches"       : {"type": "categorical", "choices": [2, 3, 4]},
            "blocks_per_stage" : {"type": "categorical", "choices": [1, 2, 3]},
            "activation"       : {"type": "categorical", "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"    : {"type": "categorical", "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = (
            list(model.stem.parameters()) +
            list(model.transition_modules.parameters()) +
            list(model.stage_modules.parameters()) +
            list(model.fuse_modules.parameters())
        )
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.final_fuse.parameters()),  'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class MultiResUNetConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_factor   : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    upsample_mode       : str       = "convtranspose"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"          : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "bottleneck_factor" : {"type": "categorical",         "choices": [1, 2]},
            "activation"        : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"     : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
            "upsample_mode"     : {"type": "categorical",         "choices": ["convtranspose", "bilinear"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.encoder_blocks.parameters()) + list(model.downsample_layers.parameters()) + list(model.res_paths.parameters())
        decoder_params = list(model.upsample_layers.parameters()) + list(model.decoder_blocks.parameters())
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bottleneck.parameters()),  'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bottleneck'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class FPNNetConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    pyramid_channels    : int       = 128
    segmentation_convs  : int       = 2
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"           : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "pyramid_channels"   : {"type": "categorical",         "choices": [64, 128, 256]},
            "segmentation_convs" : {"type": "categorical",         "choices": [1, 2, 3]},
            "activation"         : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization"      : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.encoder_blocks.parameters()) + list(model.downsample_layers.parameters())
        decoder_params = (
            list(model.lateral_convs.parameters()) +
            list(model.smooth_convs.parameters()) +
            list(model.segmentation_blocks.parameters()) +
            list(model.fuse_block.parameters())
        )
        return [g for g in [
            {'params': encoder_params,                       'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': decoder_params,                       'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class U2NetLiteConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    rsu_heights         : tuple     = (5, 4, 3)
    dropout             : float     = 0.15
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"      : {"type": "indexed_categorical", "choices": [[32, 64, 128, 256], [64, 128, 256, 512], [48, 96, 192, 384]]},
            "rsu_heights"   : {"type": "indexed_categorical", "choices": [(4, 3, 2), (5, 4, 3), (6, 5, 4)]},
            "activation"    : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization" : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.encoder_stages.parameters()), 'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.bridge.parameters()),         'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'bridge'},
            {'params': list(model.decoder_stages.parameters()), 'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),                 'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class PixelMLPNetConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    features            : list[int] = field(default_factory=lambda: [3200, 3200, 3200, 3200])
    dropout             : float     = 0.30
    activation          : str       = "relu"
    normalization       : str       = "batch"
    conv_bias           : bool      = False
    init_mode           : str       = "default"

    trunk_lr       : float = 3e-4
    output_head_lr : float = 1e-3

    trunk_wd       : float = 1e-2
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Dropout,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "trunk_lr"       : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "trunk_wd"       : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"      : {"type": "indexed_categorical", "choices": [[2048, 2048, 2048, 2048], [3200, 3200, 3200, 3200], [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]]},
            "activation"    : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization" : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.trunk.parameters()),       'lr': self.trunk_lr,       'weight_decay': self.trunk_wd,       'name': 'trunk'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class LocalCNNConfig:
    in_channels         : int              = 1
    out_channels        : int              = 6
    params_per_gaussian : int              = 3
    head                : str              = "conv"
    features            : list[int]        = field(default_factory=lambda: [1072, 1072])
    block_kernels       : list[int] | None = None
    dropout             : float            = 0.30
    activation          : str              = "relu"
    normalization       : str              = "batch"
    conv_bias           : bool             = False
    init_mode           : str              = "default"

    trunk_lr       : float = 3e-4
    output_head_lr : float = 1e-3

    trunk_wd       : float = 1e-2
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "trunk_lr"       : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "trunk_wd"       : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.5},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "features"      : {"type": "indexed_categorical", "choices": [[512, 512], [1072, 1072], [1408, 1408]]},
            "activation"    : {"type": "categorical",         "choices": ["relu", "leaky_relu", "gelu", "silu"]},
            "normalization" : {"type": "categorical",         "choices": ["batch", "instance", "group"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.trunk.parameters()),       'lr': self.trunk_lr,       'weight_decay': self.trunk_wd,       'name': 'trunk'},
            {'params': model.head_parameters(),              'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


@dataclass
class NAFNetConfig:
    in_channels         : int       = 1
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "conv"
    width               : int       = 32
    enc_blocks          : list[int] = field(default_factory=lambda: [2, 2, 4, 8])
    middle_blocks       : int       = 12
    dec_blocks          : list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dw_expand           : int       = 2
    ffn_expand          : int       = 2
    dropout             : float     = 0.0
    init_mode           : str       = "default"

    encoder_lr     : float = 3e-4
    bottleneck_lr  : float = 3e-4
    decoder_lr     : float = 3e-4
    output_head_lr : float = 1e-3

    encoder_wd     : float = 1e-4
    bottleneck_wd  : float = 1e-4
    decoder_wd     : float = 1e-4
    output_head_wd : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.PixelShuffle, nn.AdaptiveAvgPool2d,
        nn.LayerNorm, nn.Dropout,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "encoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "bottleneck_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "decoder_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "encoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "bottleneck_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "decoder_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "dropout"        : {"type": "float", "low": 0.0, "high": 0.3},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "width"         : {"type": "categorical",         "choices": [16, 24, 32, 48]},
            "enc_blocks"    : {"type": "indexed_categorical", "choices": [[2, 2, 4, 8], [1, 1, 1, 28], [2, 2, 2, 2]]},
            "middle_blocks" : {"type": "categorical",         "choices": [1, 6, 12]},
            "dw_expand"     : {"type": "categorical",         "choices": [1, 2]},
            "ffn_expand"    : {"type": "categorical",         "choices": [1, 2]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        encoder_params = list(model.intro.parameters()) + list(model.encoder_stages.parameters()) + list(model.downsample_layers.parameters())
        decoder_params = list(model.upsample_layers.parameters()) + list(model.decoder_stages.parameters())
        return [g for g in [
            {'params': encoder_params,                          'lr': self.encoder_lr,     'weight_decay': self.encoder_wd,     'name': 'encoder'},
            {'params': list(model.middle_stage.parameters()),   'lr': self.bottleneck_lr,  'weight_decay': self.bottleneck_wd,  'name': 'middle'},
            {'params': decoder_params,                          'lr': self.decoder_lr,     'weight_decay': self.decoder_wd,     'name': 'decoder'},
            {'params': model.head_parameters(),                 'lr': self.output_head_lr, 'weight_decay': self.output_head_wd, 'name': 'output_head'},
        ] if len(g['params']) > 0]


