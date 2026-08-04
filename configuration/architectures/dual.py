from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class DualResUNetConfig:
    in_channels         : int       = 9
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "set_pred"
    params_backbone     : str       = "unet_skip"
    existence_backbone  : str       = "unet_skip"
    params_input        : tuple     = ("pass", "ifg")
    existence_input     : tuple     = ("pass", "ifg")
    params_channels     : tuple     = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    existence_channels  : tuple     = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    params_features     : list[int] = field(default_factory=lambda: [64, 128, 248, 472])
    existence_features  : list[int] = field(default_factory=lambda: [24, 48, 84, 156])
    params_overrides    : dict      = field(default_factory=dict)
    existence_overrides : dict      = field(default_factory=dict)
    head_activation     : str       = "relu"
    init_mode           : str       = "default"

    params_trunk_lr    : float = 3e-4
    existence_trunk_lr : float = 3e-4
    output_head_lr     : float = 1e-3

    params_trunk_wd    : float = 1e-4
    existence_trunk_wd : float = 1e-4
    output_head_wd     : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.Dropout2d,
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "params_trunk_lr"    : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "existence_trunk_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "output_head_lr"     : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "params_trunk_wd"    : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "existence_trunk_wd" : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "output_head_wd"     : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "params_backbone"    : {"type": "categorical",         "choices": ["unet_skip", "resunet", "unet"]},
            "existence_backbone" : {"type": "categorical",         "choices": ["unet_skip", "resunet", "unet"]},
            "params_features"    : {"type": "indexed_categorical", "choices": [[48, 96, 184, 352], [64, 128, 248, 472]]},
            "existence_features" : {"type": "indexed_categorical", "choices": [[48, 96, 184, 352], [24, 48, 84, 156]]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': list(model.trunk_params.parameters()),    'lr': self.params_trunk_lr,    'weight_decay': self.params_trunk_wd,    'name': 'params_trunk'},
            {'params': list(model.trunk_existence.parameters()), 'lr': self.existence_trunk_lr, 'weight_decay': self.existence_trunk_wd, 'name': 'existence_trunk'},
            {'params': model.head_parameters(),                  'lr': self.output_head_lr,     'weight_decay': self.output_head_wd,     'name': 'output_head'},
        ] if len(g['params']) > 0]
