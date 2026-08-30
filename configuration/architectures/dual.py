"""Architecture configuration for the dual-trunk backbone family.

A dual model runs two independent trunks over the same tomographic patch: one
predicting Gaussian scatterer parameters, one predicting scatterer existence,
each free to use its own backbone family, input representation and channel
subset before a shared output head combines them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class DualResUNetConfig:
    """Configuration of the two-trunk model with separate parameter and existence branches.

    Attributes:
        in_channels         : Channels of the assembled input stack.
        out_channels        : Number of predicted values per pixel.
        params_per_gaussian : Values describing a single Gaussian scatterer.
        head                : Name of the shared output head fed by both trunks.
        params_backbone     : Backbone family name of the parameter trunk.
        existence_backbone  : Backbone family name of the existence trunk.
        params_input        : Input representations concatenated for the parameter trunk.
        existence_input     : Input representations concatenated for the existence trunk.
        params_channels     : Indices of the input stack the parameter trunk consumes.
        existence_channels  : Indices of the input stack the existence trunk consumes.
        params_features     : Stage widths of the parameter trunk.
        existence_features  : Stage widths of the existence trunk.
        params_overrides    : Extra backbone config fields applied to the parameter trunk.
        existence_overrides : Extra backbone config fields applied to the existence trunk.
        head_activation     : Activation used inside the output head.
        init_mode           : Weight initialisation scheme name.
        shape_logger_types  : Module types whose output shapes are logged.
    """

    in_channels         : int       = 9
    out_channels        : int       = 6
    params_per_gaussian : int       = 3
    head                : str       = "set_pred"
    params_backbone     : str       = "unet_skip"
    existence_backbone  : str       = "unet_skip"
    params_input        : tuple     = ("chan", "derived", "aux")
    existence_input     : tuple     = ("chan", "derived", "aux")
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
        """Returns the tuner search space for per-trunk learning rates and weight decays.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
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
        """Returns the tuner search space for the trunk families and their stage widths.

        Returns:
            Mapping from field name to an Optuna-style sampling specification.
        """
        return {
            "params_backbone"    : {"type": "categorical",         "choices": ["unet_skip", "resunet", "unet"]},
            "existence_backbone" : {"type": "categorical",         "choices": ["unet_skip", "resunet", "unet"]},
            "params_features"    : {"type": "indexed_categorical", "choices": [[48, 96, 184, 352], [64, 128, 248, 472]]},
            "existence_features" : {"type": "indexed_categorical", "choices": [[48, 96, 184, 352], [24, 48, 84, 156]]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        """Returns optimiser parameter groups for the two trunks and the shared head.

        Args:
            model : Instantiated dual model exposing both trunks and the head.

        Returns:
            List of optimiser parameter-group dicts, each carrying ``params``,
            ``lr``, ``weight_decay`` and a ``name``; empty groups are dropped.
        """
        return [g for g in [
            {'params': list(model.trunk_params.parameters()),    'lr': self.params_trunk_lr,    'weight_decay': self.params_trunk_wd,    'name': 'params_trunk'},
            {'params': list(model.trunk_existence.parameters()), 'lr': self.existence_trunk_lr, 'weight_decay': self.existence_trunk_wd, 'name': 'existence_trunk'},
            {'params': model.head_parameters(),                  'lr': self.output_head_lr,     'weight_decay': self.output_head_wd,     'name': 'output_head'},
        ] if len(g['params']) > 0]
