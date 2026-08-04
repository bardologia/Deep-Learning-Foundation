from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class GammaNetConfig:
    n_iterations     : int   = 8
    prox_hidden      : int   = 16
    prox_kernel_size : int   = 5
    step_init        : float = 1.0
    threshold_init   : float = 0.01
    activation       : str   = "relu"

    steps_lr : float = 1e-3
    prox_lr  : float = 1e-3

    steps_wd : float = 0.0
    prox_wd  : float = 1e-4

    shape_logger_types: tuple = field(default_factory=lambda: (
        nn.Conv1d, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.SiLU,
    ))

    @classmethod
    def tunable_lr_params(cls) -> dict:
        return {
            "steps_lr" : {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "prox_lr"  : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "steps_wd" : {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
            "prox_wd"  : {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        }

    @classmethod
    def tunable_arch_params(cls) -> dict:
        return {
            "n_iterations"     : {"type": "categorical", "choices": [4, 8, 12, 16]},
            "prox_hidden"      : {"type": "categorical", "choices": [8, 16, 32]},
            "prox_kernel_size" : {"type": "categorical", "choices": [3, 5, 7]},
            "activation"       : {"type": "categorical", "choices": ["relu", "leaky_relu", "gelu", "silu"]},
        }

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        return [g for g in [
            {'params': [model.raw_steps, model.raw_thresholds],  'lr': self.steps_lr, 'weight_decay': self.steps_wd, 'name': 'steps'},
            {'params': list(model.prox_blocks.parameters()),     'lr': self.prox_lr,  'weight_decay': self.prox_wd,  'name': 'prox'},
        ] if len(g['params']) > 0]
