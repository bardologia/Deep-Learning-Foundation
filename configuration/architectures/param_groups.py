
from __future__ import annotations

import torch.nn as nn


class ParamGroups:

    @staticmethod
    def verified(model: nn.Module, groups: list[dict]) -> list[dict]:
        return [g for g in groups if len(g["params"]) > 0]
