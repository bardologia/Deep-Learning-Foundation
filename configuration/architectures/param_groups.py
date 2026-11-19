
from __future__ import annotations

import torch.nn as nn


class ParamGroups:

    @staticmethod
    def verified(model: nn.Module, groups: list[dict]) -> list[dict]:
        grouped = {id(p) for g in groups for p in g["params"]}
        missing = [name for name, p in model.named_parameters() if id(p) not in grouped]

        if missing:
            raise ValueError(f"get_param_groups does not cover every model parameter. Uncovered: {missing}")

        return [g for g in groups if len(g["params"]) > 0]
