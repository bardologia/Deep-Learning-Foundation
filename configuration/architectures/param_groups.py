"""Validation helper for optimiser parameter-group construction."""

from __future__ import annotations

import torch.nn as nn


class ParamGroups:
    """Checks that hand-written parameter groups cover a model completely."""

    @staticmethod
    def verified(model: nn.Module, groups: list[dict]) -> list[dict]:
        """Returns the non-empty parameter groups after asserting full model coverage.

        Args:
            model  : Model whose named parameters must all appear in a group.
            groups : Candidate optimiser parameter-group dicts, each with a ``params`` list.

        Returns:
            The input groups with the empty ones removed.

        Raises:
            ValueError : If any model parameter is absent from every group.
        """
        grouped = {id(p) for g in groups for p in g["params"]}
        missing = [name for name, p in model.named_parameters() if id(p) not in grouped]

        if missing:
            raise ValueError(f"get_param_groups does not cover every model parameter. Uncovered: {missing}")

        return [g for g in groups if len(g["params"]) > 0]
