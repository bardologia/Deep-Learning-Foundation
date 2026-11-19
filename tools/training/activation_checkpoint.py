
from __future__ import annotations

from fnmatch import fnmatch

import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedModule(torch.nn.Module):

    PREFIX = "module."

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

        self._register_state_dict_hook(self._strip_prefix)
        self._register_load_state_dict_pre_hook(self._insert_prefix, with_module=True)

    @staticmethod
    def _strip_prefix(module, state_dict, prefix, local_metadata) -> dict:
        marker = prefix + CheckpointedModule.PREFIX
        for key in [key for key in state_dict if key.startswith(marker)]:
            state_dict[prefix + key[len(marker):]] = state_dict.pop(key)

        return state_dict

    @staticmethod
    def _insert_prefix(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        marker = prefix + CheckpointedModule.PREFIX
        for key in [key for key in state_dict if key.startswith(prefix) and not key.startswith(marker)]:
            state_dict[marker + key[len(prefix):]] = state_dict.pop(key)

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.module, *args, use_reentrant=False, **kwargs)

        return self.module(*args, **kwargs)


class ActivationCheckpointing:

    @staticmethod
    def _expand_containers(model: torch.nn.Module, name: str) -> list[str]:
        module = model.get_submodule(name)

        if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            expanded: list[str] = []
            for child_name, _child in module.named_children():
                expanded.extend(ActivationCheckpointing._expand_containers(model, f"{name}.{child_name}"))
            return expanded

        return [name]

    @classmethod
    def matching_names(cls, model: torch.nn.Module, patterns: tuple[str, ...]) -> list[str]:
        matched: list[str] = []
        for name, _module in model.named_modules():
            if not name or not any(fnmatch(name, pattern) for pattern in patterns):
                continue
            if any(name == kept or name.startswith(f"{kept}.") for kept in matched):
                continue
            matched.append(name)

        expanded: list[str] = []
        for name in matched:
            expanded.extend(cls._expand_containers(model, name))

        return expanded

    @classmethod
    def apply(cls, model: torch.nn.Module, patterns: tuple[str, ...], logger) -> list[str]:
        names = cls.matching_names(model, patterns)

        for name in names:
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, CheckpointedModule(getattr(parent, child_name)))

        logger.info(f"Activation checkpointing wraps {len(names)} module(s): {names}")

        return names
