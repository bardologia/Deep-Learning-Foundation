"""Gradient checkpointing of named submodules, transparent to checkpoints.

`CheckpointedModule` wraps one submodule so its forward runs under
`torch.utils.checkpoint`, trading activation memory for a second forward pass
in backward. The wrapper rewrites its state-dict keys so saved checkpoints and
the state-dict baseline are byte-identical to an unwrapped model, and
`ActivationCheckpointing` installs wrappers over a model by fnmatch patterns
against dotted submodule names.
"""

from __future__ import annotations

from fnmatch import fnmatch

import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedModule(torch.nn.Module):
    """Runs a wrapped submodule under gradient checkpointing.

    Checkpointing only engages in training mode with gradients enabled;
    evaluation and no-grad passes run the submodule directly. State-dict keys
    are rewritten on save and load so the wrapper never appears in them.

    Attributes:
        PREFIX : Key prefix the wrapper strips from its state dict.
        module : The wrapped submodule.
    """

    PREFIX = "module."

    def __init__(self, module: torch.nn.Module) -> None:
        """Wraps the submodule and installs the key-rewriting hooks."""
        super().__init__()
        self.module = module

        self._register_state_dict_hook(self._strip_prefix)
        self._register_load_state_dict_pre_hook(self._insert_prefix, with_module=True)

    @staticmethod
    def _strip_prefix(module, state_dict, prefix, local_metadata) -> dict:
        """Renames every saved key from '<prefix>module.' back to '<prefix>'."""
        marker = prefix + CheckpointedModule.PREFIX
        for key in [key for key in state_dict if key.startswith(marker)]:
            state_dict[prefix + key[len(marker):]] = state_dict.pop(key)

        return state_dict

    @staticmethod
    def _insert_prefix(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        """Renames every incoming key from '<prefix>' to '<prefix>module.' before loading."""
        marker = prefix + CheckpointedModule.PREFIX
        for key in [key for key in state_dict if key.startswith(prefix) and not key.startswith(marker)]:
            state_dict[marker + key[len(prefix):]] = state_dict.pop(key)

    def forward(self, *args, **kwargs):
        """Runs the submodule, under gradient checkpointing when training with gradients."""
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.module, *args, use_reentrant=False, **kwargs)

        return self.module(*args, **kwargs)


class ActivationCheckpointing:
    """Installs gradient-checkpoint wrappers over a model by name pattern."""

    @staticmethod
    def _expand_containers(model: torch.nn.Module, name: str) -> list[str]:
        """Returns the name itself, or the wrappable children of a bare container.

        ModuleList and ModuleDict have no forward and are iterated by their
        parent, so wrapping them would break that iteration; their children are
        wrapped instead.
        """
        module = model.get_submodule(name)

        if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            expanded: list[str] = []
            for child_name, _child in module.named_children():
                expanded.extend(ActivationCheckpointing._expand_containers(model, f"{name}.{child_name}"))
            return expanded

        return [name]

    @classmethod
    def matching_names(cls, model: torch.nn.Module, patterns: tuple[str, ...]) -> list[str]:
        """Returns the dotted submodule names matched by any pattern, shallowest first.

        A name nested inside an already-matched module is dropped so wrappers
        never nest, and a matched ModuleList or ModuleDict is expanded into its
        children, which are the modules the parent forward actually calls.

        Args:
            model    : Model whose named_modules are matched.
            patterns : fnmatch patterns against dotted submodule names.

        Returns:
            The matched names in registration order.
        """
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
        """Replaces every matched submodule with a checkpointed wrapper, in place.

        Args:
            model    : Model rewritten in place.
            patterns : fnmatch patterns against dotted submodule names.
            logger   : Logger receiving the wrapped-module line.

        Returns:
            The dotted names that were wrapped.

        Raises:
            ValueError : If a pattern matches no submodule, naming the available top-level modules.
        """
        names = cls.matching_names(model, patterns)

        all_names = [name for name, _module in model.named_modules() if name]
        unmatched = [pattern for pattern in patterns if not any(fnmatch(name, pattern) for name in all_names)]
        if unmatched:
            available = sorted({name.split(".")[0] for name, _ in model.named_modules() if name})
            raise ValueError(f"activation_checkpointing patterns {unmatched} match no submodule; top-level modules are {available}")

        for name in names:
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, CheckpointedModule(getattr(parent, child_name)))

        logger.info(f"Activation checkpointing wraps {len(names)} module(s): {names}")

        return names
