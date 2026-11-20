
from __future__ import annotations

from dataclasses import fields

from models import BACKBONE_CONFIG_REGISTRY, BACKBONE_IMAGE_SIZE_MODELS


class ModelBuilder:

    GROUP_FANOUTS = {"all_groups_lr": "_lr", "all_groups_wd": "_wd"}

    @staticmethod
    def model_key(model_name: str, head: str) -> str:
        return model_name if head == "conv" else f"{model_name}-{head}"

    @staticmethod
    def image_size_override(model_name: str, patch_size) -> dict:
        if model_name not in BACKBONE_IMAGE_SIZE_MODELS:
            return {}

        return {"image_size": int(max(patch_size))}

    @staticmethod
    def square_patch(model_name: str, size, stride) -> tuple[tuple[int, int], tuple[int, int]]:
        if model_name not in BACKBONE_IMAGE_SIZE_MODELS:
            return tuple(size), tuple(stride)

        side = int(max(size))
        step = int(max(stride))

        return (side, side), (step, step)

    @staticmethod
    def split_key(model_key: str) -> tuple[str, str]:
        name, sep, head = model_key.partition("-")
        return (name, head) if sep else (name, "conv")

    @staticmethod
    def config_from_registry(model_name: str, model_overrides: dict, head: str | None = None, registry: dict | None = None):
        registry = BACKBONE_CONFIG_REGISTRY if registry is None else registry
        config   = registry[model_name]()

        if head is not None:
            model_overrides = {"head": head, **model_overrides}

        for key, suffix in ModelBuilder.GROUP_FANOUTS.items():
            if key not in model_overrides:
                continue

            model_overrides = dict(model_overrides)
            uniform         = model_overrides.pop(key)
            group_fields    = [spec.name for spec in fields(config) if spec.name.endswith(suffix)]

            for name in group_fields:
                setattr(config, name, uniform)

        for attribute, value in model_overrides.items():
            setattr(config, attribute, value)

        return config
