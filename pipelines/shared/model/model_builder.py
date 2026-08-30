"""Naming and configuration helpers for backbone model instantiation.

Encodes the ``<backbone>-<head>`` model-key convention shared by the training,
benchmark and tuning pipelines, and materialises dataclass model configs from
the backbone registry with user-supplied overrides applied.
"""

from __future__ import annotations

from dataclasses import fields

from models import BACKBONE_CONFIG_REGISTRY, BACKBONE_IMAGE_SIZE_MODELS


class ModelBuilder:
    """Builds model keys, patch overrides and model configs for backbones.

    Attributes:
        GROUP_FANOUTS : Mapping from an override key that sets a whole group of per-group fields uniformly to the field-name suffix it fans out to.
    """

    GROUP_FANOUTS = {"all_groups_lr": "_lr", "all_groups_wd": "_wd"}

    @staticmethod
    def model_key(model_name: str, head: str) -> str:
        """Returns the composite model key, omitting the default ``conv`` head."""
        return model_name if head == "conv" else f"{model_name}-{head}"

    @staticmethod
    def image_size_override(model_name: str, patch_size) -> dict:
        """Returns the ``image_size`` override required by fixed-input backbones.

        Args:
            model_name : Backbone name, without the head suffix.
            patch_size : Patch extent as (height, width) in pixels.

        Returns:
            A dict holding ``image_size`` set to the larger patch side, or an
            empty dict when the backbone accepts arbitrary input sizes.
        """
        if model_name not in BACKBONE_IMAGE_SIZE_MODELS:
            return {}

        return {"image_size": int(max(patch_size))}

    @staticmethod
    def square_patch(model_names, size, stride) -> tuple[tuple[int, int], tuple[int, int]]:
        """Squares the patch geometry when any listed backbone needs square input.

        Args:
            model_names : Backbone names participating in the run.
            size        : Patch extent as (height, width) in pixels.
            stride      : Patch stride as (height, width) in pixels.

        Returns:
            The (size, stride) pair unchanged, or both squared to their larger
            side when at least one backbone requires a fixed square input.
        """
        if not any(name in BACKBONE_IMAGE_SIZE_MODELS for name in model_names):
            return tuple(size), tuple(stride)

        side = int(max(size))
        step = int(max(stride))

        return (side, side), (step, step)

    @staticmethod
    def split_key(model_key: str) -> tuple[str, str]:
        """Splits a model key into (backbone name, head), defaulting to ``conv``."""
        name, sep, head = model_key.partition("-")
        return (name, head) if sep else (name, "conv")

    @staticmethod
    def config_from_registry(model_name: str, model_overrides: dict, head: str | None = None, registry: dict | None = None):
        """Instantiates a model config from the registry with overrides applied.

        Group fan-out keys (``all_groups_lr``, ``all_groups_wd``) are expanded
        onto every config field carrying the matching suffix before the
        remaining overrides are set field by field.

        Args:
            model_name      : Backbone name, without the head suffix.
            model_overrides : Field name to value mapping applied to the config.
            head            : Head identifier written into the config's ``head`` field, or None to leave the registry default in place.
            registry        : Backbone name to config class mapping; defaults to BACKBONE_CONFIG_REGISTRY.

        Returns:
            The instantiated model config dataclass with overrides applied.

        Raises:
            ValueError     : If the head is passed both explicitly and through ``model_overrides``.
            AttributeError : If a group fan-out key matches no field, or an override names a field the config does not have.
        """
        registry = BACKBONE_CONFIG_REGISTRY if registry is None else registry
        config   = registry[model_name]()

        if head is not None:
            if "head" in model_overrides:
                raise ValueError("Select the head via the dedicated head field, not model_overrides['head']")
            model_overrides = {"head": head, **model_overrides}

        for key, suffix in ModelBuilder.GROUP_FANOUTS.items():
            if key not in model_overrides:
                continue

            model_overrides = dict(model_overrides)
            uniform         = model_overrides.pop(key)
            group_fields    = [spec.name for spec in fields(config) if spec.name.endswith(suffix)]

            if not group_fields:
                raise AttributeError(f"Model override '{key}' found no *{suffix} fields on {type(config).__name__}")

            for name in group_fields:
                setattr(config, name, uniform)

        for attribute, value in model_overrides.items():
            if not hasattr(config, attribute):
                raise AttributeError(f"Unknown model override '{attribute}' for {type(config).__name__}")
            setattr(config, attribute, value)

        return config
