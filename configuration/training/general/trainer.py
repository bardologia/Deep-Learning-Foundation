"""Mixin that copies the shared trainer sub-configs from a base configuration."""

from __future__ import annotations

_SHARED_SUBCONFIGS = (
    "early_stopping", "warmup", "scheduler", "io", "optimizer",
    "training", "resources", "gradient_clipper", "memory",
)


class SharedSubConfigInheritance:
    """Gives a trainer config the geometry, optimization and runtime blocks of a base."""

    def inherit_shared_from(self, base) -> None:
        """Copies the shared sub-config objects from `base` onto this configuration.

        Args:
            base : Configuration holding the geometry, early-stopping, warmup, scheduler, io, optimizer, training, resources, gradient-clipper and memory blocks to adopt by reference.
        """
        for name in _SHARED_SUBCONFIGS:
            setattr(self, name, getattr(base, name))
