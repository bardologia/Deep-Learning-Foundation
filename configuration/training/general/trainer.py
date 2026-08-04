from __future__ import annotations

_SHARED_SUBCONFIGS = (
    "geometry", "early_stopping", "warmup", "scheduler", "io", "optimizer",
    "training", "resources", "gradient_clipper", "memory",
)


class SharedSubConfigInheritance:
    def inherit_shared_from(self, base) -> None:
        for name in _SHARED_SUBCONFIGS:
            setattr(self, name, getattr(base, name))
