
from __future__ import annotations

from enum    import Enum
from pathlib import Path


class ConfigValueRenderer:

    @classmethod
    def clean(cls, value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {cls.clean(key): cls.clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.clean(item) for item in value]
        return value

    @classmethod
    def render(cls, value) -> str:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return str(value.value)
        if isinstance(value, (tuple, list, dict)):
            return str(cls.clean(value))
        if value is None:
            return "None"
        return str(value)
