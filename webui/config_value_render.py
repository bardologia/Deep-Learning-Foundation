"""Rendering of one configuration leaf into the string the launch form compares.

A single rule with three consumers: the resolver bootstrap that instantiates an
entry's configuration under the target interpreter and prints its defaults, the
run-config reader that loads a finished run's saved configuration back, and the
tests that pin the two against each other. The module deliberately imports
nothing from the project so the bootstrap subprocess can import it with only the
repository root on ``sys.path``.
"""

from __future__ import annotations

from enum    import Enum
from pathlib import Path


class ConfigValueRenderer:
    """Renders configuration values as the launch form's comparison strings."""

    @classmethod
    def clean(cls, value):
        """Returns a value with enums and paths reduced to their plain contents.

        Args:
            value : Configuration value, possibly a nested container.

        Returns:
            The same structure with every Enum replaced by its value, every Path
            by its string form and every tuple by a list.
        """
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
        """Returns one configuration leaf rendered as its launch-form string.

        None renders as ``"None"``, paths and enums as their plain contents,
        containers as the Python literal of their cleaned form, and every other
        scalar through ``str``. Values decoded from a saved run configuration
        render identically to the live dataclass values they came from, which is
        what keeps a copied run free of phantom overrides.

        Args:
            value : Configuration value, live from a dataclass or JSON-decoded from a run's resolved configuration file.

        Returns:
            The rendered string.
        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return str(value.value)
        if isinstance(value, (tuple, list, dict)):
            return str(cls.clean(value))
        if value is None:
            return "None"
        return str(value)
