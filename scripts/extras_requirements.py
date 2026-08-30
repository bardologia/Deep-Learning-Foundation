"""Flattening of pyproject optional-dependency groups into a requirements file."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


class ExtrasRequirementsWriter:
    """Flattens the optional-dependency groups of a pyproject into one requirements file.

    Attributes:
        pyproject_path : Source pyproject.toml.
        output_path    : Requirements file that is written.
    """

    def __init__(self, pyproject_path: Path, output_path: Path) -> None:
        """Binds the writer to its input and output paths.

        Args:
            pyproject_path : Source pyproject.toml.
            output_path    : Requirements file to write.
        """
        self.pyproject_path = Path(pyproject_path)
        self.output_path    = Path(output_path)

    def read_extras(self) -> dict[str, list[str]]:
        """Reads the optional-dependency groups of the pyproject.

        Returns:
            Mapping of extra name to its requirement specifiers.

        Raises:
            ValueError : If the project declares no optional dependencies.
        """
        metadata = tomllib.loads(self.pyproject_path.read_text(encoding="utf-8"))
        extras   = metadata["project"]["optional-dependencies"]

        if not extras:
            raise ValueError(f"{self.pyproject_path} declares no optional-dependencies to install")

        return extras

    def flatten(self, extras: dict[str, list[str]]) -> list[str]:
        """Returns the deduplicated, sorted union of every group's requirements.

        Args:
            extras : Mapping of extra name to requirement specifiers.

        Returns:
            Sorted requirement specifiers.
        """
        return sorted({requirement for group in extras.values() for requirement in group})

    def write(self, requirements: list[str]) -> Path:
        """Writes one requirement per line to the output path.

        Args:
            requirements : Requirement specifiers in output order.

        Returns:
            Path of the written file.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

        return self.output_path

    def build(self) -> Path:
        """Reads, flattens and writes the extras.

        Returns:
            Path of the written requirements file.
        """
        extras = self.read_extras()
        return self.write(self.flatten(extras))


if __name__ == "__main__":
    ExtrasRequirementsWriter(Path(sys.argv[1]), Path(sys.argv[2])).build()
