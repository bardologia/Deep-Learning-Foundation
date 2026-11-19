
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


class ExtrasRequirementsWriter:

    def __init__(self, pyproject_path: Path, output_path: Path) -> None:
        self.pyproject_path = Path(pyproject_path)
        self.output_path    = Path(output_path)

    def read_extras(self) -> dict[str, list[str]]:
        metadata = tomllib.loads(self.pyproject_path.read_text(encoding="utf-8"))
        extras   = metadata["project"]["optional-dependencies"]

        if not extras:
            raise ValueError(f"{self.pyproject_path} declares no optional-dependencies to install")

        return extras

    def flatten(self, extras: dict[str, list[str]]) -> list[str]:
        return sorted({requirement for group in extras.values() for requirement in group})

    def write(self, requirements: list[str]) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

        return self.output_path

    def build(self) -> Path:
        extras = self.read_extras()
        return self.write(self.flatten(extras))


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten the optional-dependency groups of a pyproject into one requirements file, one deduplicated requirement per line in sorted order.")
    parser.add_argument("pyproject_path", type=Path, help="pyproject.toml whose optional dependencies are read.")
    parser.add_argument("output_path", type=Path, help="Requirements file to write; its parent is created.")
    args = parser.parse_args()

    ExtrasRequirementsWriter(args.pyproject_path, args.output_path).build()


if __name__ == "__main__":
    main()
