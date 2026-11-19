
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT   = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from extras_requirements import ExtrasRequirementsWriter


def _pyproject(tmp_path: Path, extras: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(f"[project]\nname = \"demo\"\n\n[project.optional-dependencies]\n{extras}", encoding="utf-8")

    return path


def test_the_groups_are_flattened_into_one_sorted_deduplicated_file(tmp_path):
    pyproject = _pyproject(tmp_path, 'dev = ["pytest==8.0", "pyflakes==3.4.0"]\nexport = ["onnx==1.22.0", "pytest==8.0"]\n')

    written = ExtrasRequirementsWriter(pyproject, tmp_path / "out" / "extras.txt").build()

    assert written.read_text(encoding="utf-8") == "onnx==1.22.0\npyflakes==3.4.0\npytest==8.0\n"


def test_a_project_without_optional_dependencies_is_refused(tmp_path):
    pyproject = _pyproject(tmp_path, "")

    with pytest.raises(ValueError, match="declares no optional-dependencies"):
        ExtrasRequirementsWriter(pyproject, tmp_path / "extras.txt").build()


def test_the_command_line_takes_the_pyproject_and_the_output_path(tmp_path):
    pyproject = _pyproject(tmp_path, 'dev = ["pytest==8.0"]\n')
    output    = tmp_path / "extras.txt"

    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / "extras_requirements.py"), str(pyproject), str(output)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "pytest==8.0\n"
