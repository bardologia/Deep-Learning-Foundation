
from __future__ import annotations

import re
import tomllib
from pathlib import Path

IMPORT_NAMES = {"pillow": "PIL", "scikit-image": "skimage"}
SOURCE_ROOTS = ("main", "pipelines", "tools", "models", "configuration", "webui", "scripts", "tests")


def _pyproject_dependencies(repo_root: Path) -> list[str]:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]


def _requirements(repo_root: Path) -> list[str]:
    lines = (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()

    return [line.strip() for line in lines if line.strip()]


def _distribution(requirement: str) -> str:
    return re.split(r"[=<>!~\[]", requirement, maxsplit=1)[0].strip()


def _import_name(requirement: str) -> str:
    distribution = _distribution(requirement)

    return IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))


def _imported_modules(repo_root: Path) -> set[str]:
    pattern = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    modules = set()

    for root in SOURCE_ROOTS:
        for path in (repo_root / root).rglob("*.py"):
            if "__pycache__" in path.parts or "node_modules" in path.parts:
                continue
            modules.update(pattern.findall(path.read_text(encoding="utf-8")))

    return modules


def test_requirements_mirror_pyproject_dependencies(repo_root):
    assert _requirements(repo_root) == _pyproject_dependencies(repo_root)


def test_every_dependency_is_pinned_exactly(repo_root):
    unpinned = [requirement for requirement in _pyproject_dependencies(repo_root) if "==" not in requirement]

    assert unpinned == []


def test_every_dependency_is_imported_somewhere(repo_root):
    imported = _imported_modules(repo_root)
    unused   = [_distribution(requirement) for requirement in _pyproject_dependencies(repo_root) if _import_name(requirement) not in imported]

    assert unused == []


def test_unused_dependency_is_reported(repo_root):
    imported = _imported_modules(repo_root)

    assert _import_name("scikit-learn==1.8.0") not in imported
    assert _import_name("pillow==12.1.1")      in imported
