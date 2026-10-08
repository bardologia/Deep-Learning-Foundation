"""Tests for the headless visual harness under ``webui/tools/visual``.

The harness README tells the user to run ``npm ci``, which installs only from a
tracked ``package-lock.json``. These tests check that the lock file is shipped,
is not swallowed by the repository-wide ``package-lock.json`` ignore rule, and
pins the same ``playwright-core`` release as ``package.json``; that every node
script the README runs is shipped; and that the snapshot harnesses default to
the console port.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from configuration.reference_layout import ReferenceLayout

from tests.webui.conftest import REPO_ROOT, WEBUI_ROOT


VISUAL_DIR = WEBUI_ROOT / "tools" / "visual"


def test_the_lock_file_pins_the_package_manifest():
    """package-lock.json names the harness package and records the playwright-core release package.json pins."""
    manifest = json.loads((VISUAL_DIR / "package.json").read_text())
    lock     = json.loads((VISUAL_DIR / "package-lock.json").read_text())

    assert lock["name"] == manifest["name"]
    assert lock["packages"][""]["dependencies"] == manifest["dependencies"]
    assert lock["packages"]["node_modules/playwright-core"]["version"] == manifest["dependencies"]["playwright-core"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required to evaluate the ignore rules")
def test_the_lock_file_is_not_ignored():
    """The repository ignore rules exempt the harness lock file while still ignoring node_modules."""
    lock    = (VISUAL_DIR / "package-lock.json").relative_to(REPO_ROOT)
    modules = (VISUAL_DIR / "node_modules" / "playwright-core").relative_to(REPO_ROOT)

    assert subprocess.run(["git", "check-ignore", "-q", "--no-index", str(lock)], cwd=REPO_ROOT).returncode == 1
    assert subprocess.run(["git", "check-ignore", "-q", "--no-index", str(modules)], cwd=REPO_ROOT).returncode == 0


def test_every_script_the_readme_runs_is_shipped():
    """Each ``node <script>.js`` command in the harness README names a script present in the harness directory."""
    scripts = set(re.findall(r"^node (\w+\.js)", (VISUAL_DIR / "README.md").read_text(), re.M))

    assert "snap_experiments.js" in scripts
    assert sorted(name for name in scripts if not (VISUAL_DIR / name).is_file()) == []


def test_snapshot_harnesses_default_to_the_console_port():
    """Every browser harness that drives a live server defaults to the console port, and the README examples use it."""
    for path in VISUAL_DIR.glob("snap_*.js"):
        assert re.search(r"this\.port\s+=\s+(\d+);", path.read_text()).group(1) == str(ReferenceLayout.CONSOLE_PORT), path.name

    assert set(re.findall(r"--port (\d+)", (VISUAL_DIR / "README.md").read_text())) == {str(ReferenceLayout.CONSOLE_PORT)}
