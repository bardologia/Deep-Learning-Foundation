"""Tests for how the console reads the machine profile.

Covers the interpreter preference, browse roots and conda scan of ProjectPaths,
the GPU-schedule and ntfy seeds, and the launch-form resolver marking the leaves
the profile rewrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gpu_schedule           import GpuSchedule
from notifier               import JobNotifier
from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver
from web_logger             import WebLogger

from configuration.reference_layout import ReferenceLayout
from tests.webui.conftest           import REPO_ROOT, StubPaths
from tools.runtime.machine_profile  import MachineProfile


def _profile(tmp_path: Path, **sections) -> MachineProfile:
    """Returns a saved profile holding the given sections."""
    profile = MachineProfile(tmp_path / "machine.json", {"schema": MachineProfile.SCHEMA, **sections})
    profile.save()

    return MachineProfile.load(profile.path)


def test_project_paths_prefer_the_profile_environment(tmp_path):
    """The launch interpreter preference follows the profile's environment name for every script."""
    profile = _profile(tmp_path, machine={"env_name": "ProjectEnv"})
    paths   = ProjectPaths(REPO_ROOT, profile)
    found   = [{"label": "conda:base", "path": "/c/bin/python", "current": False}, {"label": "conda:Dune", "path": "/c/envs/d/bin/python", "current": False}, {"label": "conda:ProjectEnv", "path": "/c/envs/p/bin/python", "current": False}]

    assert paths.priority                                       == ["conda:ProjectEnv"]
    assert paths.preferred_interpreter(found)                   == "/c/envs/p/bin/python"
    assert paths.preferred_interpreter(found, "train_backbone") == "/c/envs/p/bin/python"
    assert ProjectPaths(REPO_ROOT, MachineProfile(None)).priority == ["conda:Dune"]


def test_project_paths_take_the_runs_and_data_roots_from_the_profile(tmp_path):
    """The runs tree and the browse roots move to the profile's roots, and fall back to the checkout without one."""
    profile = _profile(tmp_path, roots={"runs": str(tmp_path / "scratch" / "runs"), "data": str(tmp_path / "data")})

    with_profile = ProjectPaths(REPO_ROOT, profile)
    without      = ProjectPaths(REPO_ROOT, MachineProfile(None))

    assert with_profile.runs_dir  == tmp_path / "scratch" / "runs"
    assert with_profile.roots()   == {"datasets": str(tmp_path / "data"), "runs": str(tmp_path / "scratch" / "runs")}
    assert without.runs_dir       == REPO_ROOT / ReferenceLayout.RUNS_ROOT
    assert without.roots()        == {"datasets": str(REPO_ROOT / ReferenceLayout.DATA_ROOT), "runs": str(REPO_ROOT / ReferenceLayout.RUNS_ROOT)}
    assert with_profile.profile_signature()
    assert without.profile_signature() == ()


def test_project_paths_scan_the_profile_conda_root_first(tmp_path, monkeypatch):
    """An interpreter under the profile's conda root is discovered even outside the home directory."""
    root = tmp_path / "opt" / "conda"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").write_text("")
    (root / "envs" / "ProjectEnv" / "bin").mkdir(parents=True)
    (root / "envs" / "ProjectEnv" / "bin" / "python").write_text("")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))

    profile = _profile(tmp_path, machine={"env_name": "ProjectEnv", "conda_root": str(root)})
    labels  = {item["label"]: item["path"] for item in ProjectPaths(REPO_ROOT, profile).discover_interpreters()}

    assert labels["conda:base"]       == str(root / "bin" / "python")
    assert labels["conda:ProjectEnv"] == str(root / "envs" / "ProjectEnv" / "bin" / "python")


def test_gpu_schedule_seeds_every_pool_from_the_profile(tmp_path):
    """The schedule defaults take the profile's GPU pool when it names one and keep the built-ins otherwise."""
    seeded = GpuSchedule.seed(_profile(tmp_path, hardware={"gpu_pool": [0, 1]}))
    plain  = GpuSchedule.seed(MachineProfile(None))

    assert seeded["weekday_gpus"] == [0, 1]
    assert seeded["night_gpus"]   == [0, 1]
    assert seeded["weekend_gpus"] == [0, 1]
    assert seeded["enabled"]      is False
    assert plain                  == GpuSchedule.DEFAULTS


def test_notifier_seeds_its_settings_from_the_profile(tmp_path):
    """Without a settings file the notifier starts from the profile's ntfy section; a saved file still wins."""
    paths         = StubPaths(tmp_path)
    paths.profile = _profile(tmp_path, ntfy={"enabled": True, "topic": "my-topic", "server": "https://ntfy.example"})
    notifier      = JobNotifier(paths, WebLogger())

    assert notifier.settings == {"enabled": True, "topic": "my-topic", "server": "https://ntfy.example"}

    notifier.configure({"enabled": False, "topic": "other", "server": "https://ntfy.sh"})
    reloaded = JobNotifier(paths, WebLogger())

    assert reloaded.settings["topic"] == "other"
    assert JobNotifier.DEFAULTS       == MachineProfile.DEFAULTS["ntfy"]


def test_the_resolver_marks_the_leaves_the_profile_rewrote(tmp_path, monkeypatch):
    """Resolved leaves carry a source of profile for rewritten defaults and default for the rest."""
    profile = _profile(tmp_path, roots={"runs": str(tmp_path / "runs")}, dataset={"path": str(tmp_path / "scene")}, hardware={"gpu_pool": [1]})
    monkeypatch.setenv(MachineProfile.ENV_VAR, str(profile.path))

    result = ScriptConfigResolver(ProjectPaths(REPO_ROOT, profile)).resolve("train_backbone", sys.executable)

    assert result["ok"], result.get("error")

    leaves = {leaf["path"]: leaf for leaf in result["leaves"]}

    assert leaves["paths.dataset_path"]["source"] == "profile"
    assert leaves["paths.dataset_path"]["value"]  == str(tmp_path / "scene")
    assert leaves["logdir"]["source"]             == "profile"
    assert leaves["logdir"]["value"]              == str(tmp_path / "runs" / "backbone")
    assert leaves["gpus"]["source"]               == "profile"
    assert leaves["training.epochs"]["source"]    == "default"
