"""Tests the revision provenance recorded beside run outputs.

Covers how GitState judges a checkout clean or dirty, the working-tree state
saved with a dirty run, the unavailable report outside a checkout, the unknown
tree reported when git status times out, and the fields RunProvenance gathers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.runtime.git_state import GitState, RunProvenance


def _git(repo, *args):
    """Runs one git command with a throwaway identity inside the given repository."""
    subprocess.run(["git", "-c", "user.email=test@example.org", "-c", "user.name=test", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """Returns a one-commit git repository that GitState is pointed at."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.py").write_text("value = 1\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.setattr(GitState, "REPO_ROOT", tmp_path)
    return tmp_path


def test_commit_returns_a_hash_or_a_reasoned_unavailable():
    """The commit string of this checkout is either a 12-character hash (annotated or not) or an unavailable notice."""
    commit = GitState.commit()

    assert isinstance(commit, str) and commit
    assert commit.startswith("unavailable") or len(commit.split()[0]) == 12


def test_clean_checkout_reports_the_bare_hash(checkout):
    """Verifies a checkout without changes reports a 12-character hash and nothing else."""
    commit = GitState.commit()

    assert len(commit) == 12
    assert "(" not in commit


def test_untracked_files_do_not_make_the_checkout_dirty(checkout):
    """Verifies untracked outputs inside the checkout leave it clean, and the dirty state still lists them."""
    (checkout / "notes.md").write_text("scratch\n")

    state = GitState.dirty_state()

    assert GitState.DIRTY_SUFFIX not in GitState.commit()
    assert state == {"dirty": False, "diff": "", "untracked": ["notes.md"]}


def test_tracked_changes_make_the_checkout_dirty(checkout):
    """Verifies an uncommitted edit to a tracked file is reported by the commit and carried as a diff by the dirty state."""
    (checkout / "tracked.py").write_text("value = 2\n")

    state = GitState.dirty_state()

    assert GitState.commit().endswith(GitState.DIRTY_SUFFIX)
    assert state["dirty"] is True
    assert "+value = 2" in state["diff"]


def test_dirty_state_contract():
    """The dirty state always carries the dirty flag, the diff text and the untracked list, and a clean tree carries no diff."""
    state = GitState.dirty_state()

    assert set(state) == {"dirty", "diff", "untracked"}
    assert isinstance(state["dirty"], bool)
    assert isinstance(state["diff"], str)
    assert isinstance(state["untracked"], list)

    if not state["dirty"]:
        assert state["diff"] == ""


def test_dirty_state_agrees_with_the_commit_annotation():
    """The dirty flag and the '(dirty working tree)' suffix of commit() tell the same story."""
    commit = GitState.commit()
    state  = GitState.dirty_state()

    if not commit.startswith("unavailable"):
        assert state["dirty"] == commit.endswith(GitState.DIRTY_SUFFIX)


def test_outside_a_checkout_the_commit_is_unavailable(tmp_path, monkeypatch):
    """Verifies a directory that is not a checkout yields an unavailable report naming the reason."""
    monkeypatch.setattr(GitState, "REPO_ROOT", tmp_path)

    assert GitState.commit().startswith("unavailable (")
    assert GitState.dirty_state() == {"dirty": False, "diff": "", "untracked": []}


def test_commit_reads_a_named_checkout(checkout, tmp_path_factory):
    """A checkout other than this repository is read by path with the same clean and dirty annotations."""
    other = tmp_path_factory.mktemp("other")

    _git(other, "init", "-q")
    (other / "module.py").write_text("handler = 'original'\n")
    _git(other, "add", "module.py")
    _git(other, "commit", "-q", "-m", "init")

    clean = GitState.commit(other)

    assert len(clean) == 12
    assert clean != GitState.commit()

    (other / "module.py").write_text("handler = 'patched in place'\n")

    assert GitState.commit(other) == f"{clean} {GitState.DIRTY_SUFFIX}"
    assert GitState.commit(tmp_path_factory.mktemp("plain")).startswith("unavailable (")
    assert GitState.commit(Path("/nonexistent/dependency")) == "unavailable (no such directory: /nonexistent/dependency)"


def test_status_timeout_reports_an_unknown_working_tree(checkout, monkeypatch):
    """A git status that times out on a slow checkout reports the commit with an unknown tree state instead of raising."""
    calls = []
    real  = GitState._git

    def _slow_status(*args, repo_root=None):
        """Records the git subcommand and times out on status."""
        calls.append(args[0])
        if args[0] == "status":
            raise subprocess.TimeoutExpired(cmd="git status", timeout=10)
        return real(*args, repo_root=repo_root)

    monkeypatch.setattr(GitState, "_git", staticmethod(_slow_status))

    assert GitState.commit(checkout).endswith("(working tree state unknown)")
    assert calls == ["rev-parse", "status"]


def test_provenance_carries_the_sources_of_the_parsed_command_line(monkeypatch):
    """Verifies the configuration sources are None without a parsed command line and the ConfigCli mapping with one."""
    from dataclasses import dataclass

    from tools.runtime.config_cli      import ConfigCli
    from tools.runtime.machine_profile import MachineProfile

    @dataclass
    class EntryConfig:
        """One-leaf config standing in for an entry configuration."""
        seed : int = 1

    monkeypatch.setattr(ConfigCli, "parsed", None)

    assert RunProvenance.collect()["config_sources"] is None

    cli = ConfigCli(EntryConfig(), profile=MachineProfile(None))
    cli.apply(["--seed", "5"])
    monkeypatch.setattr(ConfigCli, "parsed", cli)

    sources = RunProvenance.collect()["config_sources"]

    assert sources["machine_profile"] == {"path": None, "state": "disabled"}
    assert sources["leaves"]          == [{"path": "seed", "origin": "command line", "before": 1, "after": 5}]


def test_provenance_reads_only_imported_libraries(monkeypatch):
    """Verifies RunProvenance reports None for a library the process never imported."""
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)

    provenance = RunProvenance.collect()

    assert provenance["torch"] is None
    assert provenance["cuda"] is None
    assert provenance["python"]
