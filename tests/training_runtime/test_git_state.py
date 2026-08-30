"""Tests for the git revision and working-tree provenance reporting."""

from __future__ import annotations

from tools.runtime.git_state import GitState


def test_commit_returns_a_hash_or_a_reasoned_unavailable():
    """The commit string is either a 12-character hash (dirty-annotated or not) or an unavailable notice."""
    commit = GitState.commit()

    assert isinstance(commit, str) and commit
    assert commit.startswith("unavailable") or len(commit.split()[0]) == 12


def test_dirty_state_contract():
    """The dirty state always carries the dirty flag, the diff text and the untracked list, consistently filled."""
    state = GitState.dirty_state()

    assert set(state) == {"dirty", "diff", "untracked"}
    assert isinstance(state["dirty"], bool)
    assert isinstance(state["diff"], str)
    assert isinstance(state["untracked"], list)

    if not state["dirty"]:
        assert state["diff"] == "" and state["untracked"] == []


def test_dirty_state_agrees_with_the_commit_annotation():
    """The dirty flag and the '(dirty working tree)' suffix of commit() tell the same story."""
    commit = GitState.commit()
    state  = GitState.dirty_state()

    if not commit.startswith("unavailable"):
        assert state["dirty"] == commit.endswith("(dirty working tree)")
