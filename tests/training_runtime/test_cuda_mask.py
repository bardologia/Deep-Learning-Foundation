"""Tests for the shared restoration of a pinned parent's inherited CUDA mask."""

from __future__ import annotations

from tools.runtime.cuda_mask import InheritedCudaMask


def test_restored_drops_the_mask_when_the_stash_is_empty(monkeypatch):
    """A parent pinned on a bare machine hands its children an environment without any CUDA mask."""
    monkeypatch.setenv("TOMOSAR_INHERITED_CUDA_MASK", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    env = InheritedCudaMask.restored()

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "TOMOSAR_INHERITED_CUDA_MASK" not in env


def test_restored_reinstates_a_scheduler_allocation_mask(monkeypatch):
    """A parent pinned inside a SLURM allocation hands its children the allocation's full mask."""
    monkeypatch.setenv("TOMOSAR_INHERITED_CUDA_MASK", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    env = InheritedCudaMask.restored()

    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "TOMOSAR_INHERITED_CUDA_MASK" not in env


def test_restored_without_a_stash_leaves_the_mask_alone(monkeypatch):
    """An unpinned parent forwards its environment unchanged, mask included."""
    monkeypatch.delenv("TOMOSAR_INHERITED_CUDA_MASK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")

    env = InheritedCudaMask.restored()

    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"
