"""Tests for deterministic index subsampling in Sampler.

Covers the pass-through cases where no subsampling applies, the shape,
ordering, uniqueness and range of drawn indices, seed reproducibility, and
sampling from a real secondary-image window.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.data.sampling import Sampler


def test_returns_all_when_max_exceeds_total():
    """All indices are returned unchanged as int64 when the cap exceeds the population."""
    idx = Sampler.deterministic_indices(10, 100)

    assert np.array_equal(idx, np.arange(10))
    assert idx.dtype == np.int64


def test_returns_all_when_max_zero():
    """A cap of zero disables subsampling and returns the full index range."""
    idx = Sampler.deterministic_indices(10, 0)

    assert np.array_equal(idx, np.arange(10))


def test_returns_all_when_max_negative():
    """A negative cap disables subsampling and returns the full index range."""
    idx = Sampler.deterministic_indices(7, -1)

    assert np.array_equal(idx, np.arange(7))


def test_subsamples_correct_count():
    """The number of drawn indices equals the requested cap."""
    idx = Sampler.deterministic_indices(100, 10)

    assert idx.shape == (10,)


def test_subsample_indices_sorted_and_unique():
    """Drawn indices are strictly increasing and free of duplicates."""
    idx = Sampler.deterministic_indices(1000, 50)

    assert np.all(np.diff(idx) > 0)
    assert len(np.unique(idx)) == len(idx)


def test_subsample_indices_in_range():
    """Drawn indices stay within the half-open interval of the population size."""
    idx = Sampler.deterministic_indices(500, 100)

    assert idx.min() >= 0
    assert idx.max() < 500


def test_deterministic_same_seed():
    """Two draws with the same seed produce identical indices."""
    a = Sampler.deterministic_indices(1000, 100, seed=7)
    b = Sampler.deterministic_indices(1000, 100, seed=7)

    assert np.array_equal(a, b)


def test_different_seed_differs():
    """Draws with different seeds produce different indices."""
    a = Sampler.deterministic_indices(1000, 100, seed=1)
    b = Sampler.deterministic_indices(1000, 100, seed=2)

    assert not np.array_equal(a, b)


def test_equal_total_and_max_returns_full_arange():
    """A cap equal to the population size returns the full index range."""
    idx = Sampler.deterministic_indices(50, 50)

    assert np.array_equal(idx, np.arange(50))


@pytest.mark.dataset
def test_sample_scene_secondary_pixels(secondaries):
    """Sampling a flattened scene secondary window yields the requested count of in-range pixels."""
    window  = np.asarray(secondaries[0, :64, :64])
    flat    = window.reshape(-1)
    idx     = Sampler.deterministic_indices(flat.shape[0], 100, seed=42)
    sampled = flat[idx]

    assert idx.shape     == (100,)
    assert sampled.shape == (100,)
    assert idx.max() < flat.shape[0]
