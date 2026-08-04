from __future__ import annotations

import numpy as np
import pytest

from tools.data.sampling import Sampler


def test_returns_all_when_max_exceeds_total():
    idx = Sampler.deterministic_indices(10, 100)

    assert np.array_equal(idx, np.arange(10))
    assert idx.dtype == np.int64


def test_returns_all_when_max_zero():
    idx = Sampler.deterministic_indices(10, 0)

    assert np.array_equal(idx, np.arange(10))


def test_returns_all_when_max_negative():
    idx = Sampler.deterministic_indices(7, -1)

    assert np.array_equal(idx, np.arange(7))


def test_subsamples_correct_count():
    idx = Sampler.deterministic_indices(100, 10)

    assert idx.shape == (10,)


def test_subsample_indices_sorted_and_unique():
    idx = Sampler.deterministic_indices(1000, 50)

    assert np.all(np.diff(idx) > 0)
    assert len(np.unique(idx)) == len(idx)


def test_subsample_indices_in_range():
    idx = Sampler.deterministic_indices(500, 100)

    assert idx.min() >= 0
    assert idx.max() < 500


def test_deterministic_same_seed():
    a = Sampler.deterministic_indices(1000, 100, seed=7)
    b = Sampler.deterministic_indices(1000, 100, seed=7)

    assert np.array_equal(a, b)


def test_different_seed_differs():
    a = Sampler.deterministic_indices(1000, 100, seed=1)
    b = Sampler.deterministic_indices(1000, 100, seed=2)

    assert not np.array_equal(a, b)


def test_equal_total_and_max_returns_full_arange():
    idx = Sampler.deterministic_indices(50, 50)

    assert np.array_equal(idx, np.arange(50))


@pytest.mark.real_data
def test_sample_real_secondary_pixels(secondaries):
    window  = np.asarray(secondaries[0, :64, :64])
    flat    = window.reshape(-1)
    idx     = Sampler.deterministic_indices(flat.shape[0], 100, seed=42)
    sampled = flat[idx]

    assert idx.shape     == (100,)
    assert sampled.shape == (100,)
    assert idx.max() < flat.shape[0]
