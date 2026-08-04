from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from tools.runtime.reproducibility import Reproducibility, WorkerInitializer


def test_seed_everything_makes_torch_deterministic():
    Reproducibility.seed_everything(123)
    a = torch.randn(5)

    Reproducibility.seed_everything(123)
    b = torch.randn(5)

    assert torch.equal(a, b)


def test_seed_everything_makes_numpy_and_random_deterministic():
    Reproducibility.seed_everything(7)
    np_a = np.random.rand(4)
    py_a = [random.random() for _ in range(4)]

    Reproducibility.seed_everything(7)
    np_b = np.random.rand(4)
    py_b = [random.random() for _ in range(4)]

    assert np.array_equal(np_a, np_b)
    assert py_a == py_b


def test_different_seeds_differ():
    Reproducibility.seed_everything(1)
    a = torch.randn(8)

    Reproducibility.seed_everything(2)
    b = torch.randn(8)

    assert not torch.equal(a, b)


def test_seed_everything_sets_cudnn_flags():
    Reproducibility.seed_everything(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark     is False


def test_generator_is_reproducible():
    g1 = Reproducibility.generator(42)
    a  = torch.randn(6, generator=g1)

    g2 = Reproducibility.generator(42)
    b  = torch.randn(6, generator=g2)

    assert torch.equal(a, b)


def test_generator_different_seed_differs():
    g1 = Reproducibility.generator(42)
    g2 = Reproducibility.generator(43)

    a = torch.randn(6, generator=g1)
    b = torch.randn(6, generator=g2)

    assert not torch.equal(a, b)


def test_seed_everything_handles_large_seed_via_modulus():
    big = Reproducibility.SEED_MODULUS + 5
    Reproducibility.seed_everything(big)
    a = np.random.rand(3)

    Reproducibility.seed_everything(big)
    b = np.random.rand(3)

    assert np.array_equal(a, b)


def test_worker_init_factory_returns_initializer():
    init = Reproducibility.worker_init(99)
    assert isinstance(init, WorkerInitializer)
    assert init.base_seed == 99


def test_worker_init_seeds_numpy_per_worker_deterministically():
    init = WorkerInitializer(1000)

    init(3)
    a = np.random.rand(4)

    init(3)
    b = np.random.rand(4)

    assert np.array_equal(a, b)


def test_worker_init_distinct_workers_differ():
    init = WorkerInitializer(1000)

    init(0)
    a = np.random.rand(4)

    init(1)
    b = np.random.rand(4)

    assert not np.array_equal(a, b)


def test_worker_init_uses_modulus_for_overflow():
    init = WorkerInitializer(Reproducibility.SEED_MODULUS - 1)
    init(5)
    a = np.random.rand(2)

    init(5)
    b = np.random.rand(2)

    assert np.array_equal(a, b)
