from __future__ import annotations

import torch
from torch.utils.data import Dataset

from pipelines.shared.dataset.loaders import Loader
from tools.monitoring.logger          import Logger


class _CountingDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.tensor([float(idx)]), torch.tensor([float(idx)])


def _logger(tmp_path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="loader_test", level="ERROR")


def _drain_indices(loader) -> list[int]:
    return [int(x.item()) for batch in loader for x in batch[0].reshape(-1)]


def test_build_returns_three_loaders(tmp_path):
    train, val, test = Loader.build(
        _CountingDataset(20), _CountingDataset(10), _CountingDataset(6),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert {type(train).__name__, type(val).__name__, type(test).__name__} == {"DataLoader"}


def test_batch_counts_respect_drop_last(tmp_path):
    train, val, test = Loader.build(
        _CountingDataset(22), _CountingDataset(10), _CountingDataset(7),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert len(train) == 5
    assert len(val)   == 3
    assert len(test)  == 2


def test_train_drops_last_partial_batch_val_test_keep_it(tmp_path):
    train, val, test = Loader.build(
        _CountingDataset(10), _CountingDataset(10), _CountingDataset(10),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert len(_drain_indices(train)) == 8
    assert len(_drain_indices(val))   == 10
    assert len(_drain_indices(test))  == 10


def test_train_shuffle_true_changes_order(tmp_path):
    train, _, _ = Loader.build(
        _CountingDataset(32), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0, shuffle_train=True,
    )

    assert _drain_indices(train) != sorted(_drain_indices(train))


def test_val_test_preserve_sequential_order(tmp_path):
    _, val, test = Loader.build(
        _CountingDataset(8), _CountingDataset(8), _CountingDataset(8),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert _drain_indices(val)  == list(range(8))
    assert _drain_indices(test) == list(range(8))


def test_train_ordering_reproducible_for_fixed_seed(tmp_path):
    train_a, _, _ = Loader.build(
        _CountingDataset(40), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=123,
    )
    train_b, _, _ = Loader.build(
        _CountingDataset(40), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=123,
    )

    assert _drain_indices(train_a) == _drain_indices(train_b)


def test_train_ordering_differs_for_different_seed(tmp_path):
    train_a, _, _ = Loader.build(
        _CountingDataset(64), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=1,
    )
    train_b, _, _ = Loader.build(
        _CountingDataset(64), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=2,
    )

    assert _drain_indices(train_a) != _drain_indices(train_b)


def test_zero_workers_disables_prefetch_and_persistence(tmp_path):
    train, val, _ = Loader.build(
        _CountingDataset(8), _CountingDataset(8), _CountingDataset(8),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0, prefetch_factor=8,
    )

    assert train.prefetch_factor    is None
    assert train.persistent_workers is False
    assert val.prefetch_factor      is None
