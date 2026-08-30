"""Tests for the shared DataLoader builder used by every training pipeline."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from pipelines.shared.dataset.loaders import Loader
from tools.monitoring.logger          import Logger
from tools.training.distributed       import DistributedContext


class _CountingDataset(Dataset):
    """Dataset whose sample at index i is the scalar i, used to trace loader ordering.

    Attributes:
        n : Number of samples.
    """
    def __init__(self, n: int) -> None:
        """Stores the sample count."""
        self.n = n

    def __len__(self) -> int:
        """Returns the number of samples."""
        return self.n

    def __getitem__(self, idx: int):
        """Returns the input and target pair for idx, both one-element tensors holding idx."""
        return torch.tensor([float(idx)]), torch.tensor([float(idx)])


def _logger(tmp_path) -> Logger:
    """Returns a silent logger writing under tmp_path."""
    return Logger(log_dir=str(tmp_path / "logs"), name="loader_test", level="ERROR")


def _drain_indices(loader) -> list[int]:
    """Returns the flattened sample indices produced by iterating the loader once."""
    return [int(x.item()) for batch in loader for x in batch[0].reshape(-1)]


def test_build_returns_three_loaders(tmp_path):
    """Loader.build returns train, val, and test DataLoader objects."""
    train, val, test = Loader.build(
        _CountingDataset(20), _CountingDataset(10), _CountingDataset(6),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert {type(train).__name__, type(val).__name__, type(test).__name__} == {"DataLoader"}


def test_batch_counts_respect_drop_last(tmp_path):
    """Train drops its partial batch while val and test keep theirs."""
    train, val, test = Loader.build(
        _CountingDataset(22), _CountingDataset(10), _CountingDataset(7),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert len(train) == 5
    assert len(val)   == 3
    assert len(test)  == 2


def test_train_drops_last_partial_batch_val_test_keep_it(tmp_path):
    """The train loader yields fewer samples than the dataset while val and test yield all of them."""
    train, val, test = Loader.build(
        _CountingDataset(10), _CountingDataset(10), _CountingDataset(10),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert len(_drain_indices(train)) == 8
    assert len(_drain_indices(val))   == 10
    assert len(_drain_indices(test))  == 10


def test_train_shuffle_true_changes_order(tmp_path):
    """Enabling shuffle_train makes the train loader emit samples out of index order."""
    train, _, _ = Loader.build(
        _CountingDataset(32), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0, shuffle_train=True,
    )

    assert _drain_indices(train) != sorted(_drain_indices(train))


def test_val_test_preserve_sequential_order(tmp_path):
    """Val and test loaders iterate samples in index order."""
    _, val, test = Loader.build(
        _CountingDataset(8), _CountingDataset(8), _CountingDataset(8),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert _drain_indices(val)  == list(range(8))
    assert _drain_indices(test) == list(range(8))


def test_train_ordering_reproducible_for_fixed_seed(tmp_path):
    """Two loaders built with the same seed emit the same train ordering."""
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
    """Two loaders built with different seeds emit different train orderings."""
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
    """With num_workers zero the prefetch factor and persistent workers are switched off."""
    train, val, _ = Loader.build(
        _CountingDataset(8), _CountingDataset(8), _CountingDataset(8),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0, prefetch_factor=8,
    )

    assert train.prefetch_factor    is None
    assert train.persistent_workers is False
    assert val.prefetch_factor      is None


def test_train_dataset_smaller_than_the_batch_raises(tmp_path):
    """A train dataset shorter than one batch raises ValueError."""
    with pytest.raises(ValueError, match="Train loader yields zero batches"):
        Loader.build(
            _CountingDataset(3), _CountingDataset(8), _CountingDataset(8),
            batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
        )


def _activate_ddp(monkeypatch, rank: int, world: int) -> None:
    """Makes the process look like one DDP rank and re-reads the cached context."""
    for key, value in DistributedContext.environment(rank=rank, world_size=world, master_port=29501).items():
        monkeypatch.setenv(key, value)
    DistributedContext.reset()


@pytest.fixture
def _restore_context():
    """Clears the cached distributed context after a test that faked one."""
    yield
    DistributedContext.reset()


def test_ddp_divides_the_batch_and_shards_every_split(tmp_path, monkeypatch, _restore_context):
    """An active DDP context halves the batch, pads the train shards and leaves eval unpadded."""
    _activate_ddp(monkeypatch, rank=0, world=2)

    train, val, test = Loader.build(
        _CountingDataset(40), _CountingDataset(8), _CountingDataset(8),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    assert train.batch_size == 2
    assert type(train.sampler).__name__ == "DistributedSampler"
    assert {type(val.sampler).__name__, type(test.sampler).__name__} == {"EvaluationShardSampler"}


def test_ddp_eval_shards_never_duplicate_a_sample(tmp_path, monkeypatch, _restore_context):
    """An odd split is partitioned without padding, so no sample is counted twice."""
    _activate_ddp(monkeypatch, rank=0, world=2)
    _, val_zero, _ = Loader.build(
        _CountingDataset(16), _CountingDataset(7), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    _activate_ddp(monkeypatch, rank=1, world=2)
    _, val_one, _ = Loader.build(
        _CountingDataset(16), _CountingDataset(7), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    shard_zero = _drain_indices(val_zero)
    shard_one  = _drain_indices(val_one)

    assert len(shard_zero) == 4 and len(shard_one) == 3
    assert sorted(shard_zero + shard_one) == list(range(7))


def test_ddp_rank_shards_are_disjoint_and_cover_the_split(tmp_path, monkeypatch, _restore_context):
    """The two ranks' unshuffled val shards partition the dataset between them."""
    _activate_ddp(monkeypatch, rank=0, world=2)
    _, val_zero, _ = Loader.build(
        _CountingDataset(16), _CountingDataset(12), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    _activate_ddp(monkeypatch, rank=1, world=2)
    _, val_one, _ = Loader.build(
        _CountingDataset(16), _CountingDataset(12), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=0,
    )

    shard_zero = set(_drain_indices(val_zero))
    shard_one  = set(_drain_indices(val_one))

    assert len(shard_zero) == len(shard_one) == 6
    assert shard_zero.isdisjoint(shard_one)
    assert shard_zero | shard_one == set(range(12))


def test_ddp_train_shards_reshuffle_identically_from_the_seed(tmp_path, monkeypatch, _restore_context):
    """Rebuilding the same rank with the same seed reproduces the train shard order."""
    _activate_ddp(monkeypatch, rank=0, world=2)

    train_a, _, _ = Loader.build(
        _CountingDataset(32), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=7,
    )
    train_b, _, _ = Loader.build(
        _CountingDataset(32), _CountingDataset(4), _CountingDataset(4),
        batch_size=4, num_workers=0, logger=_logger(tmp_path), seed=7,
    )

    assert _drain_indices(train_a) == _drain_indices(train_b)


def test_ddp_rejects_a_batch_that_does_not_divide(tmp_path, monkeypatch, _restore_context):
    """A global batch with a remainder across the ranks raises ValueError."""
    _activate_ddp(monkeypatch, rank=0, world=2)

    with pytest.raises(ValueError, match="does not divide"):
        Loader.build(
            _CountingDataset(16), _CountingDataset(8), _CountingDataset(8),
            batch_size=5, num_workers=0, logger=_logger(tmp_path), seed=0,
        )
