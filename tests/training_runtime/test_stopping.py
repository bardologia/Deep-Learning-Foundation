"""Tests for early stopping on validation loss.

Covers best-loss tracking, the patience counter, the minimum improvement
delta, non-finite rounds, reset and state dict round trips.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.training.stopping import EarlyStopping


def _es_config(patience=3, min_delta=0.0):
    """Returns a namespace holding an early_stopping section with the given settings."""
    return SimpleNamespace(early_stopping=SimpleNamespace(patience=patience, min_delta=min_delta, restore_best=True))


def test_early_stopping_first_call_sets_best(logger, tracker):
    """Verifies that the first call records the best loss and does not stop."""
    es   = EarlyStopping(_es_config(patience=3), logger, tracker)
    stop = es(5.0, epoch=0)

    assert stop            is False
    assert es.best_loss    == 5.0
    assert es.best_epoch   == 0
    assert es.counter      == 0


def test_early_stopping_improvement_resets_counter(logger, tracker):
    """Verifies that an improvement resets the patience counter and best epoch."""
    es = EarlyStopping(_es_config(patience=3), logger, tracker)
    es(5.0, 0)
    es(6.0, 1)
    es(6.0, 2)
    assert es.counter == 2

    es(4.0, 3)
    assert es.counter    == 0
    assert es.best_loss  == 4.0
    assert es.best_epoch == 3


def test_early_stopping_triggers_after_patience(logger, tracker):
    """Verifies that stopping triggers once patience epochs pass without improvement."""
    es = EarlyStopping(_es_config(patience=3), logger, tracker)

    assert es(1.0, 0) is False
    assert es(2.0, 1) is False
    assert es(2.0, 2) is False
    assert es(2.0, 3) is True
    assert es.triggered is True


def test_early_stopping_does_not_trigger_below_patience(logger, tracker):
    """Verifies that the counter grows without triggering below the patience limit."""
    es = EarlyStopping(_es_config(patience=5), logger, tracker)

    es(1.0, 0)
    for epoch in range(1, 5):
        assert es(2.0, epoch) is False
    assert es.counter == 4


def test_early_stopping_strict_less_than_for_improvement(logger, tracker):
    """Verifies that an equal loss counts as no improvement."""
    es = EarlyStopping(_es_config(patience=2), logger, tracker)

    es(1.0, 0)
    es(1.0, 1)
    assert es.counter == 1
    assert es.best_epoch == 0


def test_early_stopping_best_tracks_running_minimum(logger, tracker):
    """Verifies that the best loss and epoch track the running minimum."""
    es     = EarlyStopping(_es_config(patience=10), logger, tracker)
    losses = [5.0, 4.0, 4.5, 3.0, 3.2, 2.0]

    for epoch, loss in enumerate(losses):
        es(loss, epoch)

    assert es.best_loss  == 2.0
    assert es.best_epoch == 5


def test_early_stopping_reset(logger, tracker):
    """Verifies that reset clears best loss, counter, best epoch and trigger flag."""
    es = EarlyStopping(_es_config(patience=2), logger, tracker)
    es(1.0, 0)
    es(2.0, 1)
    es.reset()

    assert es.best_loss  is None
    assert es.counter    == 0
    assert es.best_epoch == -1
    assert es.triggered  is False


def test_early_stopping_min_delta_treats_small_gains_as_no_improvement(logger, tracker):
    """Verifies that gains smaller than min_delta do not reset the counter."""
    es = EarlyStopping(_es_config(patience=2, min_delta=0.1), logger, tracker)

    assert es(1.0, 0) is False
    assert es(0.95, 1) is False
    assert es(0.92, 2) is True
    assert es.best_loss == pytest.approx(1.0)


def test_early_stopping_min_delta_accepts_large_gains(logger, tracker):
    """Verifies that gains larger than min_delta update the best loss and reset the counter."""
    es = EarlyStopping(_es_config(patience=2, min_delta=0.1), logger, tracker)

    es(1.0, 0)
    es(0.8, 1)

    assert es.best_loss  == pytest.approx(0.8)
    assert es.counter    == 0
    assert es.best_epoch == 1



def test_early_stopping_state_roundtrip(logger, tracker):
    """Verifies that best loss, counter, epoch and trigger flag survive a state round trip."""
    es = EarlyStopping(_es_config(patience=3), logger, tracker)
    es(1.0, 0)
    es(1.2, 1)

    other = EarlyStopping(_es_config(patience=3), logger, tracker)
    other.load_state_dict(es.state_dict())

    assert other.best_loss  == pytest.approx(1.0)
    assert other.counter    == 1
    assert other.best_epoch == 0
    assert other.triggered is False


def test_non_finite_first_round_is_never_the_best_loss(recording_logger, tracker):
    """Verifies a NaN round warns, counts as no improvement and leaves the first finite round to set the best loss."""
    es = EarlyStopping(_es_config(patience=3), recording_logger, tracker)

    assert es(float("nan"), 0) is False
    assert es.best_loss  is None
    assert es.counter    == 1
    assert any(level == "warning" and "epoch 1 is nan" in message for level, message in recording_logger.messages)

    es(2.0, 1)

    assert es.best_loss  == 2.0
    assert es.best_epoch == 1
    assert es.counter    == 0


def test_non_finite_rounds_exhaust_patience(logger, tracker):
    """Verifies consecutive non-finite rounds advance the counter until early stopping fires."""
    es = EarlyStopping(_es_config(patience=2), logger, tracker)

    es(1.0, 0)

    assert es(float("inf"), 1) is False
    assert es(float("nan"), 2) is True
    assert es.best_loss  == 1.0
    assert es.best_epoch == 0
