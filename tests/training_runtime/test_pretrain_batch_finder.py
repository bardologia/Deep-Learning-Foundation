"""Tests for the pre-training batch size search and memory probe.

Covers candidate enumeration, selection of the largest batch fitting the
VRAM budget, out-of-memory handling and probe argument validation.
"""

from __future__ import annotations

import pytest
import torch

from tools.training.pretraining.batch_finder import BatchSizeFinder, TrainStepMemoryProbe


class _Logger:
    """Logger stub swallowing the section calls made by the finder."""
    def section(self, *args, **kwargs):
        """Ignores a section heading."""
        pass

    def subsection(self, *args, **kwargs):
        """Ignores a subsection heading."""
        pass


def test_candidates_are_powers_of_two_up_to_ceiling():
    """Verifies that candidates are powers of two not exceeding the ceiling."""
    assert BatchSizeFinder.candidates(16) == [1, 2, 4, 8, 16]
    assert BatchSizeFinder.candidates(10) == [1, 2, 4, 8]


def test_run_selects_last_fitting_batch_before_over():
    """Verifies that the last batch fitting the budget is selected and trials recorded."""
    peaks  = {1: 1.0, 2: 2.0, 4: 5.0, 8: 6.0}
    finder = BatchSizeFinder(trial_step=lambda bs: peaks[bs], budget_gb=4.0, ceiling=8, device=torch.device("cpu"), logger=_Logger(), model_name="m")

    result = finder.run()

    assert result["status"]     == "PASS"
    assert result["batch_size"] == 2
    assert result["peak_gb"]    == 2.0
    assert [trial["status"] for trial in result["trials"]] == ["FIT", "FIT", "OVER"]


def test_run_handles_oom_and_calls_on_oom():
    """Verifies that an out-of-memory trial stops the search and fires the on_oom hook."""
    seen = {"oom": 0}

    def trial(batch_size):
        """Returns the peak memory for `batch_size`, raising OOM at 4 and above."""
        if batch_size >= 4:
            raise torch.cuda.OutOfMemoryError("oom")
        return float(batch_size)

    finder = BatchSizeFinder(trial_step=trial, budget_gb=100.0, ceiling=8, device=torch.device("cpu"), logger=_Logger(), on_oom=lambda: seen.__setitem__("oom", seen["oom"] + 1))

    result = finder.run()

    assert result["status"]     == "PASS"
    assert result["batch_size"] == 2
    assert seen["oom"]          == 1
    assert result["trials"][-1]["status"] == "OOM"


def test_run_fails_when_batch_one_exceeds_budget():
    """Verifies that a batch size of one exceeding the budget yields a FAIL result."""
    finder = BatchSizeFinder(trial_step=lambda bs: 100.0, budget_gb=4.0, ceiling=8, device=torch.device("cpu"), logger=_Logger(), model_name="m")

    result = finder.run()

    assert result["status"]     == "FAIL"
    assert result["batch_size"] is None
    assert result["error"]      is not None


def test_result_dict_has_expected_keys():
    """Verifies that the result dictionary carries exactly the documented keys."""
    finder = BatchSizeFinder(trial_step=lambda bs: 1.0, budget_gb=4.0, ceiling=2, device=torch.device("cpu"), logger=_Logger())

    result = finder.run()

    assert set(result) == {"model", "status", "batch_size", "peak_gb", "budget_gb", "ceiling", "context_gb", "trials", "error"}


def test_result_records_the_measured_context():
    """Verifies that the measured context memory is echoed in the result."""
    finder = BatchSizeFinder(trial_step=lambda bs: 1.0, budget_gb=4.0, ceiling=2, device=torch.device("cpu"), logger=_Logger(), context_gb=0.7)

    result = finder.run()

    assert result["context_gb"] == 0.7


def test_measure_context_is_zero_without_cuda():
    """Verifies that context measurement is zero on a CPU device."""
    assert TrainStepMemoryProbe.measure_context(torch.device("cpu")) == 0.0


def test_probe_rejects_non_positive_measure_steps():
    """Verifies that a non-positive measure_steps raises ValueError."""
    with pytest.raises(ValueError, match="measure_steps"):
        TrainStepMemoryProbe(trainer=None, dataset=[1, 2, 3], measure_steps=0, device=torch.device("cpu"))


def test_trial_raises_cleanly_when_dataset_is_smaller_than_one_batch():
    """Verifies that a dataset smaller than one batch raises a descriptive RuntimeError."""
    probe = TrainStepMemoryProbe(trainer=None, dataset=[1, 2, 3], measure_steps=1, device=torch.device("cuda"), context_gb=0.0)

    with pytest.raises(RuntimeError, match="fewer than one full batch"):
        probe(batch_size=8)
