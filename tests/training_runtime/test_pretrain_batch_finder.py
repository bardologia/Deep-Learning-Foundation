from __future__ import annotations

import pytest
import torch

from tools.training.pretraining.batch_finder import BatchSizeFinder, TrainStepMemoryProbe


class _Logger:
    def section(self, *args, **kwargs):
        pass

    def subsection(self, *args, **kwargs):
        pass


def test_candidates_are_powers_of_two_up_to_ceiling():
    assert BatchSizeFinder.candidates(16) == [1, 2, 4, 8, 16]
    assert BatchSizeFinder.candidates(10) == [1, 2, 4, 8]


def test_run_selects_last_fitting_batch_before_over():
    peaks  = {1: 1.0, 2: 2.0, 4: 5.0, 8: 6.0}
    finder = BatchSizeFinder(trial_step=lambda bs: peaks[bs], budget_gb=4.0, ceiling=8, device=torch.device("cpu"), logger=_Logger(), model_name="m")

    result = finder.run()

    assert result["status"]     == "PASS"
    assert result["batch_size"] == 2
    assert result["peak_gb"]    == 2.0
    assert [trial["status"] for trial in result["trials"]] == ["FIT", "FIT", "OVER"]


def test_run_handles_oom_and_calls_on_oom():
    seen = {"oom": 0}

    def trial(batch_size):
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
    finder = BatchSizeFinder(trial_step=lambda bs: 100.0, budget_gb=4.0, ceiling=8, device=torch.device("cpu"), logger=_Logger(), model_name="m")

    result = finder.run()

    assert result["status"]     == "FAIL"
    assert result["batch_size"] is None
    assert result["error"]      is not None


def test_result_dict_has_expected_keys():
    finder = BatchSizeFinder(trial_step=lambda bs: 1.0, budget_gb=4.0, ceiling=2, device=torch.device("cpu"), logger=_Logger())

    result = finder.run()

    assert set(result) == {"model", "status", "batch_size", "peak_gb", "budget_gb", "ceiling", "context_gb", "trials", "error"}


def test_result_records_the_measured_context():
    finder = BatchSizeFinder(trial_step=lambda bs: 1.0, budget_gb=4.0, ceiling=2, device=torch.device("cpu"), logger=_Logger(), context_gb=0.7)

    result = finder.run()

    assert result["context_gb"] == 0.7


def test_measure_context_is_zero_without_cuda():
    assert TrainStepMemoryProbe.measure_context(torch.device("cpu")) == 0.0


def test_trial_raises_cleanly_when_dataset_is_smaller_than_one_batch():
    probe = TrainStepMemoryProbe(trainer=None, dataset=[1, 2, 3], measure_steps=1, device=torch.device("cuda"), context_gb=0.0)

    with pytest.raises(RuntimeError, match="fewer than one full batch"):
        probe(batch_size=8)
