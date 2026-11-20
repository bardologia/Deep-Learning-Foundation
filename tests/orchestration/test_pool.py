
from __future__ import annotations

import time

import pytest

from tools.orchestration.pool import ProcessPoolRunner

from tests.conftest import SilentLogger
from unittest.mock import Mock


def _square(x):
    return x * x


def _identity(x):
    return x


def _raise_on_three(x):
    if x == 3:
        Mock(side_effect=ValueError(f"boom at {x}"))()
    return x


def _fail_fast_or_sleep(x):
    if x == 0:
        Mock(side_effect=ValueError("early failure"))()
    time.sleep(2.0)
    return x


def _const(_x):
    return 42


def _label(job):
    return f"job {job}"


class RecordingLogger(SilentLogger):
    def __init__(self):
        self.subsections = []
        self.errors      = []
        self.warnings    = []

    def subsection(self, message, *args, **kwargs):
        self.subsections.append(str(message))

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))

    def warning(self, message, *args, **kwargs):
        self.warnings.append(str(message))


@pytest.fixture
def logger():
    return SilentLogger()


def test_empty_jobs_returns_empty(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=2)
    results = runner.run([], _square, _label)

    assert results == []


def test_results_pair_job_with_result(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=2)
    results = runner.run([1, 2, 4], _square, _label)

    assert dict(results) == {1: 1, 2: 4, 4: 16}


def test_parallel_matches_serial(logger):
    jobs    = list(range(8))
    serial  = {job: _square(job) for job in jobs}

    runner  = ProcessPoolRunner(logger=logger, max_workers=4)
    results = dict(runner.run(jobs, _square, _label))

    assert results == serial


def test_each_job_appears_once(logger):
    jobs    = [10, 20, 30, 40]
    runner  = ProcessPoolRunner(logger=logger, max_workers=3)
    results = runner.run(jobs, _identity, _label)

    returned_jobs = [job for job, _ in results]
    assert sorted(returned_jobs) == sorted(jobs)
    assert len(results) == len(jobs)


def test_error_propagates(logger):
    runner = ProcessPoolRunner(logger=logger, max_workers=2)

    with pytest.raises(ValueError):
        runner.run([1, 2, 3, 4], _raise_on_three, _label)


def test_max_workers_capped_to_job_count(logger):
    runner = ProcessPoolRunner(logger=logger, max_workers=64)

    assert dict(runner.run([5, 6], _square, _label)) == {5: 25, 6: 36}


def test_single_worker_runs_all_jobs(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=1)
    results = dict(runner.run([1, 2, 3], _square, _label))

    assert results == {1: 1, 2: 4, 3: 9}


def test_unbounded_workers_when_none(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=None)
    results = dict(runner.run([2, 3], _square, _label))

    assert results == {2: 4, 3: 9}


def test_empty_jobs_with_unbounded_workers_returns_empty(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=None)
    results = runner.run([], _square, _label)

    assert results == []


def test_each_completion_is_logged():
    recorder = RecordingLogger()
    runner   = ProcessPoolRunner(logger=recorder, max_workers=2)

    runner.run([1, 2, 3], _square, _label)

    assert len(recorder.subsections) == 3
    assert sorted(line.split(" completed")[0] for line in recorder.subsections) == ["Job job 1", "Job job 2", "Job job 3"]
    assert all("/3," in line for line in recorder.subsections)


def test_failing_job_is_named_in_the_log():
    recorder = RecordingLogger()
    runner   = ProcessPoolRunner(logger=recorder, max_workers=1)

    with pytest.raises(ValueError):
        runner.run([3], _raise_on_three, _label)

    assert len(recorder.errors) == 1
    assert "job 3" in recorder.errors[0]


def test_iterable_input_is_consumed(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=2)
    results = dict(runner.run(iter([1, 2, 3]), _const, _label))

    assert results == {1: 42, 2: 42, 3: 42}


def test_first_failure_names_the_running_jobs_it_waits_for():
    recorder = RecordingLogger()
    runner   = ProcessPoolRunner(logger=recorder, max_workers=2)

    with pytest.raises(ValueError, match="early failure"):
        runner.run([0, 1], _fail_fast_or_sleep, _label)

    assert recorder.warnings == ["Waiting for 1 in-flight job(s) to finish before re-raising, since a running worker cannot be cancelled: job 1"]
