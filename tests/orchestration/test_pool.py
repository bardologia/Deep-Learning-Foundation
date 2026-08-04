from __future__ import annotations

import pytest

from tools.orchestration.pool import ProcessPoolRunner

from tests.conftest import SilentLogger


def _square(x):
    return x * x


def _identity(x):
    return x


def _raise_on_three(x):
    if x == 3:
        raise ValueError(f"boom at {x}")
    return x


def _const(_x):
    return 42


@pytest.fixture
def logger():
    return SilentLogger()


def test_empty_jobs_returns_empty(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=2)
    results = runner.run([], _square)

    assert results == []


def test_results_pair_job_with_result(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=2)
    results = runner.run([1, 2, 4], _square)

    assert dict(results) == {1: 1, 2: 4, 4: 16}


def test_parallel_matches_serial(logger):
    jobs    = list(range(8))
    serial  = {job: _square(job) for job in jobs}

    runner  = ProcessPoolRunner(logger=logger, max_workers=4)
    results = dict(runner.run(jobs, _square))

    assert results == serial


def test_each_job_appears_once(logger):
    jobs    = [10, 20, 30, 40]
    runner  = ProcessPoolRunner(logger=logger, max_workers=3)
    results = runner.run(jobs, _identity)

    returned_jobs = [job for job, _ in results]
    assert sorted(returned_jobs) == sorted(jobs)
    assert len(results) == len(jobs)


def test_error_propagates(logger):
    runner = ProcessPoolRunner(logger=logger, max_workers=2)

    with pytest.raises(ValueError):
        runner.run([1, 2, 3, 4], _raise_on_three)


def test_max_workers_capped_to_job_count(logger):
    runner = ProcessPoolRunner(logger=logger, max_workers=64)

    assert dict(runner.run([5, 6], _square)) == {5: 25, 6: 36}


def test_single_worker_runs_all_jobs(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=1)
    results = dict(runner.run([1, 2, 3], _square))

    assert results == {1: 1, 2: 4, 3: 9}


def test_unbounded_workers_when_none(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=None)
    results = dict(runner.run([2, 3], _square))

    assert results == {2: 4, 3: 9}


def test_empty_jobs_with_unbounded_workers_returns_empty(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=None)
    results = runner.run([], _square)

    assert results == []


def test_iterable_input_is_consumed(logger):
    runner  = ProcessPoolRunner(logger=logger, max_workers=2)
    results = dict(runner.run(iter([1, 2, 3]), _const))

    assert results == {1: 42, 2: 42, 3: 42}
