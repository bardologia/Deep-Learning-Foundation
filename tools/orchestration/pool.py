from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing             import Any, Callable, Iterable

from tools.monitoring.logger import Logger


class ProcessPoolRunner:
    def __init__(self, logger: Logger, max_workers: int | None = None) -> None:
        self.logger      = logger
        self.max_workers = max_workers

    def run(self, jobs: Iterable[Any], worker_fn: Callable[[Any], Any]) -> list[tuple[Any, Any]]:
        job_list = list(jobs)
        workers  = max(1, min(self.max_workers, len(job_list)) if self.max_workers is not None else len(job_list))
        results  : list[tuple[Any, Any]] = []

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(worker_fn, job): job for job in job_list}

            try:
                for future in as_completed(futures):
                    job    = futures[future]
                    result = future.result()

                    results.append((job, result))
            except Exception:
                for future in futures:
                    future.cancel()
                raise

        return results
