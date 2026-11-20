
from __future__ import annotations

import time

from tools.monitoring.logger  import Logger
from tools.orchestration.pool import ProcessPoolRunner


class SequentialSessionScheduler:

    SESSION_NOUN : str = "sessions"

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def _sessions(self) -> list:
        raise NotImplementedError

    def _session_runner(self):
        raise NotImplementedError

    def _result_key(self, session) -> str:
        raise NotImplementedError

    def _completion_message(self, session) -> str:
        raise NotImplementedError

    def _outputs_table(self, outputs: dict) -> dict:
        return {name: str(path) for name, path in outputs.items()}

    def run(self) -> dict:
        sessions = self._sessions()

        self.logger.subsection(f"Dispatching {len(sessions)} {self.SESSION_NOUN} sequentially")

        runner  = ProcessPoolRunner(logger=self.logger, max_workers=1)
        results = {}

        for session in sessions:
            started   = time.perf_counter()
            completed = runner.run([session], self._session_runner(), self._result_key)
            outputs   = completed[0][1]

            results[self._result_key(session)] = outputs

            self.logger.subsection(f"{self._completion_message(session)} in {time.perf_counter() - started:.1f} s")
            self.logger.kv_table(self._outputs_table(outputs), title="Outputs")

        return results
