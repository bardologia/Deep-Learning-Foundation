from __future__ import annotations

from tools.monitoring.logger  import Logger
from tools.orchestration.pool import ProcessPoolRunner


class SequentialSessionScheduler:
    EMPTY_MESSAGE : str | None = None
    SESSION_NOUN  : str        = "sessions"

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
        if not sessions and self.EMPTY_MESSAGE is not None:
            raise RuntimeError(self.EMPTY_MESSAGE)

        self.logger.subsection(f"Dispatching {len(sessions)} {self.SESSION_NOUN} sequentially")

        runner    = ProcessPoolRunner(logger=self.logger, max_workers=1)
        completed = runner.run(sessions, self._session_runner())

        results = {}
        for session, outputs in completed:
            results[self._result_key(session)] = outputs
            self.logger.section(self._completion_message(session))
            self.logger.kv_table(self._outputs_table(outputs), title="Outputs")

        return results
