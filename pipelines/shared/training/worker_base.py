"""Common base for pipeline workers addressed by a run tag.

Resolves the run directory a worker writes into from the configured log base
directory and the run tag it was dispatched with.
"""

from __future__ import annotations

from pathlib import Path


class WorkerBase:
    """Base class holding a worker's configuration and run directory.

    Attributes:
        config  : Configuration of the pipeline the worker belongs to.
        run_tag : Tag naming the run directory.
        run_dir : Run directory under the configured log base directory.
    """

    def __init__(self, config, run_tag: str) -> None:
        """Stores the configuration and resolves the tagged run directory."""
        self.config  = config
        self.run_tag = run_tag
        self.run_dir = Path(config.paths.log_base_dir) / run_tag

    def skip_complete(self, run_directory: Path, log_name: str, resume: bool) -> bool:
        """Returns whether a finished unit is left as it is, recording the decision in the unit's own log.

        A local stage drops its finished units before it queues anything, but a
        dispatcher that submits every unit it knows of hands finished ones over
        too, so the worker has to recognise a unit a resume must not train again.
        With resume off a finished unit is never trained over in place: the
        worker refuses it instead of overwriting a completed run.

        Args:
            run_directory : Run directory of the unit.
            log_name      : Name of the log file the decision is written to.
            resume        : Whether the launch reuses completed units.

        Returns:
            Whether the unit already holds a completed run that this launch reuses.

        Raises:
            SystemExit : If the unit holds a completed run while resume is off.
        """
        from tools.monitoring.logger  import Logger
        from tools.runtime.completion import CompletionMarker

        if not CompletionMarker.is_complete(run_directory):
            return False

        if not resume:
            raise SystemExit(f"{run_directory} already holds a completed run and resume is off; relaunch under a new run_tag to train it again, or delete the directory first, so a finished run is never overwritten in place")

        logger = Logger(log_dir=str(run_directory / "logs"), name=log_name, level="INFO")
        logger.info(f"{run_directory.name} already holds a completed run; this resume leaves it as it is")
        logger.close()

        return True
