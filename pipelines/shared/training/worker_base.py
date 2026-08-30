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
