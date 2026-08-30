"""Persistence of an entry point's resolved configuration into its run directory.

Every entry that owns a run directory records the configuration it actually ran
with under ``docs/resolved_entry_config.json``, which is what the launch
console's copy-from-run control reads back. One writer for all of them, so the
file always lands in the same place, always carries post-preflight values, and
its path is always reported in the run log.
"""

from __future__ import annotations

from pathlib import Path

from tools.runtime.config_cli import ConfigCli


class ResolvedEntryConfigWriter:
    """Writes the resolved entry configuration into a run directory's docs.

    Attributes:
        RELATIVE_PATH : Location of the file inside a run directory.
    """

    RELATIVE_PATH = Path("docs") / "resolved_entry_config.json"

    @classmethod
    def path_for(cls, run_directory) -> Path:
        """Returns the resolved config path of a run directory.

        Args:
            run_directory : Run directory owning the file.

        Returns:
            Absolute path of ``docs/resolved_entry_config.json`` in that run.
        """
        return Path(run_directory) / cls.RELATIVE_PATH

    @classmethod
    def save(cls, config, run_directory, logger) -> Path:
        """Writes the resolved entry configuration and reports the written path.

        Args:
            config        : Entry configuration dataclass, with every value the run will actually use already resolved.
            run_directory : Run directory the file is written into.
            logger        : Logger the written path is reported on.

        Returns:
            Path of the written file.
        """
        written = ConfigCli.save_resolved(config, cls.path_for(run_directory))
        logger.info(f"Resolved entry config saved: {written}")

        return written
