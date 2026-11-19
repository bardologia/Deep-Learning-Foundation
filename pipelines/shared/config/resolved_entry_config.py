
from __future__ import annotations

from pathlib import Path

from tools.data.io            import FileIO
from tools.runtime.config_cli import ConfigCli
from tools.runtime.git_state  import RunProvenance
from tools.runtime.run_tag    import RunTag


class ResolvedEntryConfigWriter:

    RELATIVE_PATH   = Path("docs") / "resolved_entry_config.json"
    PROVENANCE_PATH = Path("docs") / "provenance.json"

    @classmethod
    def path_for(cls, run_directory) -> Path:
        return Path(run_directory) / cls.RELATIVE_PATH

    @classmethod
    def provenance_path_for(cls, run_directory) -> Path:
        return Path(run_directory) / cls.PROVENANCE_PATH

    @classmethod
    def save(cls, config, run_directory, logger) -> Path:
        written    = ConfigCli.save_resolved(config, cls.path_for(run_directory))
        provenance = cls.provenance_path_for(run_directory)

        FileIO.save_json({"written_at": RunTag.timestamp(), **RunProvenance.collect()}, provenance)

        logger.ok(f"Resolved entry config saved: {written}")
        logger.ok(f"Provenance saved: {provenance}")

        return written
