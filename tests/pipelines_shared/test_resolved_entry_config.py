
from __future__ import annotations

import json
from dataclasses import dataclass

from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from tests.conftest                               import SilentLogger


@dataclass
class _Entry:

    epochs : int = 3


def test_save_writes_the_config_and_a_provenance_record(tmp_path):
    written    = ResolvedEntryConfigWriter.save(_Entry(), tmp_path, SilentLogger())
    provenance = json.loads(ResolvedEntryConfigWriter.provenance_path_for(tmp_path).read_text())

    assert written == tmp_path / "docs" / "resolved_entry_config.json"
    assert written.is_file()
    assert provenance["written_at"]
    assert provenance["git_commit"]
    assert provenance["host"]
