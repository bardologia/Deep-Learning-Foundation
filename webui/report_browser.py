
from __future__ import annotations

import json
from pathlib import Path

from catalog_roots import CatalogRoots
from web_logger    import WebLogger


class AnalysisPayloadBrowser:

    LABEL     : str
    SUBDIR    : str
    MARKER    : str
    SCHEMA    : int
    COUNT_KEY : str
    ENTRY     : str

    def __init__(self, logger: WebLogger) -> None:
        self.roots  = CatalogRoots()
        self.logger = logger

    def analyses(self, base: str) -> dict:
        root, error = self.roots.open(base, empty_error="set the datasets directory first")
        if error:
            self.logger.warning(f"{self.LABEL}: scan refused for {base!r}: {error}")
            return {"ok": False, "error": error, "entries": []}

        entries = []
        for marker in sorted(root.rglob(f"{self.SUBDIR}/*/{self.MARKER}")):
            analysis_dir = marker.parent
            run_dir      = analysis_dir.parent.parent

            entries.append({
                "id"    : str(analysis_dir),
                "run"   : run_dir.name,
                "group" : str(run_dir.relative_to(root).parent),
                "stamp" : analysis_dir.name,
            })

        entries.sort(key=lambda entry: (entry["stamp"], entry["id"]), reverse=True)

        self.logger.info(f"{self.LABEL}: {len(entries)} analyses under {root}")

        return {"ok": True, "root": str(root), "entries": entries}

    def report(self, analysis_id: str) -> dict:
        raw = (analysis_id or "").strip()
        if not raw:
            return {"ok": False, "error": "no analysis selected"}

        target = Path(raw).resolve()
        if not self.roots.contains(target):
            return self._refuse(target, "analysis lies outside the scanned datasets roots; rescan first")

        marker = target / self.MARKER
        if not marker.is_file():
            return self._refuse(target, f"no {self.MARKER} in {target}")

        with open(marker, "r", encoding="utf-8") as f:
            payload = json.load(f)

        schema = payload.get("schema")
        if schema != self.SCHEMA:
            return self._refuse(target, f"analysis payload is schema {schema}, this build reads schema {self.SCHEMA}; rerun {self.ENTRY} on this parameter run to regenerate it")

        self.logger.info(f"{self.LABEL}: read {len(payload[self.COUNT_KEY])} {self.COUNT_KEY} from {target}")

        return {"ok": True, "id": str(target), "payload": payload}

    def _refuse(self, target: Path, error: str) -> dict:
        self.logger.warning(f"{self.LABEL}: {target} refused: {error}")

        return {"ok": False, "error": error}
