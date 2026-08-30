"""Shared discovery and serving of analysis payloads for the web UI results tabs.

An analysis pipeline writes a schema-versioned JSON payload into
``<params run>/<subdir>/<timestamp>/``; this browser scans a datasets root for
those payloads and serves them to the matching interactive tab, refusing reads
outside the catalogued roots and payloads written under another schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from catalog_roots import CatalogRoots
from web_logger    import WebLogger


class AnalysisPayloadBrowser:
    """Lists and reads one pipeline's analysis payloads under catalogued dataset roots.

    Subclasses declare the payload contract and naming; the scan, containment
    and schema-rejection behaviour is shared.

    Attributes:
        roots     : Catalogue confining report reads to scanned roots.
        logger    : Web UI logger.
        LABEL     : Log prefix of the tab.
        SUBDIR    : Analysis directory name scanned beneath each parameter run.
        MARKER    : Payload filename identifying a completed analysis.
        SCHEMA    : Payload schema version the tab renders; anything else is rejected.
        COUNT_KEY : Payload list whose length is logged on a successful read.
        ENTRY     : Entry-point name named in the schema-mismatch rerun hint.
    """

    LABEL     : str
    SUBDIR    : str
    MARKER    : str
    SCHEMA    : int
    COUNT_KEY : str
    ENTRY     : str

    def __init__(self, logger: WebLogger) -> None:
        """Builds the browser with its own root catalogue.

        Args:
            logger : Web UI logger.
        """
        self.roots  = CatalogRoots()
        self.logger = logger

    def analyses(self, base: str) -> dict:
        """Lists the completed analyses under a datasets root.

        Args:
            base : Datasets root to scan; registered in the catalogue on success.

        Returns:
            Dictionary with "ok"; on success also "root" and "entries" (id, run,
            group, stamp per analysis, newest first), on failure "error" and an
            empty "entries" list.
        """
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
        """Reads the payload of one analysis.

        Args:
            analysis_id : Analysis directory as listed by `analyses`.

        Returns:
            Dictionary with "ok" and the payload under "payload" on success, or
            "error" when the directory is unknown, outside every catalogued
            root, missing its payload file, or carrying a payload schema the tab
            no longer renders.
        """
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
        """Logs a rejected payload read and returns its error result."""
        self.logger.warning(f"{self.LABEL}: {target} refused: {error}")

        return {"ok": False, "error": error}
