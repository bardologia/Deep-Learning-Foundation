from __future__ import annotations

import re
from pathlib import Path


class ModelNoteLibrary:

    OPERATIONAL_PAREN   = re.compile(r"\s*\([^()]*(?:\d{4}-\d{2}-\d{2}|referee|first[- ]pass|regression|corrected|reviewed|audit)[^()]*\)", re.IGNORECASE)
    OPERATIONAL_MARKERS = re.compile(r"\d{4}-\d{2}-\d{2}|referee|first[- ]pass|smoke[- ]test|regression|review date|reviewed|corrected on|correction|audit|flagged", re.IGNORECASE)
    OPERATIONAL_HEADING = re.compile(r"correction|checkpoint continuity|review", re.IGNORECASE)

    def note(self, key: str) -> dict | None:
        note_files = self._note_files()
        filename   = note_files.get(key)
        if filename is None:
            return None

        path = self._notes_dir() / filename
        if not path.exists():
            return None

        markdown = self._strip_operational(path.read_text(encoding="utf-8"))
        links    = {name[:-3]: model_key for model_key, name in note_files.items()}
        return {"key": key, "note": filename[:-3], "markdown": markdown, "links": links}

    def _note_files(self) -> dict[str, str]:
        return self.NOTE_FILES

    def _notes_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parent.parent
        repo_dir  = repo_root / "notes" / "models"
        if repo_dir.is_dir():
            return repo_dir

        vault_root = repo_root.parent.parent
        return vault_root / "notes" / "DLR-TomoSAR" / "Models"

    def _strip_operational(self, markdown: str) -> str:
        out        = []
        skip_level = None
        for line in markdown.split("\n"):
            heading = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if heading:
                level = len(heading.group(1))
                if skip_level is not None and level <= skip_level:
                    skip_level = None
                if skip_level is None and self.OPERATIONAL_HEADING.search(heading.group(2)):
                    skip_level = level
                    continue
            if skip_level is not None:
                continue

            cleaned = self.OPERATIONAL_PAREN.sub("", line)
            if self.OPERATIONAL_MARKERS.search(cleaned):
                sentences = re.split(r"(?<=[.!?])\s+", cleaned)
                kept      = [s for s in sentences if not self.OPERATIONAL_MARKERS.search(s)]
                cleaned   = " ".join(kept)
                if not cleaned.strip() or cleaned.strip() in ("|", "-", ">"):
                    continue
            out.append(cleaned)

        collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
        return collapsed


class ModelDefaultsLibrary(ModelNoteLibrary):

    DISPLAY_DEFAULTS   = {}
    NORMALIZATION_ATTR = "normalization"

    def collect(self) -> list[dict]:
        families = self._families()
        defaults = self._resolve_defaults()

        for family in families:
            for model in family["models"]:
                activation, normalization = defaults[model["key"]]
                model["activation"]    = activation
                model["normalization"] = normalization

        return families

    def _resolve_defaults(self) -> dict[str, tuple[str, str]]:
        resolved = {}

        for key, class_name in self.CONFIG_CLASSES.items():
            if key in self.DISPLAY_DEFAULTS:
                resolved[key] = self.DISPLAY_DEFAULTS[key]
                continue

            config        = getattr(self.CONFIG_MODULE, class_name)()
            activation    = getattr(config, "activation", self.FALLBACK_ACTIVATION)
            normalization = getattr(config, self.NORMALIZATION_ATTR, self.FALLBACK_NORMALIZATION)
            resolved[key] = (str(activation).lower(), str(normalization).lower())

        return resolved
