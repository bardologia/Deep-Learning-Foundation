"""Shared base classes for the web UI model libraries.

Supplies the note lookup and architecture-field introspection that each
concrete model library (backbone, dual, autoencoder, unrolled) specialises by
declaring its own config module, config classes and note files.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib     import Path


class ModelNoteLibrary:
    """Serves the curated markdown note attached to each model key.

    Subclasses provide ``NOTE_FILES``, a mapping from model key to the markdown
    filename inside the notes directory. Operational bookkeeping (review dates,
    correction logs) is stripped before the note reaches the browser.
    """

    OPERATIONAL_PAREN   = re.compile(r"\s*\([^()]*(?:\d{4}-\d{2}-\d{2}|referee|first[- ]pass|regression|corrected|reviewed|audit)[^()]*\)", re.IGNORECASE)
    OPERATIONAL_MARKERS = re.compile(r"\d{4}-\d{2}-\d{2}|referee|first[- ]pass|smoke[- ]test|regression|review date|reviewed|corrected on|correction|audit|flagged", re.IGNORECASE)
    OPERATIONAL_HEADING = re.compile(r"correction|checkpoint continuity|review", re.IGNORECASE)

    def note(self, key: str) -> dict | None:
        """Returns the cleaned note payload for a model key.

        Args:
            key : Model key as used by the library's ``NOTE_FILES`` mapping.

        Returns:
            A dict with ``key``, ``note`` (filename without extension),
            ``markdown`` (operational passages removed) and ``links`` (note stem
            to model key, for resolving wikilinks), or ``None`` when the key has
            no note or the file is missing on disk.
        """
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
        """Returns the model key to note filename mapping declared by the subclass."""
        return self.NOTE_FILES

    def _notes_dir(self) -> Path:
        """Returns the in-repo models notes directory."""
        return Path(__file__).resolve().parent.parent / "notes" / "models"

    def _strip_operational(self, markdown: str) -> str:
        """Removes review and correction bookkeeping from a note's markdown.

        Sections whose heading matches the operational pattern are dropped whole,
        parenthetical review markers are deleted inline, and remaining sentences
        carrying such markers are filtered out sentence by sentence.

        Args:
            markdown : Raw note text.

        Returns:
            The note text with operational passages removed and runs of blank
            lines collapsed to a single blank line.
        """
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
    """Builds the model catalogue served to the launch page.

    Reads each model's config dataclass to expose its default activation,
    normalization and the tunable architecture fields, together with the preset
    grids declared by ``tunable_arch_params``.
    """

    DISPLAY_DEFAULTS       = {}
    NORMALIZATION_ATTR     = "normalization"
    ARCH_EXCLUDED          = ()
    ARCH_EXCLUDED_BY_MODEL = {}
    ARCH_HINTS             = {}

    STRING_CHOICES = {
        "activation"     : ("relu", "leaky_relu", "gelu", "elu", "silu"),
        "embedding_norm" : ("none", "l2", "layernorm"),
        "init_mode"      : ("default", "kaiming", "xavier"),
        "normalization"  : ("batch", "instance", "group", "none"),
        "upsample_mode"  : ("convtranspose", "bilinear"),
    }

    def collect(self) -> list[dict]:
        """Returns the model families enriched with defaults and architecture fields.

        Returns:
            The family list from ``_families()``, where every model entry gains
            ``activation``, ``normalization`` and an ``arch`` list of editable
            architecture field descriptors.
        """
        families = self._families()
        defaults = self._resolve_defaults()

        for family in families:
            for model in family["models"]:
                activation, normalization = defaults[model["key"]]
                model["activation"]    = activation
                model["normalization"] = normalization
                model["arch"]          = self._arch_fields(model["key"])

        return families

    def _arch_fields(self, key: str) -> list[dict]:
        """Returns the editable architecture fields of a model's config dataclass.

        Scalar fields are emitted with their default, kind and hint; string
        fields gain ``options`` when the default belongs to a known choice set;
        numeric and list fields gain ``presets`` from the tunable grid.

        Args:
            key : Model key naming the config class in ``CONFIG_CLASSES``.

        Returns:
            One descriptor dict per exposed field, in dataclass declaration order.
        """
        config_class = getattr(self.CONFIG_MODULE, self.CONFIG_CLASSES[key])
        config       = config_class()
        tunable      = config_class.tunable_arch_params()
        excluded     = set(self.ARCH_EXCLUDED) | set(self.ARCH_EXCLUDED_BY_MODEL.get(key, ()))

        entries = []
        for spec in fields(config_class):
            if spec.name in excluded:
                continue

            value = getattr(config, spec.name)
            grid  = tunable.get(spec.name, {})
            hint  = self.ARCH_HINTS.get(spec.name, "")

            if isinstance(value, (bool, int, float, str)):
                entry = {"name": spec.name, "default": value, "kind": type(value).__name__, "hint": hint}

                if isinstance(value, bool):
                    pass
                elif isinstance(value, str):
                    options = grid["choices"] if grid.get("type") == "categorical" else self.STRING_CHOICES.get(spec.name, ())
                    if value in options:
                        entry["options"] = [str(option) for option in options]
                elif grid.get("type") == "categorical":
                    entry["presets"] = list(grid["choices"])

                entries.append(entry)
                continue

            if self._is_scalar_sequence(value) or (value is None and "list" in str(spec.type)):
                entry = {"name": spec.name, "default": list(value) if value is not None else None, "kind": "list", "hint": hint}

                if grid.get("type") == "indexed_categorical":
                    entry["presets"] = [list(choice) for choice in grid["choices"]]

                entries.append(entry)

        return entries

    @staticmethod
    def _is_scalar_sequence(value) -> bool:
        """Returns whether the value is a non-empty list or tuple of scalars."""
        return isinstance(value, (list, tuple)) and len(value) > 0 and all(isinstance(item, (bool, int, float, str)) for item in value)

    def _resolve_defaults(self) -> dict[str, tuple[str, str]]:
        """Returns the lowercase (activation, normalization) default per model key.

        Keys listed in ``DISPLAY_DEFAULTS`` take their pair from there; the rest
        are read off a freshly constructed config instance, falling back to
        ``FALLBACK_ACTIVATION`` and ``FALLBACK_NORMALIZATION``.
        """
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
