"""Propagation of experiment verdicts across the blocks of a sequence.

A sequence block may declare the configuration axes it puts under test: for
each axis, the value the sequence runs with while the question is still open,
the candidate values the experiment compares, and the verdict once it lands.
Declaring the axis is what freezes it downstream: every block after the
deciding one is stamped with the verdict, and every block that is not yet
downstream of it keeps the pre-verdict value, so a verdict is recorded once
instead of being copied by hand into each later block.

The stamping only touches blocks whose entry point actually exposes the axis,
read from the resolved configuration of each entry, and only writes an override
when the value differs from that entry's own default. Each stamped field also
carries a note, ``frozen`` after the deciding block and ``waiting`` before it,
which the block editor renders as provenance colouring.

An entry that runs several training families keeps each family's settings under
a field of its own, so the axis a training entry exposes bare, ``ae_model_name``,
is ``autoencoder.ae_model_name`` on the cross-validation entry. Such an entry
announces the family it runs through a ``training_type`` leaf, and a decided axis
is resolved against that family's namespace first and against the bare path
second, so a family axis never lands on the identically named field of another
family while the axes no family namespaces, ``training.seed`` and the rest of the
shared configuration, keep resolving bare.
"""

from __future__ import annotations

from web_logger import WebLogger


class SequenceDecisions:
    """Stamps declared decision axes onto the blocks of a sequence.

    Attributes:
        FAMILY_NAMESPACES   : Config field a training family's settings live under on an entry running several families, keyed by ``training_type``.
        FAMILY_SCOPED_PATHS : Axes whose value names fields of one model family's architecture, so a verdict on them governs only the blocks training that family.
        resolver            : Config resolver supplying the leaves each entry point exposes.
        launcher            : Run launcher supplying the interpreter an entry resolves under.
        logger              : Console logger.
    """

    FAMILY_NAMESPACES = {
        "backbone"            : "",
        "dual"                : "dual",
        "jepa"                : "jepa",
        "profile_autoencoder" : "autoencoder",
        "image_autoencoder"   : "image_autoencoder",
    }

    FAMILY_SCOPED_PATHS = ("model_overrides",)

    def __init__(self, resolver, launcher, logger: WebLogger) -> None:
        """Stores the resolver, launcher and logger."""
        self.resolver = resolver
        self.launcher = launcher
        self.logger   = logger

    def clean(self, block: dict) -> list:
        """Returns one block's decision list normalised to the stored schema.

        Args:
            block : Raw block payload whose ``decisions`` entry is being cleaned.

        Returns:
            List of decisions, each with ``path``, ``options``, ``waiting`` and
            ``choice``; entries without a path are dropped, a path declared
            twice on the same block keeps its first declaration, and repeated
            candidate values collapse to one.
        """
        cleaned = []
        seen    = set()

        for decision in list(block.get("decisions") or []):
            path = str(dict(decision).get("path", "")).strip()
            if not path or path in seen:
                continue

            seen.add(path)
            cleaned.append({
                "path"    : path,
                "options" : list(dict.fromkeys(str(option) for option in list(decision.get("options") or []))),
                "waiting" : str(decision.get("waiting", "")),
                "choice"  : str(decision.get("choice", "")),
            })

        return cleaned

    def _descendants(self, blocks: list) -> dict:
        """Returns the set of blocks reachable from each block, keyed by block id.

        Args:
            blocks : Cleaned block list whose predecessor ids all exist.
        """
        successors = {block["block_id"]: [] for block in blocks}
        for block in blocks:
            for pred in block["predecessors"]:
                successors[pred].append(block["block_id"])

        reach = {}
        for block in blocks:
            seen  = set()
            stack = list(successors[block["block_id"]])
            while stack:
                block_id = stack.pop()
                if block_id in seen:
                    continue
                seen.add(block_id)
                stack += successors[block_id]
            reach[block["block_id"]] = seen

        return reach

    def _defaults(self, scripts: set) -> dict:
        """Returns the resolved default value of every leaf of each script, keyed by script.

        Args:
            scripts : Entry-point keys used by the sequence's blocks.

        Raises:
            ValueError : If an entry point's configuration cannot be resolved.
        """
        defaults = {}

        for script in sorted(scripts):
            resolved = self.resolver.resolve(script, self.launcher.preferred_interpreter(script))
            if not resolved.get("ok"):
                raise ValueError(f"cannot resolve the configuration of '{script}' to apply the sequence decisions: {resolved.get('error', 'unknown error')}")
            defaults[script] = {leaf["path"]: str(leaf["value"]) for leaf in resolved["leaves"]}

        return defaults

    def _namespace(self, block: dict, defaults: dict) -> str:
        """Returns the config field one block's entry keeps its training family under.

        An entry running a single family exposes no ``training_type`` leaf and
        namespaces nothing, so every axis it exposes is a bare one.

        Args:
            block    : Block whose overrides may name the training family its entry runs.
            defaults : Resolved default values of that entry point, keyed by config path.

        Returns:
            The config field holding the family's settings, empty both for an
            entry that namespaces nothing and for the backbone family, which the
            cross-validation entry exposes at its own top level.

        Raises:
            ValueError : If the block runs a training family that has no namespace.
        """
        if "training_type" not in defaults:
            return ""

        family = block["overrides"].get("training_type", defaults["training_type"])
        if family not in self.FAMILY_NAMESPACES:
            raise ValueError(f"cannot apply the sequence decisions to block '{block['title']}': training_type '{family}' is none of {sorted(self.FAMILY_NAMESPACES)}")

        return self.FAMILY_NAMESPACES[family]

    def _family(self, block: dict, defaults: dict) -> str:
        """Returns the training family one block trains.

        An entry running several families announces the one it runs through a
        ``training_type`` leaf; an entry running a single family is named after
        it, and an entry training nothing has no family.

        Args:
            block    : Block whose entry and overrides name the family it trains.
            defaults : Resolved default values of that entry point, keyed by config path.

        Returns:
            The training family, empty for an entry that trains no model.
        """
        if "training_type" in defaults:
            return block["overrides"].get("training_type", defaults["training_type"])

        script = block["script"]

        return script.removeprefix("train_") if script.startswith("train_") else ""

    def _resolve(self, path: str, namespace: str, defaults: dict) -> str:
        """Returns the leaf of one entry a decided axis is stamped onto, or an empty string.

        The family namespace is tried before the bare path, so an axis a training
        entry exposes bare lands on the field of the family the target block runs
        instead of on the identically named field of another family. Axes no
        family namespaces fall through to the bare path.

        Args:
            path      : Configuration path of the decided axis, as the deciding block declares it.
            namespace : Config field the target's training family lives under, empty when its entry namespaces nothing.
            defaults  : Resolved default values of the target's entry point, keyed by config path.

        Returns:
            The path the target entry exposes the axis under, and an empty string
            when it exposes the axis under neither.
        """
        namespaced = f"{namespace}.{path}"

        if namespace and namespaced in defaults:
            return namespaced

        return path if path in defaults else ""

    def _stamp(self, block: dict, path: str, value: str, note: str, defaults: dict) -> None:
        """Writes one decided axis into one block's overrides and field notes.

        The override is written only when the value differs from the entry's own
        default, so a block running the default recipe stays free of redundant
        overrides while still carrying the provenance note.

        Args:
            block    : Block being stamped, mutated in place.
            path     : Configuration path of the decided axis.
            value    : Value this block runs the axis with.
            note     : Provenance label, ``frozen`` after the deciding block and ``waiting`` before it.
            defaults : Resolved default values of this block's entry point, keyed by config path.
        """
        block["field_notes"][path] = note

        if value == defaults[path]:
            block["overrides"].pop(path, None)
        else:
            block["overrides"][path] = value

    def apply(self, blocks: list, order: list) -> str:
        """Stamps every declared decision onto the blocks it governs, and returns an error message or an empty string.

        Blocks strictly after the deciding block are stamped with the verdict and
        marked ``frozen``; every other block, the deciding one included, is
        stamped with the pre-verdict value and marked ``waiting``. A decision
        without a verdict leaves every block waiting, and one whose pre-verdict
        value is still blank stamps nothing, so declaring an axis another entry
        point exposes never writes an empty override into those blocks. When
        two blocks decide the same axis, the topologically later declaration
        wins, which on a chain is the nearest deciding ancestor.

        Each target block resolves the decided axis against its own entry, the
        namespace of the training family it runs first and the bare path second,
        so an axis a training entry exposes bare reaches the field the
        cross-validation entry keeps that family's settings under. An axis whose
        value names architecture fields of one family governs only the blocks
        training that same family, since those field names mean nothing to
        another family's architecture.

        Args:
            blocks : Cleaned block list, mutated in place; stale field notes are discarded first.
            order  : Block ids in topological order, as returned by the store.

        Returns:
            An error message when an entry point's configuration cannot be
            resolved or a block runs an unknown training family, and an empty
            string once every decision has been stamped.
        """
        for block in blocks:
            block["field_notes"] = {}

        by_id    = {block["block_id"]: block for block in blocks}
        decided  = [(block_id, by_id[block_id]) for block_id in order if by_id[block_id]["decisions"]]
        if not decided:
            return ""

        try:
            defaults   = self._defaults({block["script"] for block in blocks})
            namespaces = {block["block_id"]: self._namespace(block, defaults[block["script"]]) for block in blocks}
            families   = {block["block_id"]: self._family(block, defaults[block["script"]]) for block in blocks}
        except ValueError as error:
            return str(error)

        reach = self._descendants(blocks)

        for block_id, decider in decided:
            for decision in decider["decisions"]:
                path = decision["path"]
                for target in blocks:
                    if path in self.FAMILY_SCOPED_PATHS and families[target["block_id"]] != families[block_id]:
                        continue

                    entry_defaults = defaults[target["script"]]
                    entry_path     = self._resolve(path, namespaces[target["block_id"]], entry_defaults)
                    if not entry_path:
                        continue

                    frozen = target["block_id"] in reach[block_id] and bool(decision["choice"])
                    value  = decision["choice"] if frozen else decision["waiting"]
                    if not value:
                        continue

                    self._stamp(target, entry_path, value, "frozen" if frozen else "waiting", entry_defaults)

        return ""
