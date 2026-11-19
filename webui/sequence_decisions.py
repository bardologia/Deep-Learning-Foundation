
from __future__ import annotations

import json
from pathlib import Path

from config_value_render import ConfigValueRenderer
from web_logger          import WebLogger


class SequenceDecisions:

    FAMILY_NAMESPACES = {
        "backbone"            : "",
        "dual"                : "dual",
        "jepa"                : "jepa",
        "profile_autoencoder" : "autoencoder",
        "image_autoencoder"   : "image_autoencoder",
    }

    FAMILY_SCOPED_PATHS = ("model_overrides",)

    def __init__(self, resolver, launcher, logger: WebLogger) -> None:
        self.resolver = resolver
        self.launcher = launcher
        self.logger   = logger

    def clean(self, block: dict) -> list:
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
        defaults = {}

        for script in sorted(scripts):
            resolved = self.resolver.resolve(script, self.launcher.preferred_interpreter(script))
            if not resolved.get("ok"):
                raise ValueError(f"cannot resolve the configuration of '{script}' to apply the sequence decisions: {resolved.get('error', 'unknown error')}")
            defaults[script] = {leaf["path"]: str(leaf["value"]) for leaf in resolved["leaves"]}

        return defaults

    def _namespace(self, block: dict, defaults: dict) -> str:
        if "training_type" not in defaults:
            return ""

        family = block["overrides"].get("training_type", defaults["training_type"])
        if family not in self.FAMILY_NAMESPACES:
            raise ValueError(f"cannot apply the sequence decisions to block '{block['title']}': training_type '{family}' is none of {sorted(self.FAMILY_NAMESPACES)}")

        return self.FAMILY_NAMESPACES[family]

    def _family(self, block: dict, defaults: dict) -> str:
        if "training_type" in defaults:
            return block["overrides"].get("training_type", defaults["training_type"])

        script = block["script"]

        return script.removeprefix("train_") if script.startswith("train_") else ""

    def _resolve(self, path: str, namespace: str, defaults: dict) -> str:
        namespaced = f"{namespace}.{path}"

        if namespace and namespaced in defaults:
            return namespaced

        return path if path in defaults else ""

    def _stamp(self, block: dict, path: str, value: str, note: str, defaults: dict) -> None:
        block["field_notes"][path] = note

        if value == defaults[path]:
            block["overrides"].pop(path, None)
        else:
            block["overrides"][path] = value

    def apply(self, blocks: list, order: list) -> str:
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


class TuningVerdict:

    SCRIPT      = "tune"
    MODEL_AXIS  = "model_overrides"
    BATCH_AXIS  = "training.batch_size"
    BATCH_PARAM = "batch_size"
    FILENAME    = "best_config.json"
    PROGRESS    = "study_progress.json"

    def records(self, block: dict) -> bool:
        return block["script"] == self.SCRIPT and bool(block["decisions"])

    def axes(self, block: dict) -> list:
        paths   = [decision["path"] for decision in block["decisions"]]
        unknown = [path for path in paths if path not in (self.MODEL_AXIS, self.BATCH_AXIS)]
        if unknown:
            raise ValueError(f"a tuning block records only the '{self.MODEL_AXIS}' and '{self.BATCH_AXIS}' axes, but '{block['title']}' declares {unknown}; drop those axes")

        return paths

    def values(self, params: dict, axes: list) -> dict:
        model  = {name: value for name, value in params.items() if name != self.BATCH_PARAM}
        values = {}

        for axis in axes:
            if axis == self.MODEL_AXIS:
                if not model:
                    raise ValueError(f"the study sampled no model parameter, so there is no '{self.MODEL_AXIS}' verdict to record; drop the axis or search a model group")
                values[axis] = ConfigValueRenderer.render(model)
            else:
                if self.BATCH_PARAM not in params:
                    raise ValueError(f"the block declares '{self.BATCH_AXIS}' but the study did not sample the batch size; drop the axis or search the batch size")
                values[axis] = str(int(params[self.BATCH_PARAM]))

        if model and self.MODEL_AXIS not in axes:
            raise ValueError(f"the study sampled model parameters but the block declares no '{self.MODEL_AXIS}' axis, so that verdict would be lost; declare it")
        if self.BATCH_PARAM in params and self.BATCH_AXIS not in axes:
            raise ValueError(f"the study sampled the batch size but the block declares no '{self.BATCH_AXIS}' axis, so that verdict would be lost; declare it")

        return values

    def run_dir(self, overrides: dict, leaves: dict, run_dir: str) -> Path:
        run_tag = overrides.get("run_tag", "") or leaves.get("run_tag", "")
        if run_tag in ("", "None"):
            raise ValueError("a tuning block that records a verdict needs a fixed 'run_tag' to locate its best configuration; set one on the block")

        base = overrides.get("paths.log_base_dir", "") or leaves.get("paths.log_base_dir", "") or run_dir

        return Path(base) / run_tag

    def read(self, tuning_dir: Path) -> dict:
        found = sorted(Path(tuning_dir).glob(f"*/{self.FILENAME}"))
        if len(found) != 1:
            raise ValueError(f"expected exactly one {self.FILENAME} under {tuning_dir} to record as the verdict, found {len(found)}{': ' + ', '.join(str(path) for path in found) if found else ''}; a tuning block that records a verdict tunes one model")

        return dict(json.loads(found[0].read_text(encoding="utf-8"))["params"])
