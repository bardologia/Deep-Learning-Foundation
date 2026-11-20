
from __future__ import annotations

import argparse
import ast
import json
import sys
import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from enum        import Enum
from pathlib     import Path

from .detacher        import Detacher
from .machine_profile import FieldRule, MachineProfile

_SUPPORTED_TYPES = (bool, int, float, str, Path, list, tuple, dict)


@dataclass(frozen=True)
class ConfigSource:

    path   : str
    origin : str
    before : object
    after  : object


class ConfigCli:

    parsed : "ConfigCli | None" = None

    BOOTSTRAP_SWITCHES = ("--trial",)
    BOOTSTRAP_OPTIONAL = ("--worker", "--resume")
    BOOTSTRAP_VALUED   = (
        "--gpu", "--model", "--n-trials", "--worker-index", "--study-name", "--storage-url",
        "--run-tag", "--run-dir", "--run-name", "--fold", "--split", "--loss-component", "--config",
    )

    def __init__(self, config, description: str | None = None, profile: MachineProfile | None = None) -> None:
        self.config              = config
        self.overrides : dict    = {}
        self.profile             = MachineProfile.current() if profile is None else profile
        self.sources             = self.apply_profile(config, self.profile)
        self.profile_overrides   = {source.path: source.after for source in self.sources}
        self.unmatched_overrides = self.unmatched_config_overrides(config, self.profile)
        self.parser              = argparse.ArgumentParser(description=description, add_help=False, allow_abbrev=False)

        self.parser.add_argument("-h", "--help", action="store_true", dest="_help")
        self.parser.add_argument("--help-config", action="store_true", dest="_help_config")
        self.parser.add_argument("--detach", "--nohup", action="store_true", dest="_detach")

        declared = {"-h", "--help", "--help-config", "--detach", "--nohup"}

        for path, value in self._leaves(config):
            if not self.is_editable(value):
                continue

            options = [f"--{path}"]
            dashed  = f"--{path.replace('_', '-')}"
            if dashed not in options:
                options.append(dashed)

            self.parser.add_argument(*options, dest=path, type=str, default=None)
            declared.update(options)

        self._declare_bootstrap_flags(declared)

    def _declare_bootstrap_flags(self, declared: set) -> None:
        for flag in self.BOOTSTRAP_SWITCHES + self.BOOTSTRAP_OPTIONAL + self.BOOTSTRAP_VALUED:
            if flag in declared:
                continue

            dest = "_bootstrap_" + flag[2:].replace("-", "_")

            if flag in self.BOOTSTRAP_SWITCHES:
                self.parser.add_argument(flag, action="store_true", dest=dest)
            elif flag in self.BOOTSTRAP_OPTIONAL:
                self.parser.add_argument(flag, nargs="?", dest=dest)
            else:
                self.parser.add_argument(flag, dest=dest)

    def apply(self, argv: list[str] | None = None):
        args = self.parser.parse_args(argv)

        if getattr(args, "_help", False) or getattr(args, "_help_config", False):
            self._print_config_help()
            sys.exit(0)

        if getattr(args, "_detach", False):
            Detacher().ensure()

        for path, current in list(self._leaves(self.config)):
            raw = getattr(args, path, None)
            if raw is None:
                continue

            value = self._coerce(raw, current, self._annotation(self.config, path))
            self.set_path(self.config, path, value)
            self.overrides[path] = value
            self.sources.append(ConfigSource(path, "command line", current, value))

        if argv is None:
            ConfigCli.parsed = self

        return self.config

    @classmethod
    def apply_profile(cls, config, profile: MachineProfile) -> list[ConfigSource]:
        anchors = profile.path_anchors()
        rules   = profile.field_rules()
        applied = []

        if not anchors and not rules:
            return applied

        for path, value, owners in cls._owned_leaves(config):
            rewritten, origin = cls._profile_value(path.rsplit(".", 1)[-1], value, owners, anchors, rules)

            if rewritten is value or (type(rewritten) is type(value) and rewritten == value):
                continue

            cls.set_path(config, path, rewritten)
            applied.append(ConfigSource(path, origin, value, rewritten))

        return applied

    @classmethod
    def unmatched_config_overrides(cls, config, profile: MachineProfile) -> list[str]:
        matched = set()

        for path, _value, owners in cls._owned_leaves(config):
            field = path.rsplit(".", 1)[-1]
            for owner in owners:
                matched.add(f"{owner}.{field}")

        return sorted(key for key in profile.config_overrides if key not in matched)

    @classmethod
    def _owned_leaves(cls, config, prefix: str = ""):
        owners = tuple(klass.__name__ for klass in type(config).__mro__ if is_dataclass(klass))

        for f in fields(config):
            value = getattr(config, f.name)
            path  = f"{prefix}{f.name}"

            if is_dataclass(value):
                yield from cls._owned_leaves(value, prefix=f"{path}.")
            else:
                yield path, value, owners

    @classmethod
    def _profile_value(cls, field: str, value, owners: tuple, anchors: list, rules: list) -> tuple:
        current = value
        origin  = None

        if isinstance(current, Path):
            rebased = MachineProfile.rebase(str(current), anchors)
            if rebased is not None:
                current = Path(rebased)
                origin  = "profile rebase"
        elif isinstance(current, str) and current:
            rebased = MachineProfile.rebase(current, anchors)
            if rebased is not None:
                current = rebased
                origin  = "profile rebase"

        for rule in rules:
            if rule.field != field or (rule.owner is not None and rule.owner not in owners):
                continue
            if rule.expects is not None and (not isinstance(current, rule.expects) or isinstance(current, bool)):
                continue

            current = cls._ruled_value(current, rule)
            origin  = f"profile {rule.kind}"

        return current, origin

    @classmethod
    def _ruled_value(cls, current, rule: FieldRule):
        if rule.kind == "cap":
            return min(current, int(rule.value))

        if isinstance(current, Path):
            return Path(rule.value)
        if isinstance(current, Enum):
            return type(current)(rule.value)
        if isinstance(current, tuple):
            return tuple(rule.value)

        return rule.value

    @classmethod
    def _leaves(cls, config, prefix: str = ""):
        for path, value, _section, _owner in cls.detailed_leaves(config, prefix=prefix):
            yield path, value

    @classmethod
    def detailed_leaves(cls, config, prefix: str = "", section: str = "", section_class: str | None = None):
        owner = section_class or type(config).__name__

        for f in fields(config):
            value = getattr(config, f.name)
            path  = f"{prefix}{f.name}"

            if is_dataclass(value):
                yield from cls.detailed_leaves(value, prefix=f"{path}.", section=path, section_class=type(value).__name__)
            else:
                yield path, value, section, owner

    @classmethod
    def is_editable(cls, value) -> bool:
        return value is None or isinstance(value, (*_SUPPORTED_TYPES, Enum))

    def _coerce(self, raw: str, current, annotation=None):
        if isinstance(current, Enum):
            return type(current)(raw.strip())

        if isinstance(current, bool):
            return self._parse_bool(raw)

        if isinstance(current, int):
            return int(raw)
        if isinstance(current, float):
            return float(raw)
        if isinstance(current, Path):
            return Path(raw)
        if isinstance(current, list):
            return self._parse_sequence(raw, list)

        if isinstance(current, dict):
            return ast.literal_eval(raw)

        if isinstance(current, tuple):
            return self._parse_sequence(raw, tuple)

        if current is None:
            return self._coerce_by_annotation(raw, annotation)

        return raw

    @staticmethod
    def _parse_sequence(raw: str, container):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return container(token.strip() for token in raw.split(",") if token.strip())

        return container(parsed) if isinstance(parsed, (list, tuple)) else container([parsed])

    @staticmethod
    def _parse_bool(raw: str) -> bool:
        lowered = raw.strip().lower()
        return {"true": True, "1": True, "yes": True, "on": True, "false": False, "0": False, "no": False, "off": False}[lowered]

    @classmethod
    def _coerce_by_annotation(cls, raw: str, annotation):
        target = cls._unwrap_optional(annotation)

        if target is bool:
            return cls._parse_bool(raw)
        if target is int:
            return int(raw)
        if target is float:
            return float(raw)
        if target is str:
            return raw
        if target is Path:
            return Path(raw)
        if isinstance(target, type) and issubclass(target, Enum):
            return target(raw.strip())

        origin = typing.get_origin(target)
        if origin in (list, tuple) or target in (list, tuple):
            return cls._parse_sequence(raw, list if origin is list or target is list else tuple)

        if origin is dict or target is dict:
            return ast.literal_eval(raw)

        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw

    @staticmethod
    def _annotation(config, path: str):
        parts  = path.split(".")
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        return typing.get_type_hints(type(target)).get(parts[-1])

    @staticmethod
    def _unwrap_optional(annotation):
        if annotation is None:
            return None

        if typing.get_origin(annotation) in (typing.Union, types.UnionType):
            concrete = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
            return concrete[0] if len(concrete) == 1 else None

        return annotation

    def sources_mapping(self) -> dict:
        return {
            "machine_profile"            : {"path": None if self.profile.path is None else str(self.profile.path), "state": self.profile.state},
            "leaves"                     : [{"path": source.path, "origin": source.origin, "before": self._plain(source.before), "after": self._plain(source.after)} for source in self.sources],
            "unmatched_config_overrides" : list(self.unmatched_overrides),
        }

    def report(self, logger) -> None:
        from_profile = [source for source in self.sources if source.origin != "command line"]

        logger.kv_table({
            "Machine profile"            : self.profile.describe(),
            "Defaults rewritten"         : len(from_profile),
            "Command-line overrides"     : len(self.overrides),
            "Unmatched config_overrides" : ", ".join(self.unmatched_overrides) or "none",
        }, title="Configuration sources")

        if not self.sources:
            return

        rows = [{"Leaf": source.path, "Origin": source.origin, "Before": self._plain(source.before), "After": self._plain(source.after)} for source in self.sources]
        logger.metrics_table(rows, columns=["Leaf", "Origin", "Before", "After"])

    def _print_config_help(self) -> None:
        rows  = [(path, type(value).__name__ if value is not None else "any", repr(value)) for path, value in self._leaves(self.config)]
        width = max(len(path) for path, _, _ in rows)

        print(f"Configuration overrides for {type(self.config).__name__} (pass as --<path> <value>):")
        for path, type_name, default in rows:
            origin = "  [machine profile]" if path in self.profile_overrides else ""
            print(f"  --{path:<{width}}  {type_name:<6}  default: {default}{origin}")
        if self.profile.exists:
            print(f"Machine profile: {self.profile.path} ({len(self.profile_overrides)} default(s) rewritten)")
        print("Execution flags:")
        print("  --detach (alias --nohup)  relaunch detached from the terminal, output to logs/<script>_<stamp>.out")

    @staticmethod
    def set_path(config, path: str, value) -> None:
        parts  = path.split(".")
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)

    @classmethod
    def apply_overrides(cls, config, overrides: dict):
        for path, value in overrides.items():
            cls.set_path(config, path, value)
        return config

    @staticmethod
    def _plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return list(value)

        return value

    @classmethod
    def to_mapping(cls, config) -> dict:
        mapping = {}
        for path, value in cls._leaves(config):
            if cls.is_editable(value):
                mapping[path] = cls._plain(value)
        return mapping

    @classmethod
    def save_resolved(cls, config, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.to_mapping(config), f, indent=2)
        return path

    @classmethod
    def load_worker_config(cls, config, run_tag: str, run_dir: str | None):
        base = Path(run_dir) if run_dir else Path(config.paths.log_base_dir) / run_tag
        return cls.load_resolved(config, base / "pipeline" / "resolved_config.json")

    @classmethod
    def load_resolved(cls, config, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        for leaf, current in list(cls._leaves(config)):
            if leaf not in mapping:
                continue

            value = mapping[leaf]
            if isinstance(current, Path) and isinstance(value, str):
                value = Path(value)
            elif isinstance(current, Enum) and not isinstance(value, Enum):
                value = type(current)(value)
            elif isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            elif current is None and isinstance(value, str) and cls._unwrap_optional(cls._annotation(config, leaf)) is Path:
                value = Path(value)

            cls.set_path(config, leaf, value)

        return config

    @staticmethod
    def to_argv(overrides: dict) -> list[str]:
        argv = []
        for path, value in overrides.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, Enum):
                rendered = str(value.value)
            elif isinstance(value, tuple):
                rendered = str(list(value))
            else:
                rendered = str(value)
            argv += [f"--{path}", rendered]
        return argv
