from __future__ import annotations

import argparse
import ast
import json
import types
import typing
from dataclasses import fields, is_dataclass
from enum        import Enum
from pathlib     import Path

from .detacher import Detacher

_SUPPORTED_TYPES = (bool, int, float, str, Path, list, tuple, dict)


class ConfigCli:

    BOOTSTRAP_FLAGS = (
        "--help-config",
        "--detach", "--nohup",
        "--gpu",
        "--trial", "--worker", "--resume",
        "--model",
        "--n-trials", "--study-name", "--storage-url",
        "--run-tag", "--run-dir",
        "--fold", "--split",
    )

    def __init__(self, config, description: str | None = None) -> None:
        self.config    = config
        self.overrides : dict = {}
        self.parser    = argparse.ArgumentParser(description=description, add_help=False, allow_abbrev=False)

        self.parser.add_argument("-h", "--help", action="store_true", dest="_help")
        self.parser.add_argument("--help-config", action="store_true", dest="_help_config")
        self.parser.add_argument("--detach", "--nohup", action="store_true", dest="_detach")

        for path, value in self._leaves(config):
            if value is not None and not isinstance(value, (*_SUPPORTED_TYPES, Enum)):
                continue

            options = [f"--{path}"]
            dashed  = f"--{path.replace('_', '-')}"
            if dashed not in options:
                options.append(dashed)

            self.parser.add_argument(*options, dest=path, type=str, default=None)

    def apply(self, argv: list[str] | None = None):
        args, leftover = self.parser.parse_known_args(argv)

        self._reject_unknown_options(leftover)

        if getattr(args, "_help", False) or getattr(args, "_help_config", False):
            self._print_config_help()
            raise SystemExit(0)

        if getattr(args, "_detach", False):
            Detacher().ensure()

        for path, current in list(self._leaves(self.config)):
            raw = getattr(args, path, None)
            if raw is None:
                continue

            value = self._coerce(raw, current, self._annotation(self.config, path))
            self.set_path(self.config, path, value)
            self.overrides[path] = value

        return self.config

    def _reject_unknown_options(self, leftover: list[str]) -> None:
        offenders = []

        for token in leftover:
            if not token.startswith("--") and not token.startswith("-"):
                continue

            name = token.split("=", 1)[0]
            if name in self.BOOTSTRAP_FLAGS:
                continue

            offenders.append(name)

        if offenders:
            keys = ", ".join(sorted(set(offenders)))
            raise ValueError(f"Unrecognized override option(s): {keys}. Known overrides: --<path> from {type(self.config).__name__}; bootstrap flags: {', '.join(self.BOOTSTRAP_FLAGS)}")

    @classmethod
    def _leaves(cls, config, prefix: str = ""):
        for f in fields(config):
            value = getattr(config, f.name)
            path  = f"{prefix}{f.name}"

            if is_dataclass(value):
                yield from cls._leaves(value, prefix=f"{path}.")
            else:
                yield path, value

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
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return [token.strip() for token in raw.split(",") if token.strip()]
            return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]

        if isinstance(current, dict):
            return ast.literal_eval(raw)

        if isinstance(current, tuple):
            parsed = ast.literal_eval(raw)
            return tuple(parsed) if isinstance(parsed, (list, tuple)) else (parsed,)

        if current is None:
            return self._coerce_by_annotation(raw, annotation)

        return raw

    @staticmethod
    def _parse_bool(raw: str) -> bool:
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"Cannot parse boolean from '{raw}'")

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

    def _print_config_help(self) -> None:
        rows  = [(path, type(value).__name__ if value is not None else "any", repr(value)) for path, value in self._leaves(self.config)]
        width = max(len(path) for path, _, _ in rows)

        print(f"Configuration overrides for {type(self.config).__name__} (pass as --<path> <value>):")
        for path, type_name, default in rows:
            print(f"  --{path:<{width}}  {type_name:<6}  default: {default}")
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

    @classmethod
    def to_mapping(cls, config) -> dict:
        mapping = {}
        for path, value in cls._leaves(config):
            if isinstance(value, Path):
                mapping[path] = str(value)
            elif isinstance(value, Enum):
                mapping[path] = value.value
            elif isinstance(value, tuple):
                mapping[path] = list(value)
            elif value is None or isinstance(value, _SUPPORTED_TYPES):
                mapping[path] = value
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
        if not Path(path).exists():
            raise FileNotFoundError(f"Resolved config not found at {path} for {type(config).__name__}")

        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        known   = set(cls.to_mapping(config))
        unknown = sorted(key for key in mapping if key not in known)
        if unknown:
            raise KeyError(f"Unknown key(s) in resolved config {path}: {', '.join(unknown)}. Known keys belong to {type(config).__name__}")

        missing = sorted(known - set(mapping))
        if missing:
            raise KeyError(f"Resolved config {path} is missing key(s): {', '.join(missing)}. It was written by an older {type(config).__name__}; regenerate the run instead of silently resuming it under today's defaults")

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
