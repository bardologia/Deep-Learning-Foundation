
from __future__ import annotations

import json
import os
import shlex
import socket
from dataclasses import dataclass
from datetime    import datetime
from pathlib     import Path

from configuration.reference_layout import ReferenceLayout


@dataclass(frozen=True)
class FieldRule:

    field   : str
    kind    : str
    value   : object
    owner   : str | None = None
    expects : type | tuple | None = None


class MachineProfile:

    ENV_VAR       = "MLTEMPLATE_MACHINE_PROFILE"
    FILE_NAME     = "machine.json"
    ENV_FILE_NAME = "machine.env"
    SCHEMA        = 1

    DEFAULTS = {
        "machine"  : {"hostname": "", "conda_root": "", "env_name": "Dune", "interpreter": "", "torch_index": "", "driver": "", "cpu_count": 0, "ram_gb": 0.0, "gpus": []},
        "console"  : {"host": ReferenceLayout.CONSOLE_HOST, "port": ReferenceLayout.CONSOLE_PORT},
        "roots"    : {"repo": "", "data": "", "runs": "", "results": ""},
        "dataset"  : {"path": "", "parameters_path": ""},
        "hardware" : {"gpu_pool": [], "vram_budget_gb": 0.0, "cpu_workers": 0, "gif_workers": 0, "max_workers": 0},
        "mail"     : {"sender": "", "recipient": "", "smtp_host": "smtp.gmail.com", "smtp_port": 465},
        "ntfy"     : {"enabled": False, "topic": "", "server": "https://ntfy.sh"},
    }

    def __init__(self, path: Path | None, data: dict | None = None) -> None:
        self.path = path
        self.data = dict(data or {})

    @classmethod
    def default_path(cls) -> Path:
        return Path.home() / ".config" / "ml-template" / cls.FILE_NAME

    @classmethod
    def location(cls) -> Path | None:
        raw = os.environ.get(cls.ENV_VAR)

        if raw is None:
            return cls.default_path()
        if raw.strip() == "":
            return None

        return Path(raw).expanduser()

    @classmethod
    def current(cls) -> "MachineProfile":
        named = os.environ.get(cls.ENV_VAR)
        path  = cls.location()

        if path is None:
            return cls(path)

        if not path.is_file():
            if named is None:
                return cls(path)
            raise FileNotFoundError(f"{cls.ENV_VAR} names {path} but no file is there; fix the variable, run ./install.sh to write the profile, or set the variable to an empty value to run without one")

        return cls.load(path)

    @classmethod
    def load(cls, path: Path) -> "MachineProfile":
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"machine profile not found at {path}; run ./install.sh from the repository root")

        data   = json.loads(path.read_text(encoding="utf-8"))
        schema = data.get("schema")

        if schema != cls.SCHEMA:
            raise ValueError(f"machine profile {path} has schema {schema!r}, this code expects {cls.SCHEMA}; rerun ./install.sh to regenerate it")

        return cls(path, data)

    @property
    def exists(self) -> bool:
        return bool(self.data)

    @property
    def env_path(self) -> Path | None:
        return None if self.path is None else self.path.with_name(self.ENV_FILE_NAME)

    @property
    def state(self) -> str:
        if self.path is None:
            return "disabled"

        return "loaded" if self.exists else "absent"

    def describe(self) -> str:
        if self.path is None:
            return f"disabled by an empty {self.ENV_VAR}"
        if self.exists:
            return f"{self.path} (loaded)"

        return f"absent at {self.path}; the reference-layout defaults apply"

    def section(self, name: str) -> dict:
        return {**self.DEFAULTS[name], **self.data.get(name, {})}

    @property
    def machine(self) -> dict:
        return self.section("machine")

    @property
    def console(self) -> dict:
        return self.section("console")

    @property
    def roots(self) -> dict:
        return self.section("roots")

    @property
    def dataset(self) -> dict:
        return self.section("dataset")

    @property
    def hardware(self) -> dict:
        return self.section("hardware")

    @property
    def mail(self) -> dict:
        return self.section("mail")

    @property
    def ntfy(self) -> dict:
        return self.section("ntfy")

    @property
    def config_overrides(self) -> dict:
        return dict(self.data.get("config_overrides", {}))

    @property
    def interpreter(self) -> str:
        return self.machine["interpreter"]

    @property
    def env_name(self) -> str:
        return self.machine["env_name"]

    @property
    def gpu_pool(self) -> list:
        return [int(index) for index in self.hardware["gpu_pool"]]

    def path_anchors(self) -> list[tuple[str, str]]:
        if not self.exists:
            return []

        roots   = self.roots
        dataset = self.dataset
        pairs   = [
            (str(ReferenceLayout.PARAMETERS),   dataset["parameters_path"]),
            (str(ReferenceLayout.DATASET_DIR),  dataset["path"]),
            (str(ReferenceLayout.DATA_ROOT),    roots["data"]),
            (str(ReferenceLayout.RUNS_ROOT),    roots["runs"]),
            (str(ReferenceLayout.RESULTS_ROOT), roots["results"]),
        ]
        chosen  = [(anchor, str(target)) for anchor, target in pairs if target]

        return sorted(chosen, key=lambda pair: len(pair[0].rstrip("/")), reverse=True)

    @staticmethod
    def rebase(text: str, anchors: list[tuple[str, str]]) -> str | None:
        for anchor, target in anchors:
            root = anchor.rstrip("/")

            if text == root or text.startswith(root + "/"):
                return target.rstrip("/") + text[len(root):]

        return None

    def field_rules(self) -> list[FieldRule]:
        if not self.exists:
            return []

        hardware = self.hardware
        mail     = self.mail
        pool     = self.gpu_pool
        rules    = []

        if pool:
            rules.append(FieldRule("gpus",           "set", list(pool), expects=list))
            rules.append(FieldRule("gpu_device_ids", "set", list(pool), expects=list))
            rules.append(FieldRule("gpu",            "set", pool[0],    expects=int))

        if hardware["vram_budget_gb"]:
            rules.append(FieldRule("vram_budget_gb", "set", float(hardware["vram_budget_gb"]), expects=float))
        if hardware["cpu_workers"]:
            rules.append(FieldRule("cpu_workers", "set", int(hardware["cpu_workers"]), expects=int))
        if hardware["gif_workers"]:
            rules.append(FieldRule("gif_workers", "set", int(hardware["gif_workers"]), expects=int))
        if hardware["max_workers"]:
            for name in ("num_workers", "parameter_workers"):
                rules.append(FieldRule(name, "cap", int(hardware["max_workers"]), expects=int))

        for name in ("sender", "recipient", "smtp_host", "smtp_port"):
            if mail[name]:
                rules.append(FieldRule(name, "set", mail[name], owner="ReportMailEntryConfig"))

        for key, value in self.config_overrides.items():
            owner, _, field = key.rpartition(".")
            if not owner or not field:
                raise ValueError(f"config_overrides key {key!r} in {self.path} must be written as Class.field")
            rules.append(FieldRule(field, "set", value, owner=owner))

        return rules

    def summary(self) -> dict:
        return {
            "path"       : None if self.path is None else str(self.path),
            "exists"     : self.exists,
            "hostname"   : self.machine["hostname"],
            "env_name"   : self.env_name,
            "interpreter": self.interpreter,
            "roots"      : self.roots,
            "dataset"    : self.dataset,
            "gpu_pool"   : self.gpu_pool,
            "written_at" : self.data.get("written_at", ""),
        }

    def env_lines(self) -> list[str]:
        console = self.console
        values  = {
            "MLTEMPLATE_MACHINE_PROFILE": str(self.path),
            "MLTEMPLATE_INTERPRETER"    : self.interpreter,
            "MLTEMPLATE_ENV_NAME"       : self.env_name,
            "MLTEMPLATE_CONDA_ROOT"     : self.machine["conda_root"],
            "MLTEMPLATE_REPO_ROOT"      : self.roots["repo"],
            "MLTEMPLATE_CONSOLE_HOST"   : console["host"],
            "MLTEMPLATE_CONSOLE_PORT"   : str(console["port"]),
        }

        return [f"export {key}={shlex.quote(str(value))}" for key, value in values.items()]

    def save(self) -> Path:
        if self.path is None:
            raise RuntimeError(f"cannot save a machine profile while {self.ENV_VAR} is set to an empty value")

        self.data.setdefault("machine", {}).setdefault("hostname", socket.gethostname())
        payload = {"schema": self.SCHEMA, "written_at": datetime.now().isoformat(timespec="seconds"), **{key: value for key, value in self.data.items() if key not in ("schema", "written_at")}}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_atomic(self.path, json.dumps(payload, indent=2, default=str) + "\n")
        self._write_atomic(self.env_path, "\n".join(self.env_lines()) + "\n")

        return self.path

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        temporary = path.with_name(path.name + ".tmp")

        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        temporary.chmod(0o600)
        os.replace(temporary, path)
