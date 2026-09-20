"""Per-machine profile: the values that change from one installation to the next.

The installer (``install.sh`` and ``scripts/setup_machine.py``) writes one JSON file
per machine and user. ``ConfigCli`` reads it to rebase the checkout-relative path
defaults and the hardware-shaped defaults of every entry configuration, the web
console reads it for its interpreter, roots and notification settings, and
``webui/run.sh`` reads the shell twin written beside it. The file location follows
``MLTEMPLATE_MACHINE_PROFILE``; an empty value disables the profile entirely, which is
how the test suite keeps a developer's profile out of its expectations. Without a
profile every default keeps the reference layout, which is the checkout itself.
"""

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
    """One profile-driven rewrite of configuration leaves selected by field name.

    Attributes:
        field   : Name of the dataclass field the rule applies to.
        kind    : ``set`` replaces the value, ``cap`` lowers an integer to at most the rule value.
        value   : Value written, or the ceiling for a cap.
        owner   : Dataclass name the field must belong to, or None for every dataclass.
        expects : Type the current value must have for the rule to apply, or None for any.
    """

    field   : str
    kind    : str
    value   : object
    owner   : str | None = None
    expects : type | tuple | None = None


class MachineProfile:
    """Reads, writes and interprets the per-machine profile file.

    Attributes:
        path : Location of the profile file, or None when profiles are disabled.
        data : Loaded profile mapping; empty when no profile exists.
    """

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
        """Binds the profile to its file and content.

        Args:
            path : Profile file location, or None when profiles are disabled.
            data : Profile mapping; None or empty stands for an absent profile.
        """
        self.path = path
        self.data = dict(data or {})

    @classmethod
    def default_path(cls) -> Path:
        """Returns the profile location used when the environment names none."""
        return Path.home() / ".config" / "ml-template" / cls.FILE_NAME

    @classmethod
    def location(cls) -> Path | None:
        """Returns the profile path selected by the environment, or None when disabled.

        ``MLTEMPLATE_MACHINE_PROFILE`` names the file; set but empty it disables the
        profile; unset it falls back to ``default_path``.
        """
        raw = os.environ.get(cls.ENV_VAR)

        if raw is None:
            return cls.default_path()
        if raw.strip() == "":
            return None

        return Path(raw).expanduser()

    @classmethod
    def current(cls) -> "MachineProfile":
        """Returns the profile at the selected location, or an absent profile when there is none."""
        path = cls.location()

        if path is None or not path.is_file():
            return cls(path)

        return cls.load(path)

    @classmethod
    def load(cls, path: Path) -> "MachineProfile":
        """Loads a profile file, refusing any schema other than the current one.

        Args:
            path : Profile file to read.

        Returns:
            The loaded profile.

        Raises:
            FileNotFoundError : If the file does not exist.
            ValueError        : If the file carries another schema version; rerun the installer to regenerate it.
        """
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
        """Whether a profile was loaded, as opposed to being absent or disabled."""
        return bool(self.data)

    @property
    def env_path(self) -> Path | None:
        """Location of the shell twin, beside the profile file."""
        return None if self.path is None else self.path.with_name(self.ENV_FILE_NAME)

    def section(self, name: str) -> dict:
        """Returns one section merged over its defaults.

        Args:
            name : Section name, one of the keys of ``DEFAULTS``.

        Returns:
            Mapping with every default key present.

        Raises:
            KeyError : If the section name is unknown.
        """
        return {**self.DEFAULTS[name], **self.data.get(name, {})}

    @property
    def machine(self) -> dict:
        """Interpreter, environment name and detected host facts."""
        return self.section("machine")

    @property
    def console(self) -> dict:
        """Bind address and port of the web console."""
        return self.section("console")

    @property
    def roots(self) -> dict:
        """Local roots that replace the reference-layout roots."""
        return self.section("roots")

    @property
    def dataset(self) -> dict:
        """Default dataset directory and parameter file on this machine."""
        return self.section("dataset")

    @property
    def hardware(self) -> dict:
        """GPU pool and worker counts the hardware-shaped defaults are rewritten to."""
        return self.section("hardware")

    @property
    def mail(self) -> dict:
        """Sender, recipient and SMTP endpoint of the report mailer."""
        return self.section("mail")

    @property
    def ntfy(self) -> dict:
        """Seed of the console's ntfy notification settings."""
        return self.section("ntfy")

    @property
    def config_overrides(self) -> dict:
        """Explicit ``Class.field`` overrides applied after the derived rules."""
        return dict(self.data.get("config_overrides", {}))

    @property
    def interpreter(self) -> str:
        """Path of the project interpreter recorded by the installer."""
        return self.machine["interpreter"]

    @property
    def env_name(self) -> str:
        """Name of the project conda environment."""
        return self.machine["env_name"]

    @property
    def gpu_pool(self) -> list:
        """CUDA device indices the entry points may use by default."""
        return [int(index) for index in self.hardware["gpu_pool"]]

    def path_anchors(self) -> list[tuple[str, str]]:
        """Returns the reference roots to rebase, paired with their local replacement.

        Only roots the profile sets are returned, and an absent profile returns none;
        the list is ordered longest anchor first so the most specific root wins when
        several match.
        """
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
        """Rewrites a path onto the first matching anchor's replacement.

        Args:
            text    : Path as text, relative to the checkout or absolute.
            anchors : Anchor and replacement pairs, most specific first.

        Returns:
            The rebased path, or None when no anchor matches.
        """
        for anchor, target in anchors:
            root = anchor.rstrip("/")

            if text == root or text.startswith(root + "/"):
                return target.rstrip("/") + text[len(root):]

        return None

    def field_rules(self) -> list[FieldRule]:
        """Returns the field-name rewrites derived from the hardware and mail sections.

        Rules with an empty or zero source value are omitted so an unset profile entry
        leaves the configuration default alone, and an absent profile yields none.
        """
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
        """Returns the facts the console shows about this profile."""
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
        """Returns the shell twin content: one exported variable per line."""
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
        """Writes the profile and its shell twin atomically.

        Returns:
            The profile file path.

        Raises:
            RuntimeError : If the profile has no location because profiles are disabled.
        """
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
        """Writes text to a temporary sibling and renames it into place at mode 0600."""
        temporary = path.with_name(path.name + ".tmp")

        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        temporary.chmod(0o600)
        os.replace(temporary, path)
