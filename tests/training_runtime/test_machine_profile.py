"""Tests for the per-machine profile and its application through ConfigCli.

Covers the environment-driven location, schema refusal, section defaults, the
longest-anchor-first rebase of reference-layout paths, the hardware and mail field
rules, explicit Class.field overrides, the shell twin written beside the profile,
and the reporting of rewritten defaults in the config help.
"""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass, field
from pathlib     import Path

import pytest

from configuration.reference_layout import ReferenceLayout
from tools.runtime.config_cli       import ConfigCli
from tools.runtime.machine_profile  import FieldRule, MachineProfile


@dataclass
class PathsSection:
    """Base section with a dataset path below the reference data root."""
    dataset_path : Path = ReferenceLayout.DATASET_DIR
    log_base_dir : Path = ReferenceLayout.RUNS_ROOT / "benchmark"


@dataclass
class ExtendedPaths(PathsSection):
    """Section extending the base, so base-class rules must reach it."""
    extra : int = 1


@dataclass
class LoaderSection:
    """Section with worker counts and a flag the worker rules must skip."""
    num_workers       : int  = 16
    parameter_workers : int  = 2
    pin_memory        : bool = True


@dataclass
class MailSection:
    """Section reusing mail field names outside the mailer config, which mail rules must skip."""
    sender    : str = "unrelated"
    recipient : str = "unrelated"


@dataclass
class ReportMailEntryConfig:
    """Section named like the real mailer config, which mail rules must reach."""
    sender    : str = "owner@example.org"
    recipient : str = "owner@example.org"
    smtp_host : str = "smtp.gmail.com"
    smtp_port : int = 465


@dataclass
class EntryConfig:
    """Root config exercising rebasing, MRO ownership, caps, type filters and overrides."""
    paths          : ExtendedPaths         = field(default_factory=ExtendedPaths)
    loader         : LoaderSection         = field(default_factory=LoaderSection)
    other          : MailSection           = field(default_factory=MailSection)
    mail           : ReportMailEntryConfig = field(default_factory=ReportMailEntryConfig)
    gpus           : list                  = field(default_factory=lambda: [0, 1, 3])
    gpu            : int                   = 2
    gpu_flag       : bool                  = True
    vram_budget_gb : float                 = 40.0
    logdir         : str                   = str(ReferenceLayout.RUNS_ROOT)
    runs_dir       : str                   = str(ReferenceLayout.RUNS_ROOT / "backbone")
    output_dir     : Path                  = ReferenceLayout.RESULTS_ROOT / "paper_figures"
    sibling        : str                   = "runs_archive/old"
    plain          : str                   = "docs/resolved_entry_config.json"
    seed           : int                   = 42


FULL_DATA = {
    "schema"   : MachineProfile.SCHEMA,
    "machine"  : {"env_name": "Env", "interpreter": "/envs/Env/bin/python", "conda_root": "/opt/conda"},
    "console"  : {"host": "0.0.0.0", "port": 9001},
    "roots"    : {"repo": "/home/me/ML-Template", "data": "/data/me", "runs": "/scratch/me/runs", "results": "/home/me/results"},
    "dataset"  : {"path": "/data/me/scene", "parameters_path": "/data/me/scene/params/p/parameters.npy"},
    "hardware" : {"gpu_pool": [1], "vram_budget_gb": 7.5, "cpu_workers": 6, "gif_workers": 3, "max_workers": 4},
    "mail"     : {"sender": "me@example.org", "recipient": "you@example.org"},
}


def _profile(tmp_path: Path, data: dict) -> MachineProfile:
    """Writes a profile file with the given content and loads it back.

    Args:
        tmp_path : Directory the profile is written into.
        data     : Profile mapping, written verbatim.

    Returns:
        The loaded profile.
    """
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(data))

    return MachineProfile.load(path)


def test_location_follows_the_environment_variable(monkeypatch, tmp_path):
    """The variable names the file, an empty value disables profiles, unset falls back to the home config."""
    monkeypatch.setenv(MachineProfile.ENV_VAR, str(tmp_path / "p.json"))
    assert MachineProfile.location() == tmp_path / "p.json"

    monkeypatch.setenv(MachineProfile.ENV_VAR, "")
    assert MachineProfile.location() is None
    assert not MachineProfile.current().exists

    monkeypatch.delenv(MachineProfile.ENV_VAR)
    assert MachineProfile.location() == Path.home() / ".config" / "ml-template" / "machine.json"


def test_current_returns_an_absent_profile_when_the_file_is_missing(monkeypatch, tmp_path):
    """A missing file yields a profile that exists is False and applies nothing."""
    monkeypatch.setenv(MachineProfile.ENV_VAR, str(tmp_path / "none.json"))

    profile = MachineProfile.current()

    assert not profile.exists
    assert profile.path == tmp_path / "none.json"
    assert profile.path_anchors() == []
    assert profile.field_rules()  == []


def test_load_refuses_another_schema(tmp_path):
    """A profile written by another schema version is rejected instead of half-applied."""
    with pytest.raises(ValueError, match="schema 0"):
        _profile(tmp_path, {"schema": 0})

    with pytest.raises(FileNotFoundError):
        MachineProfile.load(tmp_path / "missing.json")


def test_sections_merge_over_defaults(tmp_path):
    """Every section carries its default keys and the loaded values win."""
    profile = _profile(tmp_path, FULL_DATA)

    assert profile.console           == {"host": "0.0.0.0", "port": 9001}
    assert profile.mail["smtp_host"] == "smtp.gmail.com"
    assert profile.ntfy              == MachineProfile.DEFAULTS["ntfy"]
    assert profile.env_name          == "Env"
    assert profile.gpu_pool          == [1]

    absent = MachineProfile(None)
    assert absent.console == {"host": ReferenceLayout.CONSOLE_HOST, "port": ReferenceLayout.CONSOLE_PORT}

    with pytest.raises(KeyError):
        profile.section("nope")


def test_path_anchors_are_ordered_most_specific_first(tmp_path):
    """The parameter file anchor precedes the dataset, which precedes the data root, and unset roots are omitted."""
    profile = _profile(tmp_path, {"schema": 1, "roots": {"data": "/d", "runs": "/r"}, "dataset": {"path": "/d/s"}})
    anchors = [anchor for anchor, _ in profile.path_anchors()]

    assert anchors == [str(ReferenceLayout.DATASET_DIR), str(ReferenceLayout.DATA_ROOT), str(ReferenceLayout.RUNS_ROOT)]

    full = [anchor for anchor, _ in _profile(tmp_path, FULL_DATA).path_anchors()]
    assert full[0] == str(ReferenceLayout.PARAMETERS)
    assert set(full) == {str(ReferenceLayout.PARAMETERS), str(ReferenceLayout.DATASET_DIR), str(ReferenceLayout.DATA_ROOT), str(ReferenceLayout.RUNS_ROOT), str(ReferenceLayout.RESULTS_ROOT)}


def test_rebase_matches_whole_components_only():
    """A root matches itself or a child path, never a sibling sharing the prefix, and keeps a trailing slash."""
    anchors = [("runs/sub", "/x"), ("runs", "/y")]

    assert MachineProfile.rebase("runs/sub",    anchors) == "/x"
    assert MachineProfile.rebase("runs/sub/c",  anchors) == "/x/c"
    assert MachineProfile.rebase("runs/subc",   anchors) == "/y/subc"
    assert MachineProfile.rebase("runs",        anchors) == "/y"
    assert MachineProfile.rebase("runs/",       anchors) == "/y/"
    assert MachineProfile.rebase("runs_archive", anchors) is None
    assert MachineProfile.rebase("elsewhere",   anchors) is None


def test_field_rules_follow_the_hardware_and_mail_sections(tmp_path):
    """Set rules for the pool, VRAM and worker counts, caps for loader workers, and owner-bound mail rules."""
    rules = {(rule.field, rule.kind, rule.owner): rule for rule in _profile(tmp_path, FULL_DATA).field_rules()}

    assert rules[("gpus", "set", None)].value              == [1]
    assert rules[("gpu", "set", None)].value               == 1
    assert rules[("vram_budget_gb", "set", None)].value    == 7.5
    assert rules[("cpu_workers", "set", None)].value       == 6
    assert rules[("num_workers", "cap", None)].value       == 4
    assert rules[("parameter_workers", "cap", None)].value == 4
    assert rules[("sender", "set", "ReportMailEntryConfig")].value == "me@example.org"
    assert ("smtp_host", "set", "ReportMailEntryConfig") in rules

    empty = _profile(tmp_path, {"schema": 1, "hardware": {"gpu_pool": []}, "mail": {"smtp_host": "", "smtp_port": 0}})
    assert empty.field_rules() == []


def test_config_overrides_must_name_a_class_and_field(tmp_path):
    """A bare field name in config_overrides is refused."""
    profile = _profile(tmp_path, {"schema": 1, "config_overrides": {"seed": 1}})

    with pytest.raises(ValueError, match="Class.field"):
        profile.field_rules()


def test_config_cli_applies_the_profile_to_the_defaults(tmp_path):
    """Paths below the reference roots are rebased, rules apply by name, owner and type, and the changes are reported."""
    profile = _profile(tmp_path, {**FULL_DATA, "config_overrides": {"EntryConfig.seed": 7, "PathsSection.extra": 5}})
    cli     = ConfigCli(EntryConfig(), profile=profile)
    config  = cli.config

    assert config.paths.dataset_path       == Path("/data/me/scene")
    assert config.paths.log_base_dir       == Path("/scratch/me/runs/benchmark")
    assert config.paths.extra              == 5
    assert config.logdir                   == "/scratch/me/runs"
    assert config.runs_dir                 == "/scratch/me/runs/backbone"
    assert config.output_dir               == Path("/home/me/results/paper_figures")
    assert config.sibling                  == "runs_archive/old"
    assert config.plain                    == "docs/resolved_entry_config.json"
    assert config.gpus                     == [1]
    assert config.gpu                      == 1
    assert config.gpu_flag                 is True
    assert config.vram_budget_gb           == 7.5
    assert config.loader.num_workers       == 4
    assert config.loader.parameter_workers == 2
    assert config.loader.pin_memory        is True
    assert config.other.sender             == "unrelated"
    assert config.mail.sender              == "me@example.org"
    assert config.mail.recipient           == "you@example.org"
    assert config.seed                     == 7

    assert set(cli.profile_overrides) == {"paths.dataset_path", "paths.log_base_dir", "paths.extra", "logdir", "runs_dir", "output_dir", "gpus", "gpu", "vram_budget_gb", "loader.num_workers", "mail.sender", "mail.recipient", "seed"}


def test_a_dataset_path_alone_leaves_the_other_roots_relative(tmp_path):
    """Only the roots the profile sets move; a dataset path without a data root leaves the runs tree in the checkout."""
    profile = _profile(tmp_path, {"schema": 1, "dataset": {"path": "/data/me/scene"}})
    config  = ConfigCli(EntryConfig(), profile=profile).config

    assert config.paths.dataset_path == Path("/data/me/scene")
    assert config.paths.log_base_dir == ReferenceLayout.RUNS_ROOT / "benchmark"
    assert config.logdir             == str(ReferenceLayout.RUNS_ROOT)


def test_command_line_flags_override_the_profile(tmp_path):
    """A flag wins over the profile value for the same leaf."""
    profile = _profile(tmp_path, FULL_DATA)
    config  = ConfigCli(EntryConfig(), profile=profile).apply(["--paths.dataset_path", "/cli/scene", "--gpus", "[0, 2]"])

    assert config.paths.dataset_path == Path("/cli/scene")
    assert config.gpus               == [0, 2]


def test_a_disabled_profile_leaves_the_defaults_alone():
    """The absent profile keeps every default and reports no rewrite."""
    cli = ConfigCli(EntryConfig(), profile=MachineProfile(None))

    assert cli.profile_overrides         == {}
    assert cli.config.paths.dataset_path == ReferenceLayout.DATASET_DIR
    assert cli.config.gpus               == [0, 1, 3]


def test_help_marks_the_defaults_the_profile_rewrote(tmp_path, capsys):
    """The config help flags rewritten rows and names the profile file."""
    profile = _profile(tmp_path, FULL_DATA)

    with pytest.raises(SystemExit):
        ConfigCli(EntryConfig(), profile=profile).apply(["--help-config"])

    out = capsys.readouterr().out
    assert "--paths.dataset_path" in out
    assert "[machine profile]"    in out
    assert str(tmp_path / "machine.json") in out

    with pytest.raises(SystemExit):
        ConfigCli(EntryConfig(), profile=MachineProfile(None)).apply(["--help-config"])

    assert "[machine profile]" not in capsys.readouterr().out


def test_save_writes_the_profile_and_its_shell_twin(tmp_path):
    """Saving writes owner-only JSON with the schema and stamp, plus the exported shell variables beside it."""
    path    = tmp_path / "cfg" / "machine.json"
    profile = MachineProfile(path, {"machine": {"env_name": "Env", "interpreter": "/envs/Env/bin/python", "conda_root": "/opt/conda"}, "console": {"port": 9001}, "roots": {"repo": "/home/me/repo"}})

    assert profile.save() == path

    written = json.loads(path.read_text())
    assert written["schema"]                 == MachineProfile.SCHEMA
    assert written["written_at"]
    assert written["machine"]["hostname"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    twin = profile.env_path.read_text()
    assert f"export MLTEMPLATE_MACHINE_PROFILE={path}" in twin
    assert "export MLTEMPLATE_INTERPRETER=/envs/Env/bin/python" in twin
    assert "export MLTEMPLATE_ENV_NAME=Env"           in twin
    assert "export MLTEMPLATE_CONSOLE_PORT=9001"      in twin
    assert "export MLTEMPLATE_REPO_ROOT=/home/me/repo" in twin

    reloaded = MachineProfile.load(path)
    assert reloaded.console["port"] == 9001
    assert reloaded.summary()["env_name"] == "Env"


def test_save_refuses_when_profiles_are_disabled():
    """Without a location there is nowhere to write."""
    with pytest.raises(RuntimeError, match=MachineProfile.ENV_VAR):
        MachineProfile(None, {"machine": {}}).save()


def test_field_rule_is_immutable():
    """Rules are frozen so a shared rule list cannot be edited by a consumer."""
    rule = FieldRule("gpus", "set", [0])

    with pytest.raises(AttributeError):
        rule.value = [1]
