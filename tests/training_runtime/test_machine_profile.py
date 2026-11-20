
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
    dataset_path : Path = ReferenceLayout.DATASET_DIR
    log_base_dir : Path = ReferenceLayout.RUNS_ROOT / "benchmark"


@dataclass
class ExtendedPaths(PathsSection):
    extra : int = 1


@dataclass
class LoaderSection:
    num_workers : int  = 16
    pin_memory  : bool = True


@dataclass
class MailSection:
    sender    : str = "unrelated"
    recipient : str = "unrelated"


@dataclass
class ReportMailEntryConfig:
    sender    : str = "owner@example.org"
    recipient : str = "owner@example.org"
    smtp_host : str = "smtp.gmail.com"
    smtp_port : int = 465


@dataclass
class EntryConfig:
    paths          : ExtendedPaths         = field(default_factory=ExtendedPaths)
    loader         : LoaderSection         = field(default_factory=LoaderSection)
    other          : MailSection           = field(default_factory=MailSection)
    mail           : ReportMailEntryConfig = field(default_factory=ReportMailEntryConfig)
    gpus           : list                  = field(default_factory=lambda: [0, 1, 3])
    gpu            : int                   = 2
    gpu_flag       : bool                  = True
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
    "dataset"  : {"path": "/data/me/dataset"},
    "hardware" : {"gpu_pool": [1], "max_workers": 4},
    "mail"     : {"sender": "me@example.org", "recipient": "you@example.org"},
}


class TableLogger:

    def __init__(self) -> None:
        self.kv_tables = []
        self.rows      = []

    def kv_table(self, data, title=None, **kwargs) -> None:
        self.kv_tables.append((title, dict(data)))

    def metrics_table(self, rows, columns, title=None, **kwargs) -> None:
        self.rows.extend(dict(row) for row in rows)


def _profile(tmp_path: Path, data: dict) -> MachineProfile:
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(data))

    return MachineProfile.load(path)


def test_location_follows_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(MachineProfile.ENV_VAR, str(tmp_path / "p.json"))
    assert MachineProfile.location() == tmp_path / "p.json"

    monkeypatch.setenv(MachineProfile.ENV_VAR, "")
    assert MachineProfile.location() is None
    assert not MachineProfile.current().exists

    monkeypatch.delenv(MachineProfile.ENV_VAR)
    assert MachineProfile.location() == Path.home() / ".config" / "ml-template" / "machine.json"


def test_current_returns_an_absent_profile_when_the_default_location_holds_none(monkeypatch, tmp_path):
    monkeypatch.delenv(MachineProfile.ENV_VAR, raising=False)
    monkeypatch.setattr(MachineProfile, "default_path", classmethod(lambda cls: tmp_path / "none.json"))

    profile = MachineProfile.current()

    assert not profile.exists
    assert profile.path == tmp_path / "none.json"
    assert profile.path_anchors() == []
    assert profile.field_rules()  == []


def test_describe_and_state_follow_the_profile_location(tmp_path):
    loaded = _profile(tmp_path, FULL_DATA)
    absent = MachineProfile(tmp_path / "none.json")
    off    = MachineProfile(None)

    assert loaded.state == "loaded"   and loaded.describe() == f"{tmp_path / 'machine.json'} (loaded)"
    assert absent.state == "absent"   and absent.describe().startswith(f"absent at {tmp_path / 'none.json'}")
    assert off.state    == "disabled" and MachineProfile.ENV_VAR in off.describe()


def test_load_refuses_another_schema(tmp_path):
    with pytest.raises(FileNotFoundError):
        MachineProfile.load(tmp_path / "missing.json")


def test_sections_merge_over_defaults(tmp_path):
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
    profile = _profile(tmp_path, {"schema": 1, "roots": {"data": "/d", "runs": "/r"}, "dataset": {"path": "/d/s"}})
    anchors = [anchor for anchor, _ in profile.path_anchors()]

    assert anchors == [str(ReferenceLayout.DATASET_DIR), str(ReferenceLayout.DATA_ROOT), str(ReferenceLayout.RUNS_ROOT)]

    full = [anchor for anchor, _ in _profile(tmp_path, FULL_DATA).path_anchors()]
    assert full[0] == str(ReferenceLayout.DATASET_DIR)
    assert set(full) == {str(ReferenceLayout.DATASET_DIR), str(ReferenceLayout.DATA_ROOT), str(ReferenceLayout.RUNS_ROOT), str(ReferenceLayout.RESULTS_ROOT)}


def test_rebase_matches_whole_components_only():
    anchors = [("runs/sub", "/x"), ("runs", "/y")]

    assert MachineProfile.rebase("runs/sub",    anchors) == "/x"
    assert MachineProfile.rebase("runs/sub/c",  anchors) == "/x/c"
    assert MachineProfile.rebase("runs/subc",   anchors) == "/y/subc"
    assert MachineProfile.rebase("runs",        anchors) == "/y"
    assert MachineProfile.rebase("runs/",       anchors) == "/y/"
    assert MachineProfile.rebase("runs_archive", anchors) is None
    assert MachineProfile.rebase("elsewhere",   anchors) is None


def test_field_rules_follow_the_hardware_and_mail_sections(tmp_path):
    rules = {(rule.field, rule.kind, rule.owner): rule for rule in _profile(tmp_path, FULL_DATA).field_rules()}

    assert rules[("gpus", "set", None)].value        == [1]
    assert rules[("gpu", "set", None)].value         == 1
    assert rules[("num_workers", "cap", None)].value == 4
    assert rules[("sender", "set", "ReportMailEntryConfig")].value == "me@example.org"
    assert ("smtp_host", "set", "ReportMailEntryConfig") in rules

    empty = _profile(tmp_path, {"schema": 1, "hardware": {"gpu_pool": []}, "mail": {"smtp_host": "", "smtp_port": 0}})
    assert empty.field_rules() == []


def test_config_cli_applies_the_profile_to_the_defaults(tmp_path):
    profile = _profile(tmp_path, {**FULL_DATA, "config_overrides": {"EntryConfig.seed": 7, "PathsSection.extra": 5}})
    cli     = ConfigCli(EntryConfig(), profile=profile)
    config  = cli.config

    assert config.paths.dataset_path       == Path("/data/me/dataset")
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
    assert config.loader.num_workers       == 4
    assert config.loader.pin_memory        is True
    assert config.other.sender             == "unrelated"
    assert config.mail.sender              == "me@example.org"
    assert config.mail.recipient           == "you@example.org"
    assert config.seed                     == 7

    assert set(cli.profile_overrides) == {"paths.dataset_path", "paths.log_base_dir", "paths.extra", "logdir", "runs_dir", "output_dir", "gpus", "gpu", "loader.num_workers", "mail.sender", "mail.recipient", "seed"}


def test_a_dataset_path_alone_leaves_the_other_roots_relative(tmp_path):
    profile = _profile(tmp_path, {"schema": 1, "dataset": {"path": "/data/me/dataset"}})
    config  = ConfigCli(EntryConfig(), profile=profile).config

    assert config.paths.dataset_path == Path("/data/me/dataset")
    assert config.paths.log_base_dir == ReferenceLayout.RUNS_ROOT / "benchmark"
    assert config.logdir             == str(ReferenceLayout.RUNS_ROOT)


def test_sources_name_the_kind_of_rule_behind_each_rewritten_default(tmp_path):
    profile = _profile(tmp_path, FULL_DATA)
    cli     = ConfigCli(EntryConfig(), profile=profile)
    origins = {source.path: source.origin for source in cli.sources}
    changes = {source.path: (source.before, source.after) for source in cli.sources}

    assert origins["paths.dataset_path"] == "profile rebase"
    assert origins["logdir"]             == "profile rebase"
    assert origins["gpus"]               == "profile set"
    assert origins["loader.num_workers"] == "profile cap"
    assert changes["loader.num_workers"] == (16, 4)
    assert changes["paths.dataset_path"] == (ReferenceLayout.DATASET_DIR, Path("/data/me/dataset"))
    assert set(origins) == set(cli.profile_overrides)
    assert cli.unmatched_overrides == []


def test_config_overrides_that_name_no_leaf_are_listed(tmp_path):
    profile = _profile(tmp_path, {**FULL_DATA, "config_overrides": {"EntryConfig.seed": 7, "LoaderSection.num_wrokers": 2}})
    cli     = ConfigCli(EntryConfig(), profile=profile)

    assert cli.unmatched_overrides == ["LoaderSection.num_wrokers"]
    assert cli.sources_mapping()["unmatched_config_overrides"] == ["LoaderSection.num_wrokers"]


def test_report_logs_the_profile_state_and_one_row_per_rewritten_leaf(tmp_path):
    profile = _profile(tmp_path, FULL_DATA)
    cli     = ConfigCli(EntryConfig(), profile=profile)
    cli.apply(["--seed", "9"])
    logger  = TableLogger()

    cli.report(logger)

    title, summary = logger.kv_tables[0]
    by_leaf        = {row["Leaf"]: row for row in logger.rows}

    assert title == "Configuration sources"
    assert summary["Machine profile"]            == f"{tmp_path / 'machine.json'} (loaded)"
    assert summary["Defaults rewritten"]         == len(cli.profile_overrides)
    assert summary["Command-line overrides"]     == 1
    assert summary["Unmatched config_overrides"] == "none"
    assert by_leaf["loader.num_workers"] == {"Leaf": "loader.num_workers", "Origin": "profile cap", "Before": 16, "After": 4}
    assert by_leaf["seed"]               == {"Leaf": "seed", "Origin": "command line", "Before": 42, "After": 9}
    assert by_leaf["paths.dataset_path"]["After"] == "/data/me/dataset"


def test_command_line_flags_override_the_profile(tmp_path):
    profile = _profile(tmp_path, FULL_DATA)
    config  = ConfigCli(EntryConfig(), profile=profile).apply(["--paths.dataset_path", "/cli/dataset", "--gpus", "[0, 2]"])

    assert config.paths.dataset_path == Path("/cli/dataset")
    assert config.gpus               == [0, 2]


def test_a_disabled_profile_leaves_the_defaults_alone():
    cli = ConfigCli(EntryConfig(), profile=MachineProfile(None))

    assert cli.profile_overrides         == {}
    assert cli.config.paths.dataset_path == ReferenceLayout.DATASET_DIR
    assert cli.config.gpus               == [0, 1, 3]


def test_help_marks_the_defaults_the_profile_rewrote(tmp_path, capsys):
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
    assert reloaded.env_name           == "Env"


def test_field_rule_is_immutable():
    rule = FieldRule("gpus", "set", [0])

    with pytest.raises(AttributeError):
        rule.value = [1]
