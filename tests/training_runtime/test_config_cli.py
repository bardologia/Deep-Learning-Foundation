
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib     import Path

import pytest

from configuration.training        import BackboneEntryConfig
from tools.runtime.config_cli      import ConfigCli, ConfigSource
from tools.runtime.machine_profile import MachineProfile


@dataclass
class NestedConfig:
    lr      : float = 0.01
    epochs  : int   = 10
    enabled : bool  = True
    name    : str   = "base"


@dataclass
class PathsConfig:
    log_base_dir: Path = Path("logs")


@dataclass
class RootConfig:
    nested  : NestedConfig            = field(default_factory=NestedConfig)
    paths   : PathsConfig             = field(default_factory=PathsConfig)
    seed    : int                     = 42
    betas   : tuple                   = (0.9, 0.999)
    layers  : list                    = field(default_factory=lambda: [16, 32])
    extra   : dict                    = field(default_factory=lambda: {"a": 1})
    opt     : str                     = None
    kernels : list[int] | None        = None
    window  : tuple[int, ...] | None  = None


def _config():
    return RootConfig()


def test_leaves_flattens_nested_paths():
    leaves = dict(ConfigCli._leaves(_config()))

    assert leaves["nested.lr"]          == 0.01
    assert leaves["nested.epochs"]      == 10
    assert leaves["paths.log_base_dir"] == Path("logs")
    assert leaves["seed"]               == 42


def test_apply_overrides_nested_float():
    cfg = _config()
    cli = ConfigCli(cfg)
    cli.apply(["--nested.lr", "0.5"])

    assert cfg.nested.lr             == 0.5
    assert cli.overrides["nested.lr"] == 0.5


def test_typed_overrides_are_recorded_with_the_value_they_replaced():
    cfg = _config()
    cli = ConfigCli(cfg, profile=MachineProfile(None))
    cli.apply(["--seed", "7", "--paths.log_base_dir", "/tmp/run"])

    assert cli.sources == [ConfigSource("paths.log_base_dir", "command line", Path("logs"), Path("/tmp/run")), ConfigSource("seed", "command line", 42, 7)]

    mapping = cli.sources_mapping()

    assert mapping["machine_profile"] == {"path": None, "state": "disabled"}
    assert mapping["leaves"]          == [{"path": "paths.log_base_dir", "origin": "command line", "before": "logs", "after": "/tmp/run"}, {"path": "seed", "origin": "command line", "before": 42, "after": 7}]
    assert mapping["unmatched_config_overrides"] == []
    assert json.dumps(mapping)


def test_only_the_parse_of_the_process_arguments_is_recorded(monkeypatch):
    monkeypatch.setattr(ConfigCli, "parsed", None)

    ConfigCli(_config(), profile=MachineProfile(None)).apply(["--seed", "7"])

    assert ConfigCli.parsed is None

    monkeypatch.setattr(sys, "argv", ["entry.py", "--seed", "3"])
    cli = ConfigCli(_config(), profile=MachineProfile(None))
    cli.apply()

    assert ConfigCli.parsed is cli
    assert cli.config.seed == 3


def test_apply_override_int():
    cfg = _config()
    ConfigCli(cfg).apply(["--seed", "7"])
    assert cfg.seed == 7
    assert isinstance(cfg.seed, int)


def test_apply_dashed_alias_maps_to_underscore_path():
    cfg = _config()
    ConfigCli(cfg).apply(["--paths.log-base-dir", "/tmp/run"])
    assert cfg.paths.log_base_dir == Path("/tmp/run")


def test_apply_bool_true_variants():
    for token in ("true", "1", "yes", "on"):
        cfg = _config()
        cfg.nested.enabled = False
        ConfigCli(cfg).apply(["--nested.enabled", token])
        assert cfg.nested.enabled is True


def test_apply_bool_false_variants():
    for token in ("false", "0", "no", "off"):
        cfg = _config()
        ConfigCli(cfg).apply(["--nested.enabled", token])
        assert cfg.nested.enabled is False


def test_apply_path_coercion():
    cfg = _config()
    ConfigCli(cfg).apply(["--paths.log_base_dir", "/var/x"])
    assert cfg.paths.log_base_dir == Path("/var/x")
    assert isinstance(cfg.paths.log_base_dir, Path)


def test_apply_list_literal():
    cfg = _config()
    ConfigCli(cfg).apply(["--layers", "[1, 2, 3]"])
    assert cfg.layers == [1, 2, 3]


def test_apply_list_comma_fallback():
    cfg = _config()
    ConfigCli(cfg).apply(["--layers", "a,b,c"])
    assert cfg.layers == ["a", "b", "c"]


def test_apply_tuple_coercion():
    cfg = _config()
    ConfigCli(cfg).apply(["--betas", "[0.8, 0.99]"])
    assert cfg.betas == (0.8, 0.99)
    assert isinstance(cfg.betas, tuple)


def test_apply_optional_list_scalar_is_wrapped():
    cfg = _config()
    ConfigCli(cfg).apply(["--kernels", "5"])
    assert cfg.kernels == [5]


def test_apply_optional_list_comma_matches_non_optional_list():
    cfg = _config()
    ConfigCli(cfg).apply(["--kernels", "3,3,5", "--layers", "3,3,5"])
    assert cfg.kernels == [3, 3, 5]
    assert cfg.kernels == cfg.layers


def test_apply_optional_tuple_coercion():
    cfg = _config()
    ConfigCli(cfg).apply(["--window", "64,32"])
    assert cfg.window == (64, 32)
    assert isinstance(cfg.window, tuple)


def test_apply_dict_coercion():
    cfg = _config()
    ConfigCli(cfg).apply(["--extra", "{'b': 2}"])
    assert cfg.extra == {"b": 2}


def test_unset_options_leave_defaults():
    cfg = _config()
    ConfigCli(cfg).apply([])
    assert cfg.nested.lr     == 0.01
    assert cfg.seed          == 42
    assert cfg.layers        == [16, 32]


def test_unknown_option_exits_with_the_argparse_error(capsys):
    cfg = _config()

    with pytest.raises(SystemExit) as stop:
        ConfigCli(cfg).apply(["--does-not-exist", "5"])

    assert stop.value.code == 2
    assert "unrecognized arguments: --does-not-exist" in capsys.readouterr().err


def test_stray_values_after_an_override_exit_with_the_argparse_error(capsys):
    cfg = _config()

    with pytest.raises(SystemExit) as stop:
        ConfigCli(cfg).apply(["--layers", "0", "1", "2"])

    assert stop.value.code == 2
    assert "unrecognized arguments: 1 2" in capsys.readouterr().err
    assert cfg.layers == [16, 32]


def test_stray_value_after_a_switch_is_not_taken_as_its_value(capsys):
    cfg = _config()

    with pytest.raises(SystemExit):
        ConfigCli(cfg).apply(["--trial", "--layers", "0", "1"])

    assert "unrecognized arguments: 1" in capsys.readouterr().err
    assert cfg.layers == [16, 32]


def test_stray_value_is_matched_to_the_flag_it_follows():
    cfg = _config()

    with pytest.raises(SystemExit):
        ConfigCli(cfg).apply(["--gpu", "--layers", "0", "1"])

    ConfigCli(cfg).apply(["--worker", "train", "--trial", "--seed", "4"])
    assert cfg.seed == 4


def test_bootstrap_flag_values_are_not_stray():
    cfg = _config()
    ConfigCli(cfg).apply(["--run-tag", "sweep_a", "--seed", "3"])
    assert cfg.seed == 3


def test_bootstrap_flags_not_rejected():
    cfg = _config()
    ConfigCli(cfg).apply(["--gpu", "0", "--seed", "3"])
    assert cfg.seed == 3


def test_multiple_overrides_applied_together():
    cfg = _config()
    cli = ConfigCli(cfg)
    cli.apply(["--nested.lr", "0.2", "--nested.epochs", "50", "--seed", "99"])

    assert cfg.nested.lr     == 0.2
    assert cfg.nested.epochs == 50
    assert cfg.seed          == 99
    assert set(cli.overrides) == {"nested.lr", "nested.epochs", "seed"}


def test_set_path_static():
    cfg = _config()
    ConfigCli.set_path(cfg, "nested.name", "changed")
    assert cfg.nested.name == "changed"


def test_apply_overrides_classmethod():
    cfg = _config()
    ConfigCli.apply_overrides(cfg, {"nested.lr": 0.9, "seed": 5})
    assert cfg.nested.lr == 0.9
    assert cfg.seed      == 5


def test_to_mapping_serializes_path_and_tuple():
    cfg     = _config()
    mapping = ConfigCli.to_mapping(cfg)

    assert mapping["paths.log_base_dir"] == "logs"
    assert mapping["betas"]              == [0.9, 0.999]
    assert mapping["nested.lr"]          == 0.01


def test_save_and_load_resolved_round_trip(tmp_path):
    cfg = _config()
    cfg.nested.lr = 0.333
    cfg.seed      = 17

    out = ConfigCli.save_resolved(cfg, tmp_path / "resolved.json")
    assert out.is_file()

    fresh = _config()
    ConfigCli.load_resolved(fresh, out)

    assert fresh.nested.lr == 0.333
    assert fresh.seed      == 17


def test_load_resolved_restores_path_and_tuple_types(tmp_path):
    cfg = _config()
    cfg.paths.log_base_dir = Path("/abc")
    cfg.betas              = (0.5, 0.6)

    out = ConfigCli.save_resolved(cfg, tmp_path / "r.json")

    fresh = _config()
    ConfigCli.load_resolved(fresh, out)

    assert isinstance(fresh.paths.log_base_dir, Path)
    assert fresh.paths.log_base_dir == Path("/abc")
    assert isinstance(fresh.betas, tuple)
    assert fresh.betas == (0.5, 0.6)


def test_load_resolved_refuses_an_unknown_key(tmp_path):
    out     = ConfigCli.save_resolved(_config(), tmp_path / "r.json")
    mapping = json.loads(out.read_text())
    mapping["removed_field"] = 1
    out.write_text(json.dumps(mapping))

    with pytest.raises(KeyError, match="removed_field"):
        ConfigCli.load_resolved(_config(), out)


def test_load_resolved_refuses_a_missing_key(tmp_path):
    out     = ConfigCli.save_resolved(_config(), tmp_path / "r.json")
    mapping = json.loads(out.read_text())
    del mapping["seed"]
    out.write_text(json.dumps(mapping))

    with pytest.raises(KeyError, match="seed"):
        ConfigCli.load_resolved(_config(), out)


def test_load_resolved_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ConfigCli.load_resolved(_config(), Path("/nonexistent/path.json"))


def test_to_argv_round_trips_overrides():
    cfg = _config()
    cli = ConfigCli(cfg)
    cli.apply(["--nested.lr", "0.25", "--nested.enabled", "false", "--betas", "[0.1, 0.2]"])

    argv  = ConfigCli.to_argv(cli.overrides)
    fresh = _config()
    ConfigCli(fresh).apply(argv)

    assert fresh.nested.lr      == 0.25
    assert fresh.nested.enabled is False
    assert fresh.betas          == (0.1, 0.2)


def test_an_optional_leaf_set_to_none_reads_as_none():
    config = BackboneEntryConfig()
    config.training.scheduler_epochs = 7
    config.run_name                  = "named"

    ConfigCli(config).apply(["--training.scheduler_epochs", "None", "--run_name", "None"])

    assert config.training.scheduler_epochs is None
    assert config.run_name                  is None
