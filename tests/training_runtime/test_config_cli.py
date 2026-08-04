from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib     import Path

import pytest

from tools.runtime.config_cli import ConfigCli


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
    nested : NestedConfig = field(default_factory=NestedConfig)
    paths  : PathsConfig  = field(default_factory=PathsConfig)
    seed   : int          = 42
    betas  : tuple        = (0.9, 0.999)
    layers : list         = field(default_factory=lambda: [16, 32])
    extra  : dict         = field(default_factory=lambda: {"a": 1})
    opt    : str          = None


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
        ConfigCli(cfg).apply([f"--nested.enabled", token])
        assert cfg.nested.enabled is True


def test_apply_bool_false_variants():
    for token in ("false", "0", "no", "off"):
        cfg = _config()
        ConfigCli(cfg).apply(["--nested.enabled", token])
        assert cfg.nested.enabled is False


def test_apply_bool_invalid_raises():
    cfg = _config()
    with pytest.raises(ValueError):
        ConfigCli(cfg).apply(["--nested.enabled", "maybe"])


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


def test_unknown_option_rejected():
    cfg = _config()
    with pytest.raises(ValueError):
        ConfigCli(cfg).apply(["--does-not-exist", "5"])


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


def test_load_resolved_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ConfigCli.load_resolved(_config(), Path("/nonexistent/path.json"))


def test_load_resolved_unknown_key_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nested.lr": 0.1, "ghost.key": 9}))

    with pytest.raises(KeyError):
        ConfigCli.load_resolved(_config(), bad)


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
