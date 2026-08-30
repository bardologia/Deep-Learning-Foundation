"""Tests for the dataclass-backed command line override layer.

Covers leaf flattening, type coercion of CLI tokens, resolved-config
persistence and the argv round trip provided by ConfigCli.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib     import Path

import pytest

from tools.runtime.config_cli import ConfigCli


@dataclass
class NestedConfig:
    """Nested dataclass exercising float, int, bool and str coercion."""
    lr      : float = 0.01
    epochs  : int   = 10
    enabled : bool  = True
    name    : str   = "base"


@dataclass
class PathsConfig:
    """Nested dataclass holding a single Path-typed field."""
    log_base_dir: Path = Path("logs")


@dataclass
class RootConfig:
    """Root config combining nested sections with scalar, sequence and optional fields."""
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
    """Returns a fresh RootConfig instance with default values."""
    return RootConfig()


def test_leaves_flattens_nested_paths():
    """Verifies that nested dataclass fields are flattened to dotted leaf paths."""
    leaves = dict(ConfigCli._leaves(_config()))

    assert leaves["nested.lr"]          == 0.01
    assert leaves["nested.epochs"]      == 10
    assert leaves["paths.log_base_dir"] == Path("logs")
    assert leaves["seed"]               == 42


def test_apply_overrides_nested_float():
    """Verifies that a nested float override reaches the config and the overrides map."""
    cfg = _config()
    cli = ConfigCli(cfg)
    cli.apply(["--nested.lr", "0.5"])

    assert cfg.nested.lr             == 0.5
    assert cli.overrides["nested.lr"] == 0.5


def test_apply_override_int():
    """Verifies that an int-typed field is coerced to int rather than str."""
    cfg = _config()
    ConfigCli(cfg).apply(["--seed", "7"])
    assert cfg.seed == 7
    assert isinstance(cfg.seed, int)


def test_apply_dashed_alias_maps_to_underscore_path():
    """Verifies that a dashed flag alias resolves to the underscored leaf path."""
    cfg = _config()
    ConfigCli(cfg).apply(["--paths.log-base-dir", "/tmp/run"])
    assert cfg.paths.log_base_dir == Path("/tmp/run")


def test_apply_bool_true_variants():
    """Verifies that true, 1, yes and on all parse to True."""
    for token in ("true", "1", "yes", "on"):
        cfg = _config()
        cfg.nested.enabled = False
        ConfigCli(cfg).apply([f"--nested.enabled", token])
        assert cfg.nested.enabled is True


def test_apply_bool_false_variants():
    """Verifies that false, 0, no and off all parse to False."""
    for token in ("false", "0", "no", "off"):
        cfg = _config()
        ConfigCli(cfg).apply(["--nested.enabled", token])
        assert cfg.nested.enabled is False


def test_apply_bool_invalid_raises():
    """Verifies that a non-boolean token for a bool field raises ValueError."""
    cfg = _config()
    with pytest.raises(ValueError):
        ConfigCli(cfg).apply(["--nested.enabled", "maybe"])


def test_apply_path_coercion():
    """Verifies that a string token for a Path field is coerced to Path."""
    cfg = _config()
    ConfigCli(cfg).apply(["--paths.log_base_dir", "/var/x"])
    assert cfg.paths.log_base_dir == Path("/var/x")
    assert isinstance(cfg.paths.log_base_dir, Path)


def test_apply_list_literal():
    """Verifies that a Python list literal is parsed into a list."""
    cfg = _config()
    ConfigCli(cfg).apply(["--layers", "[1, 2, 3]"])
    assert cfg.layers == [1, 2, 3]


def test_apply_list_comma_fallback():
    """Verifies that a comma-separated token falls back to a list of strings."""
    cfg = _config()
    ConfigCli(cfg).apply(["--layers", "a,b,c"])
    assert cfg.layers == ["a", "b", "c"]


def test_apply_tuple_coercion():
    """Verifies that a list literal assigned to a tuple field is coerced to tuple."""
    cfg = _config()
    ConfigCli(cfg).apply(["--betas", "[0.8, 0.99]"])
    assert cfg.betas == (0.8, 0.99)
    assert isinstance(cfg.betas, tuple)


def test_apply_optional_list_scalar_is_wrapped():
    """Verifies that a scalar token for an optional list field is wrapped in a list."""
    cfg = _config()
    ConfigCli(cfg).apply(["--kernels", "5"])
    assert cfg.kernels == [5]


def test_apply_optional_list_comma_matches_non_optional_list():
    """Verifies that optional and non-optional list fields parse comma tokens identically."""
    cfg = _config()
    ConfigCli(cfg).apply(["--kernels", "3,3,5", "--layers", "3,3,5"])
    assert cfg.kernels == [3, 3, 5]
    assert cfg.kernels == cfg.layers


def test_apply_optional_tuple_coercion():
    """Verifies that a comma token for an optional tuple field is coerced to tuple."""
    cfg = _config()
    ConfigCli(cfg).apply(["--window", "64,32"])
    assert cfg.window == (64, 32)
    assert isinstance(cfg.window, tuple)


def test_apply_dict_coercion():
    """Verifies that a dict literal replaces the default dict value."""
    cfg = _config()
    ConfigCli(cfg).apply(["--extra", "{'b': 2}"])
    assert cfg.extra == {"b": 2}


def test_unset_options_leave_defaults():
    """Verifies that fields without an override keep their dataclass defaults."""
    cfg = _config()
    ConfigCli(cfg).apply([])
    assert cfg.nested.lr     == 0.01
    assert cfg.seed          == 42
    assert cfg.layers        == [16, 32]


def test_unknown_option_rejected():
    """Verifies that an unknown option raises ValueError."""
    cfg = _config()
    with pytest.raises(ValueError):
        ConfigCli(cfg).apply(["--does-not-exist", "5"])


def test_bootstrap_flags_not_rejected():
    """Verifies that bootstrap flags such as --gpu are ignored instead of rejected."""
    cfg = _config()
    ConfigCli(cfg).apply(["--gpu", "0", "--seed", "3"])
    assert cfg.seed == 3


def test_multiple_overrides_applied_together():
    """Verifies that several overrides in one argv are all applied and recorded."""
    cfg = _config()
    cli = ConfigCli(cfg)
    cli.apply(["--nested.lr", "0.2", "--nested.epochs", "50", "--seed", "99"])

    assert cfg.nested.lr     == 0.2
    assert cfg.nested.epochs == 50
    assert cfg.seed          == 99
    assert set(cli.overrides) == {"nested.lr", "nested.epochs", "seed"}


def test_set_path_static():
    """Verifies that set_path assigns a value at a dotted leaf path."""
    cfg = _config()
    ConfigCli.set_path(cfg, "nested.name", "changed")
    assert cfg.nested.name == "changed"


def test_apply_overrides_classmethod():
    """Verifies that apply_overrides assigns every entry of an override mapping."""
    cfg = _config()
    ConfigCli.apply_overrides(cfg, {"nested.lr": 0.9, "seed": 5})
    assert cfg.nested.lr == 0.9
    assert cfg.seed      == 5


def test_to_mapping_serializes_path_and_tuple():
    """Verifies that to_mapping renders Path as str and tuple as list."""
    cfg     = _config()
    mapping = ConfigCli.to_mapping(cfg)

    assert mapping["paths.log_base_dir"] == "logs"
    assert mapping["betas"]              == [0.9, 0.999]
    assert mapping["nested.lr"]          == 0.01


def test_save_and_load_resolved_round_trip(tmp_path):
    """Verifies that a saved resolved config restores scalar values into a fresh config."""
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
    """Verifies that Path and tuple field types survive a save/load round trip."""
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
    """Verifies that loading a nonexistent resolved config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ConfigCli.load_resolved(_config(), Path("/nonexistent/path.json"))


def test_load_resolved_unknown_key_raises(tmp_path):
    """Verifies that a resolved config with an unknown leaf key raises KeyError."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nested.lr": 0.1, "ghost.key": 9}))

    with pytest.raises(KeyError):
        ConfigCli.load_resolved(_config(), bad)


def test_to_argv_round_trips_overrides():
    """Verifies that overrides rendered back to argv reproduce the same config state."""
    cfg = _config()
    cli = ConfigCli(cfg)
    cli.apply(["--nested.lr", "0.25", "--nested.enabled", "false", "--betas", "[0.1, 0.2]"])

    argv  = ConfigCli.to_argv(cli.overrides)
    fresh = _config()
    ConfigCli(fresh).apply(argv)

    assert fresh.nested.lr      == 0.25
    assert fresh.nested.enabled is False
    assert fresh.betas          == (0.1, 0.2)
