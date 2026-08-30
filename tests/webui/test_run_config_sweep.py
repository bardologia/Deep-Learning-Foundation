"""Exhaustive sweep of the copy-from-run path over every entry point config.

For each registered entry point this saves the real entry config the way a run
does, reads it back through RunConfigReader, and checks the two properties the
launch control rests on, field by field:

- Parity: every stored value renders to exactly the string the script config
  resolver produces for the same default, so copying an unmodified run yields
  zero overrides — no phantom overrides, no missed fields.
- Round trip: every editable field, once changed, survives the full cycle of
  run file → reader rendering → command-line override → fresh config with the
  original value intact, so a copied run relaunches identically.
- Withholding: the fields naming the source run itself (run_name, resume,
  resume_from) never reach the form, on any entry point that has them.

One case also runs the real ScriptConfigResolver subprocess and compares its
leaves against the reader, so the shared renderer is pinned against the live
bootstrap rather than against a copy of its rules.

The layout side (every leaf claimed by a launch page exactly once) is covered
by test_launch_layout.py; together the two files pin the whole copy path.
"""

from __future__ import annotations

import sys
import typing

from enum import Enum
from importlib import import_module
from pathlib import Path

import pytest

from config_value_render    import ConfigValueRenderer
from project_paths          import ProjectPaths
from run_config_reader      import RunConfigReader
from script_config_resolver import ScriptConfigResolver
from web_logger             import WebLogger

from tools.runtime.config_cli import ConfigCli

_DISPATCH_ONLY = {"generate_tomogram", "generate_interferograms"}

_ENTRY_KEYS = sorted(key for key in ProjectPaths.SCRIPT_DIRS if key not in _DISPATCH_ONLY)


def _entry_config_class(key: str):
    """Returns the entry config class of one entry point, resolved like the console does."""
    entry = ScriptConfigResolver(ProjectPaths()).entry_config(key)
    return getattr(import_module(entry["module"]), entry["class"])


def _is_identity(path: str) -> bool:
    """Returns whether a dotted path is one the reader withholds as the run's identity."""
    return path.rsplit(".", 1)[-1] in RunConfigReader.IDENTITY_LEAVES


_NONE_SAMPLES = {
    str   : "copy_probe",
    int   : 3,
    float : 0.25,
    bool  : True,
    Path  : Path("/tmp/copy_probe"),
    list  : [1, 2],
    tuple : (1, 2),
    dict  : {"copy_probe": 1},
}


def _fill_optional(config, path: str):
    """Returns a sample value for a None-valued leaf from its type annotation, or None.

    A None default carries no type, so the sample comes from the field's optional
    annotation — the same source ConfigCli's coercion uses when such a field is
    overridden. Fields with no single concrete annotated type are left alone.
    """
    target = ConfigCli._unwrap_optional(ConfigCli._annotation(config, path))
    return _NONE_SAMPLES.get(typing.get_origin(target) or target)


def _mutate(value):
    """Returns a same-type value that differs from the given leaf default, or None to skip.

    Skips None-valued leaves (handled by ``_fill_optional``) and single-member
    enums (no second member to switch to).
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        others = [member for member in type(value) if member is not value]
        return others[0] if others else None
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 2 + 0.5
    if isinstance(value, Path):
        return Path(str(value) + "_copy")
    if isinstance(value, str):
        return value + "_copy"
    if isinstance(value, list):
        flipped = list(reversed(value))
        return flipped if flipped != value else value + (value[:1] or [1])
    if isinstance(value, tuple):
        flipped = tuple(reversed(value))
        return flipped if flipped != value else value + (value[:1] or (1,))
    if isinstance(value, dict):
        return {**value, "copy_sweep_probe": 1}
    return None


def _saved_and_read(config, tmp_path: Path) -> dict:
    """Saves a config the way a run does and returns the reader's whole result."""
    run_dir = tmp_path / "run"
    ConfigCli.save_resolved(config, run_dir / "docs" / "resolved_entry_config.json")

    result = RunConfigReader(WebLogger()).read(str(run_dir))
    assert result["ok"], result
    return result


@pytest.mark.parametrize("key", _ENTRY_KEYS)
def test_default_run_copies_with_zero_overrides(key, tmp_path):
    """An unmodified run of every entry point reads back string-equal on every field.

    This is the zero-override property: the launch form marks a copied field as
    an override only when its rendered value differs from the leaf default, so
    any mismatch here shows up in the UI as a phantom override (or hides a real
    one). The sweep compares every single field of the entry config.
    """
    config = _entry_config_class(key)()
    result = _saved_and_read(config, tmp_path)
    copied = result["config"]

    leaves   = {path: value for path, value in ConfigCli._leaves(config)}
    rendered = {path: ConfigValueRenderer.render(value) for path, value in leaves.items()}

    unknown = sorted(set(copied) - set(leaves))
    assert unknown == [], f"{key}: file holds keys the form has no field for: {unknown}"

    identity = sorted(path for path in leaves if _is_identity(path))
    assert result["identity"] == identity, f"{key}: withheld set diverges from the identity fields: {result['identity']} != {identity}"

    editable_missing = sorted(path for path, value in leaves.items() if ConfigCli.is_editable(value) and path not in copied and not _is_identity(path))
    assert editable_missing == [], f"{key}: editable fields absent from the run file: {editable_missing}"

    mismatches = {path: (copied[path], rendered[path]) for path in copied if copied[path] != rendered[path]}
    assert mismatches == {}, f"{key}: {len(mismatches)} fields would show as phantom overrides: {mismatches}"


@pytest.mark.parametrize("key", _ENTRY_KEYS)
def test_every_changed_field_survives_the_copy_relaunch_cycle(key, tmp_path):
    """Every editable field, once changed, round-trips through copy and relaunch.

    Mutates every editable leaf of the entry config other than the withheld
    identity fields, saves it as a run would, reads it back through the reader,
    and checks that (a) exactly the mutated fields differ from the defaults — the copied override set is precisely the
    run's real changes — and (b) feeding those rendered strings back through
    the ConfigCli override parser reproduces every mutated value on a fresh
    config, which is what launching the copied configuration does.
    """
    config_class = _entry_config_class(key)
    source       = config_class()

    mutated = {}
    for path, value in list(ConfigCli._leaves(source)):
        if not ConfigCli.is_editable(value) or _is_identity(path):
            continue
        changed = _fill_optional(source, path) if value is None else _mutate(value)
        if changed is None:
            continue
        ConfigCli.set_path(source, path, changed)
        mutated[path] = changed

    assert mutated, f"{key}: no editable field could be mutated"

    copied   = _saved_and_read(source, tmp_path)["config"]
    defaults = {path: ConfigValueRenderer.render(value) for path, value in ConfigCli._leaves(config_class())}

    overrides = {path for path in copied if copied[path] != defaults.get(path)}
    assert overrides == set(mutated), (
        f"{key}: override set diverges from the real changes; "
        f"phantom={sorted(overrides - set(mutated))} missed={sorted(set(mutated) - overrides)}"
    )

    argv  = [f"--{path}={copied[path]}" for path in sorted(mutated)]
    fresh = ConfigCli(config_class(), description=None).apply(argv)

    relaunched = {path: value for path, value in ConfigCli._leaves(fresh)}
    wrong      = {path: (relaunched[path], mutated[path]) for path in mutated if relaunched[path] != mutated[path]}
    assert wrong == {}, f"{key}: {len(wrong)} fields relaunch with a different value: {wrong}"


@pytest.mark.parametrize("key", ["train_backbone", "train_jepa", "train_profile_autoencoder", "train_image_autoencoder", "train_unrolled", "train_dual"])
def test_preflight_fields_are_part_of_the_copy_surface(key):
    """Every training entry exposes its pretrain preflight fields to the copy path.

    The preflight probes (batch-size finder, loader tuner, LR range test) must be
    carried along when a run is copied, or a rerun silently preflights differently.
    """
    config = _entry_config_class(key)()
    leaves = {path: value for path, value in ConfigCli._leaves(config)}

    pretrain = sorted(path for path in leaves if path.startswith("pretrain."))
    assert pretrain, f"{key}: no pretrain preflight fields found"

    uneditable = [path for path in pretrain if not ConfigCli.is_editable(leaves[path])]
    assert uneditable == [], f"{key}: preflight fields the copy path cannot set: {uneditable}"


def test_reader_rendering_matches_the_live_resolver_bootstrap(tmp_path):
    """The reader reproduces, field by field, what the real resolver subprocess prints.

    Every other parity check in this file compares the reader against
    ``ConfigValueRenderer`` in-process. This one runs ``ScriptConfigResolver``
    for real — the bootstrap snippet, under an interpreter, on the live entry
    config — so a change to the bootstrap that the renderer does not follow
    fails here instead of silently turning copied fields into phantom overrides.
    """
    key      = "train_backbone"
    resolved = ScriptConfigResolver(ProjectPaths()).resolve(key, sys.executable)
    assert resolved["ok"], resolved

    copied = _saved_and_read(_entry_config_class(key)(), tmp_path)["config"]

    mismatches = {leaf["path"]: (copied[leaf["path"]], leaf["value"]) for leaf in resolved["leaves"] if leaf["path"] in copied and copied[leaf["path"]] != leaf["value"]}
    assert mismatches == {}, f"{len(mismatches)} fields render differently from the resolver bootstrap: {mismatches}"

    assert any(leaf["path"] in copied for leaf in resolved["leaves"]), "the resolver and the reader share no field"
