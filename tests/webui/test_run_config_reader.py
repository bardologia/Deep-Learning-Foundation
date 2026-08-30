"""Tests for the run-config reader behind the copy-from-run launch control.

Covers discovery of run directories holding a resolved config file, including
runs nested under a directory that carries one, the priority order of the
candidate file locations, the withholding of the source run's identity fields,
the error paths for absent or unreadable configs, and — most importantly — that
values read back from a ``ConfigCli.save_resolved`` file render through the
shared ``ConfigValueRenderer`` to exactly the strings the script config resolver
produces for equal defaults, so the launch form only marks genuinely different
fields as overrides.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pytest

from config_value_render import ConfigValueRenderer
from run_config_reader   import RunConfigReader
from web_logger          import WebLogger

from tools.runtime.config_cli import ConfigCli


class Matching(Enum):
    """Tiny enum standing in for a config choice field."""

    HUNGARIAN = "hungarian"
    SORTED    = "sorted_gt"


@dataclass
class TrainingSection:
    """Nested section exercising numeric, boolean and container leaves."""

    epochs        : int              = 65
    lr            : float            = 1e-05
    resume        : bool             = False
    patch_size    : tuple            = (64, 32)
    seeds         : list             = field(default_factory=lambda: [0, 1, 2])
    overrides     : dict             = field(default_factory=lambda: {"features": [64, 128], "depth": 3})
    matching      : Matching         = Matching.HUNGARIAN
    dataset_path  : Path             = Path("/data/site")
    run_name      : str              = ""
    infer_after   : str | None       = None


@dataclass
class EntryStub:
    """Entry-style config with a scalar root leaf and one nested section."""

    backbone_name : str             = "resunet"
    training      : TrainingSection = field(default_factory=TrainingSection)


@pytest.fixture
def reader():
    """Returns a reader with a quiet web logger."""
    return RunConfigReader(WebLogger())


def test_read_matches_resolver_rendering(reader, tmp_path):
    """Every value saved by ConfigCli reads back as the resolver's default string.

    This is the invariant the copy control rests on: a copied value equal to a
    field's default must compare string-equal to the resolved leaf, so it stays
    out of the override set.
    """
    config  = EntryStub()
    run_dir = tmp_path / "resunet_20260818_120000"
    ConfigCli.save_resolved(config, run_dir / "docs" / "resolved_entry_config.json")

    result = reader.read(str(run_dir))
    assert result["ok"]
    assert result["run"] == run_dir.name
    assert result["source"] == "docs/resolved_entry_config.json"

    expected = {path: ConfigValueRenderer.render(value) for path, value, _, _ in ConfigCli.detailed_leaves(config)}
    for path, rendered in result["config"].items():
        assert rendered == expected[path], f"{path}: {rendered!r} != {expected[path]!r}"


def test_read_prefers_entry_config_over_other_candidates(reader, tmp_path):
    """A docs entry config wins over pipeline and bare resolved config files."""
    run_dir = tmp_path / "run"
    (run_dir / "docs").mkdir(parents=True)
    (run_dir / "pipeline").mkdir()
    (run_dir / "docs" / "resolved_entry_config.json").write_text('{"a": 1}')
    (run_dir / "pipeline" / "resolved_config.json").write_text('{"b": 2}')
    (run_dir / "resolved_config.json").write_text('{"c": 3}')

    result = reader.read(str(run_dir))
    assert result["source"] == "docs/resolved_entry_config.json"
    assert result["config"] == {"a": "1"}


def test_read_refuses_run_without_config(reader, tmp_path):
    """A directory holding no candidate file is refused with the locations named."""
    (tmp_path / "bare_run").mkdir()

    result = reader.read(str(tmp_path / "bare_run"))
    assert not result["ok"]
    assert "resolved_entry_config.json" in result["error"]


def test_read_refuses_missing_directory_and_broken_json(reader, tmp_path):
    """A missing directory, invalid JSON and a non-mapping payload all fail loudly."""
    assert not reader.read(str(tmp_path / "absent"))["ok"]

    run_dir = tmp_path / "run"
    (run_dir / "docs").mkdir(parents=True)

    (run_dir / "docs" / "resolved_entry_config.json").write_text("{broken")
    assert not reader.read(str(run_dir))["ok"]

    (run_dir / "docs" / "resolved_entry_config.json").write_text("[1, 2]")
    assert not reader.read(str(run_dir))["ok"]


def test_list_runs_finds_each_candidate_kind_at_depth(reader, tmp_path):
    """Listing finds training, staged and bare resolved configs, nested runs included."""
    (tmp_path / "train_run" / "docs").mkdir(parents=True)
    (tmp_path / "train_run" / "docs" / "resolved_entry_config.json").write_text("{}")

    (tmp_path / "sweeps" / "bench_run" / "pipeline").mkdir(parents=True)
    (tmp_path / "sweeps" / "bench_run" / "pipeline" / "resolved_config.json").write_text("{}")

    (tmp_path / "tune_run").mkdir()
    (tmp_path / "tune_run" / "resolved_config.json").write_text("{}")

    (tmp_path / ".hidden" ).mkdir()
    (tmp_path / "no_config").mkdir()

    result = reader.list_runs(str(tmp_path))
    assert result["ok"]

    by_name = {entry["name"]: entry["source"] for entry in result["runs"]}
    assert by_name == {
        "train_run"        : "docs/resolved_entry_config.json",
        "sweeps/bench_run" : "pipeline/resolved_config.json",
        "tune_run"         : "resolved_config.json",
    }


def test_list_runs_refuses_bad_base(reader, tmp_path):
    """An empty or non-directory base is refused instead of listing nothing."""
    assert not reader.list_runs("")["ok"]
    assert not reader.list_runs(str(tmp_path / "absent"))["ok"]


def test_list_runs_finds_runs_nested_under_a_run_that_has_a_config(reader, tmp_path):
    """A parameter run under an already preprocessed dataset is listed alongside it.

    Preprocessing writes ``<dataset>/docs/resolved_entry_config.json`` and every
    parameter extraction writes ``<dataset>/params/<name>/docs/...``, so a walk
    that stopped at the first config would hide every extraction run of every
    dataset that has been preprocessed.
    """
    (tmp_path / "site_a" / "docs").mkdir(parents=True)
    (tmp_path / "site_a" / "docs" / "resolved_entry_config.json").write_text("{}")

    (tmp_path / "site_a" / "params" / "k3_lam0.01" / "docs").mkdir(parents=True)
    (tmp_path / "site_a" / "params" / "k3_lam0.01" / "docs" / "resolved_entry_config.json").write_text("{}")

    (tmp_path / "staged" / "pipeline").mkdir(parents=True)
    (tmp_path / "staged" / "pipeline" / "resolved_config.json").write_text("{}")
    (tmp_path / "staged" / "unit_0" / "docs").mkdir(parents=True)
    (tmp_path / "staged" / "unit_0" / "docs" / "resolved_entry_config.json").write_text("{}")

    names = [entry["name"] for entry in reader.list_runs(str(tmp_path))["runs"]]
    assert names == ["site_a", "site_a/params/k3_lam0.01", "staged", "staged/unit_0"]


def test_read_withholds_the_identity_fields_of_the_source_run(reader, tmp_path):
    """The run's own name and resume state are reported, never served as settings.

    Copying them would point the relaunch at the source run's directory, where
    the resume gate either skips the launch or deletes the copied-from run.
    """
    run_dir = tmp_path / "resunet_20260818_120000"
    (run_dir / "docs").mkdir(parents=True)
    (run_dir / "docs" / "resolved_entry_config.json").write_text(json.dumps({
        "run_name"        : "resunet_20260818_120000",
        "resume"          : True,
        "resume_from"     : "/runs/backbone/resunet_20260818_120000",
        "training.resume" : True,
        "training.epochs" : 65,
    }))

    result = reader.read(str(run_dir))
    assert result["ok"]
    assert result["config"]   == {"training.epochs": "65"}
    assert result["identity"] == ["resume", "resume_from", "run_name", "training.resume"]
