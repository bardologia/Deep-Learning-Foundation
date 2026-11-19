
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

    HUNGARIAN = "hungarian"
    SORTED    = "sorted_gt"


@dataclass
class TrainingSection:

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

    backbone_name : str             = "resunet"
    training      : TrainingSection = field(default_factory=TrainingSection)


@pytest.fixture
def reader():
    return RunConfigReader(WebLogger())


def test_read_matches_resolver_rendering(reader, tmp_path):
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
    (tmp_path / "bare_run").mkdir()

    result = reader.read(str(tmp_path / "bare_run"))
    assert not result["ok"]
    assert "resolved_entry_config.json" in result["error"]


def test_read_refuses_missing_directory_and_broken_json(reader, tmp_path):
    assert not reader.read(str(tmp_path / "absent"))["ok"]

    run_dir = tmp_path / "run"
    (run_dir / "docs").mkdir(parents=True)

    (run_dir / "docs" / "resolved_entry_config.json").write_text("{broken")
    assert not reader.read(str(run_dir))["ok"]

    (run_dir / "docs" / "resolved_entry_config.json").write_text("[1, 2]")
    assert not reader.read(str(run_dir))["ok"]


def test_list_runs_finds_each_candidate_kind_at_depth(reader, tmp_path):
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
    assert not reader.list_runs("")["ok"]
    assert not reader.list_runs(str(tmp_path / "absent"))["ok"]


def test_list_runs_finds_runs_nested_under_a_run_that_has_a_config(reader, tmp_path):
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
