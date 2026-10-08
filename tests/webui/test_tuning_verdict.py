"""Tests for the verdict a finished tuning block records off its run.

Covers which blocks record one, the axes they may declare, how the tuning run
directory follows from the launch's run tag and log base, how the best
configuration is read, how its parameters split over the model_overrides
and training.batch_size axes, and that its batch parameter and progress file
names match the ones the tuners write.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_decisions import TuningVerdict

from pipelines.tuning.tuners import SearchSpace, StudyProgress


def _block(script: str, decisions: list) -> dict:
    """Returns a minimal block with the given script and decisions."""
    return {"script": script, "title": script, "decisions": decisions}


def _axis(path: str) -> dict:
    """Returns an open decision on the given path."""
    return {"path": path, "options": [], "waiting": "", "choice": ""}


def test_only_a_tuning_block_with_decisions_records_a_verdict():
    """A tuning block with a declared axis records; one without, or another entry, does not."""
    verdict = TuningVerdict()

    assert verdict.records(_block("tune", [_axis("model_overrides")])) is True
    assert verdict.records(_block("tune", [])) is False
    assert verdict.records(_block("train_backbone", [_axis("model_overrides")])) is False


def test_the_axes_a_tuning_block_may_declare_are_model_overrides_and_the_batch_size():
    """Both known axes pass, any other is refused with its path named."""
    verdict = TuningVerdict()

    assert verdict.axes(_block("tune", [_axis("model_overrides"), _axis("training.batch_size")])) == ["model_overrides", "training.batch_size"]

    with pytest.raises(ValueError, match="training.epochs"):
        verdict.axes(_block("tune", [_axis("model_overrides"), _axis("training.epochs")]))


def test_the_batch_parameter_name_matches_the_search_space():
    """The verdict splits the batch size off under the name the search space samples it by."""
    assert TuningVerdict.BATCH_PARAM == SearchSpace.BATCH


def test_the_progress_file_name_matches_the_workers_snapshot():
    """The runner reads the snapshot under the name the study progress callback writes."""
    assert TuningVerdict.PROGRESS == StudyProgress.FILENAME


def test_the_run_dir_follows_the_effective_run_tag_and_log_base():
    """The launch override wins over the entry default, and a timestamp run tag is refused."""
    verdict = TuningVerdict()
    leaves  = {"run_tag": "None", "paths.log_base_dir": "/runs/tuning"}

    assert verdict.run_dir({"run_tag": "tag", "paths.log_base_dir": "/ws"}, leaves, "/ws") == Path("/ws/tag")
    assert verdict.run_dir({"run_tag": "tag"}, leaves, "") == Path("/runs/tuning/tag")

    with pytest.raises(ValueError, match="fixed 'run_tag'"):
        verdict.run_dir({}, leaves, "/ws")


def test_read_returns_the_one_best_configuration_and_refuses_none_or_several(tmp_path):
    """One model's parameters come back as a dict; no file and two files both fail."""
    verdict = TuningVerdict()

    with pytest.raises(ValueError, match="found 0"):
        verdict.read(tmp_path)

    path = tmp_path / "unet_skip-set_pred" / "best_config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"model": "unet_skip-set_pred", "trial": 3, "val_loss": 0.2, "params": {"encoder_lr": 0.0003, "batch_size": 512}}))

    assert verdict.read(tmp_path) == {"encoder_lr": 0.0003, "batch_size": 512}

    other = tmp_path / "resunet-set_pred" / "best_config.json"
    other.parent.mkdir()
    other.write_text(json.dumps({"params": {"dropout": 0.1}}))

    with pytest.raises(ValueError, match="found 2"):
        verdict.read(tmp_path)


def test_values_split_the_parameters_over_the_declared_axes():
    """Model parameters render as the dict literal the CLI parses, the batch size as a plain integer."""
    verdict = TuningVerdict()
    params  = {"encoder_lr": 0.0003, "features": [32, 64], "dropout": 0.1, "batch_size": 512}

    assert verdict.values(params, ["model_overrides", "training.batch_size"]) == {"model_overrides": "{'encoder_lr': 0.0003, 'features': [32, 64], 'dropout': 0.1}", "training.batch_size": "512"}
    assert verdict.values({"dropout": 0.1}, ["model_overrides"]) == {"model_overrides": "{'dropout': 0.1}"}


def test_values_refuse_a_verdict_that_would_be_lost_or_an_axis_with_nothing_to_carry():
    """A sampled group without its axis, and an axis without a sampled group, both fail loudly."""
    verdict = TuningVerdict()

    with pytest.raises(ValueError, match="no 'training.batch_size' axis"):
        verdict.values({"dropout": 0.1, "batch_size": 128}, ["model_overrides"])

    with pytest.raises(ValueError, match="no 'model_overrides' axis"):
        verdict.values({"dropout": 0.1, "batch_size": 128}, ["training.batch_size"])

    with pytest.raises(ValueError, match="did not sample the batch size"):
        verdict.values({"dropout": 0.1}, ["model_overrides", "training.batch_size"])

    with pytest.raises(ValueError, match="sampled no model parameter"):
        verdict.values({"batch_size": 128}, ["model_overrides", "training.batch_size"])
