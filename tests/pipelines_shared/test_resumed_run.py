"""Tests for resuming a staged run from any path inside it with its recorded configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuration.cross_validation             import CrossValidationConfig
from pipelines.shared.orchestration.resumed_run import ResumedRun, StagedRunLocator
from tools.runtime.config_cli                   import ConfigCli

from tests.conftest import RecordingLogger


def _recorded_run(tmp_path: Path, **overrides) -> Path:
    """Writes a cross-validation run directory whose recorded configuration carries the given field values."""
    config = CrossValidationConfig()
    for key, value in overrides.items():
        ConfigCli.set_path(config, key, value)

    run_dir = tmp_path / "runs" / "cv_alpha"
    ConfigCli.save_resolved(config, StagedRunLocator.config_path(run_dir))
    (run_dir / "folds" / "resunet_fold_0" / "meta").mkdir(parents=True)

    return run_dir


def _launch(resume_from: str, argv: list[str]) -> tuple:
    """Returns the launch config and its CLI for a resume of the given path with the given overrides."""
    cli    = ConfigCli(CrossValidationConfig())
    config = cli.apply(["--resume_from", resume_from, *argv])
    return config, cli


def test_locator_accepts_the_run_directory_and_anything_inside_it(tmp_path):
    """The run root is found from the run itself, its pipeline directory, its folds and one fold run inside."""
    run_dir = _recorded_run(tmp_path)

    assert StagedRunLocator.root(str(run_dir))                                  == run_dir
    assert StagedRunLocator.root(str(run_dir / "pipeline"))                     == run_dir
    assert StagedRunLocator.root(str(run_dir / "folds"))                        == run_dir
    assert StagedRunLocator.root(str(run_dir / "folds" / "resunet_fold_0"))     == run_dir
    assert StagedRunLocator.root(str(StagedRunLocator.config_path(run_dir)))    == run_dir


def test_locator_rejects_a_missing_path_and_a_path_outside_any_run(tmp_path):
    """A path that does not exist, or lies in no staged run, is refused with a pointer to the recorded config."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        StagedRunLocator.root(str(tmp_path / "nope"))

    (tmp_path / "plain").mkdir()
    with pytest.raises(FileNotFoundError, match="lies in no staged run"):
        StagedRunLocator.root(str(tmp_path / "plain"))


def test_resumed_run_loads_the_recorded_configuration_and_points_at_the_run(tmp_path):
    """The recorded leaves come back, the run tag and log base follow the path, and resume is on."""
    run_dir = _recorded_run(tmp_path, **{"training.epochs": 7, "backbone_name": "unet", "resume": False, "run_tag": "original"})

    launch, cli = _launch(str(run_dir / "folds"), [])
    resumed     = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    config      = resumed.load()

    assert config.training.epochs   == 7
    assert config.backbone_name     == "unet"
    assert config.run_tag           == "cv_alpha"
    assert config.paths.log_base_dir == run_dir.parent
    assert config.resume            is True
    assert config.resume_from       == ""


def test_typed_overrides_win_over_the_recording_and_are_reported(tmp_path):
    """An override typed on the resume launch replaces the recorded value and shows up in the change table."""
    run_dir = _recorded_run(tmp_path, **{"training.epochs": 7, "seeds": [0, 1, 2]})

    launch, cli = _launch(str(run_dir), ["--training.epochs", "9"])
    resumed     = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    config      = resumed.load()

    assert config.training.epochs == 9
    assert config.seeds           == [0, 1, 2]
    assert resumed.changed["training.epochs"] == (7, 9)
    assert "seeds" not in resumed.changed


def test_hardware_and_stage_come_from_the_launch_not_the_recording(tmp_path):
    """The GPU pool, the pool file, the poll interval and the stage follow the current launch."""
    run_dir = _recorded_run(tmp_path, gpus=[2, 3], poll_interval_s=9.0, stage="full")

    launch, cli = _launch(str(run_dir), ["--stage", "prepare"])
    launch.gpus = [0]
    config      = ResumedRun(CrossValidationConfig, launch, cli.overrides).load()

    assert config.gpus            == [0]
    assert config.stage           == "prepare"
    assert config.poll_interval_s == launch.poll_interval_s


def test_the_report_names_the_source_and_every_changed_leaf(tmp_path):
    """The log names the recorded file and lists each changed leaf with its recorded and new value."""
    run_dir = _recorded_run(tmp_path, **{"training.epochs": 7})

    launch, cli = _launch(str(run_dir), ["--training.epochs", "9"])
    resumed     = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    resumed.load()

    logger = RecordingLogger()
    resumed.log(logger)

    assert ("section", "Resumed run") in logger.messages
    assert ("metrics_table", "1 leaf(s) changed for this launch") in logger.messages


def test_the_report_says_when_nothing_changed(tmp_path):
    """A resume without overrides reports that the recorded configuration is used as it is."""
    run_dir = _recorded_run(tmp_path)

    launch, cli = _launch(str(run_dir), [])
    resumed     = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    resumed.load()

    logger = RecordingLogger()
    resumed.log(logger)

    assert ("ok", "the run continues with exactly the configuration it recorded") in logger.messages


def test_tuning_resume_locates_the_recording_beside_the_study_database(tmp_path):
    """A tuning run records its configuration in its run directory, and the resume finds it from any path inside."""
    from configuration.tuning                       import TuningEntryConfig
    from pipelines.shared.orchestration.resumed_run import TuningResumedRun

    recorded = TuningEntryConfig()
    recorded.tuning.n_trials = 12
    run_dir = tmp_path / "tuning" / "study_a"
    ConfigCli.save_resolved(recorded, run_dir / "resolved_config.json")
    (run_dir / "optuna.db").write_bytes(b"")

    cli    = ConfigCli(TuningEntryConfig())
    launch = cli.apply(["--resume_from", str(run_dir / "optuna.db"), "--tuning.n_trials", "20"])
    config = TuningResumedRun(TuningEntryConfig, launch, cli.overrides).load()

    assert config.run_tag            == "study_a"
    assert config.paths.log_base_dir == run_dir.parent
    assert config.tuning.n_trials    == 20
    assert config.resume_from        == ""
