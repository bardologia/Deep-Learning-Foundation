
from __future__ import annotations

from pathlib import Path


from configuration.cross_validation             import CrossValidationConfig
from pipelines.shared.orchestration.resumed_run import ResumedRun, StagedRunLocator
from tools.runtime.config_cli                   import ConfigCli

from tests.conftest import RecordingLogger


def _recorded_run(tmp_path: Path, **overrides) -> Path:
    config = CrossValidationConfig()
    for key, value in overrides.items():
        ConfigCli.set_path(config, key, value)

    run_dir = tmp_path / "runs" / "cv_alpha"
    ConfigCli.save_resolved(config, StagedRunLocator.config_path(run_dir))
    (run_dir / "folds" / "resunet_fold_0" / "meta").mkdir(parents=True)

    return run_dir


def _launch(resume_from: str, argv: list[str]) -> tuple:
    cli    = ConfigCli(CrossValidationConfig())
    config = cli.apply(["--resume_from", resume_from, *argv])
    return config, cli


def test_locator_accepts_the_run_directory_and_anything_inside_it(tmp_path):
    run_dir = _recorded_run(tmp_path)

    assert StagedRunLocator.root(str(run_dir))                                  == run_dir
    assert StagedRunLocator.root(str(run_dir / "pipeline"))                     == run_dir
    assert StagedRunLocator.root(str(run_dir / "folds"))                        == run_dir
    assert StagedRunLocator.root(str(run_dir / "folds" / "resunet_fold_0"))     == run_dir
    assert StagedRunLocator.root(str(StagedRunLocator.config_path(run_dir)))    == run_dir


def test_resumed_run_loads_the_recorded_configuration_and_points_at_the_run(tmp_path):
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
    run_dir = _recorded_run(tmp_path, **{"training.epochs": 7, "seeds": [0, 1, 2]})

    launch, cli = _launch(str(run_dir), ["--training.epochs", "9"])
    resumed     = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    config      = resumed.load()

    assert config.training.epochs == 9
    assert config.seeds           == [0, 1, 2]
    assert resumed.changed["training.epochs"] == (7, 9)
    assert "seeds" not in resumed.changed


def test_hardware_and_stage_come_from_the_launch_not_the_recording(tmp_path):
    run_dir = _recorded_run(tmp_path, gpus=[2, 3], poll_interval_s=9.0, stage="full")

    launch, cli = _launch(str(run_dir), ["--stage", "prepare"])
    launch.gpus = [0]
    config      = ResumedRun(CrossValidationConfig, launch, cli.overrides).load()

    assert config.gpus            == [0]
    assert config.stage           == "prepare"
    assert config.poll_interval_s == launch.poll_interval_s


def test_the_report_names_the_source_and_every_changed_leaf(tmp_path):
    run_dir = _recorded_run(tmp_path, **{"training.epochs": 7})

    launch, cli = _launch(str(run_dir), ["--training.epochs", "9"])
    resumed     = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    resumed.load()

    logger = RecordingLogger()
    resumed.log(logger)

    assert ("section", "Resumed run") in logger.messages
    assert ("metrics_table", "1 leaf(s) changed for this launch") in logger.messages


def test_the_report_says_when_nothing_changed(tmp_path):
    run_dir = _recorded_run(tmp_path)

    launch, cli = _launch(str(run_dir), [])
    resumed     = ResumedRun(CrossValidationConfig, launch, cli.overrides)
    resumed.load()

    logger = RecordingLogger()
    resumed.log(logger)

    assert ("ok", "the run continues with exactly the configuration it recorded") in logger.messages


def test_a_resumed_tuning_run_takes_its_stage_from_the_launch(tmp_path):
    from configuration.tuning                       import TuningEntryConfig
    from pipelines.shared.orchestration.resumed_run import TuningResumedRun

    recorded       = TuningEntryConfig()
    recorded.stage = "prepare"
    run_dir        = tmp_path / "tuning" / "study_b"
    ConfigCli.save_resolved(recorded, run_dir / "resolved_config.json")

    cli    = ConfigCli(TuningEntryConfig())
    launch = cli.apply(["--resume_from", str(run_dir), "--stage", "compare"])
    config = TuningResumedRun(TuningEntryConfig, launch, cli.overrides).load()

    assert config.stage == "compare"


def test_tuning_resume_locates_the_recording_beside_the_study_database(tmp_path):
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
