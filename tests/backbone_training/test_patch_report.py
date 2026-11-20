
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipelines.backbone.training.launcher as backbone_launcher
from configuration.training                  import BackboneEntryConfig
from configuration.training.backbone         import PatchTrialsConfig
from pipelines.backbone.training.experiments import PatchSizeTrialPlanner
from pipelines.backbone.training.patch_report import PatchTrialCollector, PatchTrialsReport
from tools.monitoring.logger                 import Logger
from tools.runtime.completion                import CompletionMarker


OPTIMUM = (24, 16)


def make_planner(maximum: tuple[int, int] = (32, 16)) -> PatchSizeTrialPlanner:
    config = BackboneEntryConfig()
    trials = PatchTrialsConfig(maximum=maximum)

    return PatchSizeTrialPlanner(trials, "unet", {"features": [64, 128, 256]}, config.training)


def synthetic_loss(patch_size: tuple[int, int]) -> float:
    row_opt, range_opt = OPTIMUM
    row,     range_len = patch_size
    return 0.1 + 0.02 * ((row - row_opt) / 32) ** 2 + 0.02 * ((range_len - range_opt) / 32) ** 2


def populate_runs(runs_root: Path, planner: PatchSizeTrialPlanner, seeds: list[int] | None = None, drop_metrics: set[str] = frozenset(), nan_units: set[str] = frozenset(), prefix: str = "") -> None:
    for name, overrides in planner.plan():
        patch_size = overrides["training.patch_size"]
        trial_dir  = runs_root / f"{prefix}{name}"

        for offset, seed in enumerate(seeds or [None]):
            unit    = name if seed is None else f"{name}/seed{seed}"
            run_dir = trial_dir if seed is None else trial_dir / f"seed{seed}"
            loss    = float("nan") if unit in nan_units else synthetic_loss(patch_size) + 0.01 * offset

            if name not in drop_metrics:
                meta = run_dir / "meta"
                meta.mkdir(parents=True, exist_ok=True)
                (meta / "test_metrics.json").write_text(json.dumps({"avg_loss": loss, "num_batches": 4}))

            run_dir.mkdir(parents=True, exist_ok=True)
            CompletionMarker.stamp(run_dir, {"stage": "training", "best_val_loss": loss * 1.05, "best_epoch": 12})


@pytest.fixture
def logger(tmp_path):
    log = Logger(log_dir=str(tmp_path / "logs"), name="patch_report_test")
    yield log
    log.close()


def _log_text(tmp_path: Path) -> str:
    return (tmp_path / "logs" / "patch_report_test.log").read_text(encoding="utf-8")


def make_report(tmp_path: Path, planner: PatchSizeTrialPlanner, logger: Logger, seeds: list[int] | None = None) -> PatchTrialsReport:
    config       = BackboneEntryConfig()
    config.seeds = seeds or [0]

    return PatchTrialsReport(config=config, planner=planner, model_label="unet", runs_root=tmp_path / "runs", out_dir=tmp_path / "report", logger=logger)


def test_collector_reads_every_trial(tmp_path, logger):
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner)

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger, seeds=[0]).collect()

    assert len(records) == len(planner.plan())
    assert all(record.complete for record in records)
    assert all(record.status == "DONE" for record in records)
    assert all(record.best_epoch == 12 for record in records)


def test_collector_resolves_label_prefixed_run_directories(tmp_path, logger):
    planner = make_planner(maximum=(16, 16))
    populate_runs(tmp_path / "runs", planner, prefix="unet_")

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger, seeds=[0]).collect()

    assert all(record.complete for record in records)
    assert all(record.status == "DONE" for record in records)


def test_collector_marks_trials_without_metrics_incomplete(tmp_path, logger):
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner, drop_metrics={"p-016x008"})

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger, seeds=[0]).collect()
    by_name = {record.name: record for record in records}

    assert not by_name["p-016x008"].complete
    assert by_name["p-008x008"].complete
    assert "p-016x008: no test metrics, excluded from the trial mean" in _log_text(tmp_path)


def test_collector_marks_unrun_trials_missing(tmp_path, logger):
    planner = make_planner()
    (tmp_path / "runs").mkdir()

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger, seeds=[0]).collect()

    assert all(record.status == "MISSING" for record in records)
    assert all(record.test_loss is None for record in records)


def test_collector_aggregates_nested_seed_runs(tmp_path, logger):
    planner = make_planner(maximum=(16, 16))
    populate_runs(tmp_path / "runs", planner, seeds=[0, 1])

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger, seeds=[0, 1]).collect()

    assert all(record.n_seeds == 2 for record in records)
    assert all(record.status == "DONE" for record in records)
    assert all(record.test_loss_std is not None for record in records)

    expected = synthetic_loss(records[0].patch_size) + 0.005
    assert records[0].test_loss == pytest.approx(expected)
    assert [run["name"] for run in records[0].seed_runs] == [f"{records[0].name}/seed0", f"{records[0].name}/seed1"]


def test_collector_marks_a_planned_seed_without_a_directory(tmp_path, logger):
    planner = make_planner(maximum=(16, 16))
    populate_runs(tmp_path / "runs", planner, seeds=[0])

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger, seeds=[0, 1]).collect()

    assert all(record.status == "PARTIAL" for record in records)
    assert all(record.n_seeds == 1 and record.n_seed_runs == 2 for record in records)
    assert all(record.test_loss == pytest.approx(synthetic_loss(record.patch_size)) for record in records)
    assert "p-008x008/seed1: no completion marker (run directory missing or run unfinished), excluded from the trial mean" in _log_text(tmp_path)


def test_collector_excludes_a_diverged_seed_and_names_it(tmp_path, logger):
    planner = make_planner(maximum=(16, 16))
    populate_runs(tmp_path / "runs", planner, seeds=[0, 1], nan_units={"p-008x008/seed1"})

    records = {record.name: record for record in PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger, seeds=[0, 1]).collect()}

    assert records["p-008x008"].status    == "PARTIAL"
    assert records["p-008x008"].n_seeds   == 1
    assert records["p-008x008"].test_loss == pytest.approx(synthetic_loss((8, 8)))
    assert records["p-016x016"].status    == "DONE"
    assert "p-008x008/seed1: test loss nan is not finite, excluded from the trial mean" in _log_text(tmp_path)


def test_report_ranks_the_synthetic_optimum_on_both_axes(tmp_path, logger):
    planner = make_planner(maximum=(32, 16))
    populate_runs(tmp_path / "runs", planner)

    make_report(tmp_path, planner, logger).write_all()

    payload = json.loads((tmp_path / "report" / "patch_trials.json").read_text())

    assert payload["best_patch_size"] == [24, 16]
    assert payload["row_sizes"]   == [8, 16, 24, 32]
    assert payload["col_sizes"]     == [8, 16]
    assert payload["patch_step"]      == 8


def test_report_writes_the_heatmap_and_the_markdown(tmp_path, logger):
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner)

    written = make_report(tmp_path, planner, logger).write_all()

    assert (tmp_path / "report" / "heatmap.png").exists()
    assert (tmp_path / "report" / "report.md").exists()
    assert all(path.exists() for path in written)


def test_report_falls_back_to_a_curve_when_only_one_axis_varies(tmp_path, logger):
    planner = make_planner(maximum=(32, 8))
    populate_runs(tmp_path / "runs", planner)

    make_report(tmp_path, planner, logger).write_all()

    assert (tmp_path / "report" / "curve.png").exists()
    assert not (tmp_path / "report" / "heatmap.png").exists()


def test_report_survives_a_sweep_with_no_metrics(tmp_path, logger):
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner, drop_metrics={name for name, _ in planner.plan()})

    make_report(tmp_path, planner, logger).write_all()

    payload = json.loads((tmp_path / "report" / "patch_trials.json").read_text())

    assert payload["best_patch_size"] is None
    assert (tmp_path / "report" / "report.md").exists()
    assert "No patch shape ranked: none of the 8 trials has a finite seed-mean test loss" in _log_text(tmp_path)


def test_report_logs_the_ranking_and_the_winner(tmp_path, logger):
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner, drop_metrics={"p-016x008"})

    make_report(tmp_path, planner, logger).write_all()

    text = _log_text(tmp_path)

    assert "Ranked 7/8 patch shapes; best 24x16 with test loss" in text
    assert "over 1/1 seeds" in text


def test_markdown_names_the_ranking_metric_and_the_model(tmp_path, logger):
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner)

    make_report(tmp_path, planner, logger).write_all()

    text = (tmp_path / "report" / "report.md").read_text()

    assert "seed-mean test avg_loss" in text
    assert "unet" in text
    assert "**24x16**" in text


def test_scheduler_writes_the_patch_report_under_the_runs_root(tmp_path):
    planner = make_planner(maximum=(16, 16))
    populate_runs(tmp_path / "patch", planner)

    config                      = BackboneEntryConfig()
    config.logdir               = tmp_path
    config.seeds                = [0]
    config.trials_mode          = "patch"
    config.backbone_name        = "unet"
    config.model_overrides      = {"features": [64, 128, 256]}
    config.patch_trials.maximum = (16, 16)

    scheduler = backbone_launcher.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))
    scheduler._write_mode_report()
    scheduler.logger.close()

    reports = sorted((tmp_path / "patch" / "report").iterdir())

    assert len(reports)                       == 1
    assert (reports[0] / "report.md").exists()


def test_scheduler_writes_no_report_outside_patch_mode(tmp_path):
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "amp"

    scheduler = backbone_launcher.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))
    scheduler._write_mode_report()

    assert not (tmp_path / "amp" / "report").exists()


def _patch_scheduler(tmp_path, cli_overrides: dict):
    config                      = BackboneEntryConfig()
    config.logdir               = tmp_path
    config.seeds                = [0]
    config.trials_mode          = "patch"
    config.backbone_name        = "unet"
    config.model_overrides      = {"features": [64, 128, 256]}
    config.patch_trials.maximum = (16, 16)

    return backbone_launcher.TrainScheduler(config=config, cli_overrides=cli_overrides, entry_script=Path("/entry/train_backbone.py"))


def test_scheduler_warns_when_a_trial_replaces_a_forwarded_override(tmp_path):
    scheduler = _patch_scheduler(tmp_path, {"training.patch_size": (4, 4), "training.epochs": 3})
    scheduler._warn_shadowed_overrides()
    scheduler.logger.close()

    text = (Path(scheduler.log_dir) / "train_scheduler.log").read_text(encoding="utf-8")

    assert "Forwarded override training.patch_size=(4, 4) is replaced by the patch planner in 4 of 4 trial(s) (trial values: (8, 8), (8, 16), (16, 8), (16, 16))" in text
    assert "training.epochs" not in text


def test_scheduler_stays_quiet_when_no_trial_replaces_a_forwarded_override(tmp_path):
    scheduler = _patch_scheduler(tmp_path, {"training.epochs": 3})
    scheduler._warn_shadowed_overrides()
    scheduler.logger.close()

    text = (Path(scheduler.log_dir) / "train_scheduler.log").read_text(encoding="utf-8")

    assert "Forwarded override" not in text
    assert backbone_launcher.TrainScheduler._same_value((8, 8), [8, 8])
