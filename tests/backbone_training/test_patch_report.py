"""Tests covering the patch trials collector and report: reading per-trial metrics and completion markers, seed aggregation, the ranking heatmap, curve, JSON, and markdown that are written, and the scheduler wiring that triggers the report in patch mode only."""

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
    """Builds a step-8 planner capped at the given patch maximum.

    Args:
        maximum : Largest (row, range) patch size in pixels.

    Returns:
        Planner over the sweep-narrowed three-level unet.
    """
    config = BackboneEntryConfig()
    trials = PatchTrialsConfig(maximum=maximum)

    return PatchSizeTrialPlanner(trials, "unet", "conv", {"features": [64, 128, 256]}, config.training)


def synthetic_loss(patch_size: tuple[int, int]) -> float:
    """Returns a quadratic test loss minimised at the synthetic optimal patch size."""
    azimuth_opt, range_opt = OPTIMUM
    row,     range_len = patch_size
    return 0.1 + 0.02 * ((row - azimuth_opt) / 32) ** 2 + 0.02 * ((range_len - range_opt) / 32) ** 2


def populate_runs(runs_root: Path, planner: PatchSizeTrialPlanner, seeds: list[int] | None = None, drop_metrics: set[str] = frozenset()) -> None:
    """Writes per-trial metrics and completion markers for a finished sweep.

    Args:
        runs_root    : Directory the trial run directories are created under.
        planner      : Planner supplying the trials to populate.
        seeds        : Seed values nested under each trial, or None for unseeded runs.
        drop_metrics : Trial names left without a test metrics file.
    """
    for name, overrides in planner.plan():
        patch_size = overrides["training.patch_size"]

        for offset, seed in enumerate(seeds or [None]):
            run_dir = runs_root / name if seed is None else runs_root / name / f"seed{seed}"
            loss    = synthetic_loss(patch_size) + 0.01 * offset

            if name not in drop_metrics:
                meta = run_dir / "meta"
                meta.mkdir(parents=True, exist_ok=True)
                (meta / "test_metrics.json").write_text(json.dumps({"avg_loss": loss, "num_batches": 4}))

            run_dir.mkdir(parents=True, exist_ok=True)
            CompletionMarker.stamp(run_dir, {"stage": "training", "best_val_loss": loss * 1.05, "best_epoch": 12})


@pytest.fixture
def logger(tmp_path):
    """Yields a file logger writing into tmp_path and closes it afterwards."""
    log = Logger(log_dir=str(tmp_path / "logs"), name="patch_report_test")
    yield log
    log.close()


def make_report(tmp_path: Path, planner: PatchSizeTrialPlanner, logger: Logger) -> PatchTrialsReport:
    """Builds a report writer over tmp_path's runs and report directories."""
    config = BackboneEntryConfig()

    return PatchTrialsReport(config=config, planner=planner, model_label="unet-conv", runs_root=tmp_path / "runs", out_dir=tmp_path / "report", logger=logger)


def test_collector_reads_every_trial(tmp_path, logger):
    """Verifies the collector returns one complete record per trial with its validation loss and best epoch."""
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner)

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger).collect()

    assert len(records) == len(planner.plan())
    assert all(record.complete for record in records)
    assert all(record.status == "DONE" for record in records)
    assert all(record.best_epoch == 12 for record in records)


def test_collector_marks_trials_without_metrics_incomplete(tmp_path, logger):
    """Verifies only the trials missing a metrics file are excluded from the ranking."""
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner, drop_metrics={"p-016x008"})

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger).collect()
    by_name = {record.name: record for record in records}

    assert not by_name["p-016x008"].complete
    assert by_name["p-008x008"].complete


def test_collector_marks_unrun_trials_missing(tmp_path, logger):
    """Verifies a trial without a run directory reports MISSING and no losses."""
    planner = make_planner()
    (tmp_path / "runs").mkdir()

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger).collect()

    assert all(record.status == "MISSING" for record in records)
    assert all(record.test_loss is None for record in records)


def test_collector_aggregates_nested_seed_runs(tmp_path, logger):
    """Verifies nested seed runs are averaged into one record carrying the seed count and spread."""
    planner = make_planner(maximum=(16, 16))
    populate_runs(tmp_path / "runs", planner, seeds=[0, 1])

    records = PatchTrialCollector(runs_root=tmp_path / "runs", planner=planner, logger=logger).collect()

    assert all(record.n_seeds == 2 for record in records)
    assert all(record.status == "DONE" for record in records)
    assert all(record.test_loss_std is not None for record in records)

    expected = synthetic_loss(records[0].patch_size) + 0.005
    assert records[0].test_loss == pytest.approx(expected)
    assert [run["name"] for run in records[0].seed_runs] == [f"{records[0].name}/seed0", f"{records[0].name}/seed1"]


def test_report_ranks_the_synthetic_optimum_on_both_axes(tmp_path, logger):
    """Verifies the report recovers the synthetic optimum and records both size grids."""
    planner = make_planner(maximum=(32, 16))
    populate_runs(tmp_path / "runs", planner)

    make_report(tmp_path, planner, logger).write_all()

    payload = json.loads((tmp_path / "report" / "patch_trials.json").read_text())

    assert payload["best_patch_size"] == [24, 16]
    assert payload["row_sizes"]   == [8, 16, 24, 32]
    assert payload["col_sizes"]     == [8, 16]
    assert payload["patch_step"]      == 8


def test_report_writes_the_heatmap_and_the_markdown(tmp_path, logger):
    """Verifies the heatmap and the markdown report are written."""
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner)

    written = make_report(tmp_path, planner, logger).write_all()

    assert (tmp_path / "report" / "heatmap.png").exists()
    assert (tmp_path / "report" / "report.md").exists()
    assert all(path.exists() for path in written)


def test_report_falls_back_to_a_curve_when_only_one_axis_varies(tmp_path, logger):
    """Verifies a single-axis sweep produces a curve instead of a heatmap."""
    planner = make_planner(maximum=(32, 8))
    populate_runs(tmp_path / "runs", planner)

    make_report(tmp_path, planner, logger).write_all()

    assert (tmp_path / "report" / "curve.png").exists()
    assert not (tmp_path / "report" / "heatmap.png").exists()


def test_report_survives_a_sweep_with_no_metrics(tmp_path, logger):
    """Verifies a sweep without any metrics yields a null optimum but still writes the report."""
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner, drop_metrics={name for name, _ in planner.plan()})

    make_report(tmp_path, planner, logger).write_all()

    payload = json.loads((tmp_path / "report" / "patch_trials.json").read_text())

    assert payload["best_patch_size"] is None
    assert (tmp_path / "report" / "report.md").exists()


def test_markdown_names_the_ranking_metric_and_the_model(tmp_path, logger):
    """Verifies the markdown carries the ranking metric, the model label, and the winning patch."""
    planner = make_planner()
    populate_runs(tmp_path / "runs", planner)

    make_report(tmp_path, planner, logger).write_all()

    text = (tmp_path / "report" / "report.md").read_text()

    assert "seed-mean test avg_loss" in text
    assert "unet-conv" in text
    assert "**24x16**" in text


def test_scheduler_writes_the_patch_report_under_the_runs_root(tmp_path):
    """Verifies patch mode writes a timestamped report folder under the sweep runs root."""
    planner = make_planner(maximum=(16, 16))
    populate_runs(tmp_path / "patch", planner)

    config                      = BackboneEntryConfig()
    config.logdir               = tmp_path
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
    """Verifies a non-patch trial mode leaves the runs root without a report folder."""
    config             = BackboneEntryConfig()
    config.logdir      = tmp_path
    config.trials_mode = "ablation"

    scheduler = backbone_launcher.TrainScheduler(config=config, cli_overrides={}, entry_script=Path("/entry/train_backbone.py"))
    scheduler._write_mode_report()

    assert not (tmp_path / "ablation" / "report").exists()
