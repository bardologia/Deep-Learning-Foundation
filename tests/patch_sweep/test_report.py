from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from configuration.patch_sweep     import PatchSweepConfig
from pipelines.patch_sweep.planner import PatchSweepPlanner
from pipelines.patch_sweep.report  import PatchSweepReport, SweepCollector
from tools.monitoring.logger       import Logger


OPTIMA = {"w20_10": (24, 16), "w20_20": (16, 8)}


def make_planner(tmp_path: Path, datasets: list[str], maximum: tuple[int, int] = (32, 16)) -> PatchSweepPlanner:
    base = tmp_path / "datasets"
    for name in datasets:
        (base / name / "data").mkdir(parents=True)

    config                   = PatchSweepConfig()
    config.dataset_base_path = base
    config.dataset_filter    = []
    config.patch.maximum     = maximum
    return PatchSweepPlanner(config)


def synthetic_loss(unit) -> float:
    azimuth_opt, range_opt = OPTIMA[unit.dataset]
    azimuth,     range_len = unit.patch_size
    return 0.1 + 0.02 * ((azimuth - azimuth_opt) / 32) ** 2 + 0.02 * ((range_len - range_opt) / 32) ** 2


def populate_runs(root: Path, planner: PatchSweepPlanner, drop_metrics: set[str] = frozenset()) -> None:
    results = []
    for unit in planner.units():
        run_dir = root / "training" / unit.name

        if unit.name not in drop_metrics:
            meta = run_dir / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "test_metrics.json").write_text(json.dumps({"avg_loss": synthetic_loss(unit), "num_batches": 4}))

        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        torch.save({"best_val_loss": synthetic_loss(unit) * 1.05, "best_epoch": 12}, checkpoints / "best_model.pt")

        results.append({"name": unit.name, "status": "DONE", "duration_s": 60.0, "gpu": 0, "returncode": 0, "log_file": ""})

    pipeline_dir = root / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "training_results.json").write_text(json.dumps(results))


@pytest.fixture
def logger(tmp_path):
    log = Logger(log_dir=str(tmp_path / "logs"), name="patch_sweep_test")
    yield log
    log.close()


def test_collector_reads_every_unit(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10", "w20_20"])
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()

    assert len(records) == len(planner.units())
    assert all(record.complete for record in records)
    assert all(record.best_val_loss is not None for record in records)


def test_collector_requires_training_results(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10"])

    with pytest.raises(FileNotFoundError, match="training_results.json"):
        SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()


def test_collector_marks_units_without_metrics_incomplete(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10"])
    populate_runs(tmp_path, planner, drop_metrics={"w20_10-p016x008"})

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    by_name = {record.unit.name: record for record in records}

    assert not by_name["w20_10-p016x008"].complete
    assert by_name["w20_10-p008x008"].complete


def test_report_ranks_the_synthetic_optimum_on_both_axes(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10", "w20_20"], maximum=(32, 16))
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    payload = json.loads((out_dir / "patch_sweep.json").read_text())

    assert payload["datasets"]["w20_10"]["best_patch_size"] == [24, 16]
    assert payload["datasets"]["w20_20"]["best_patch_size"] == [16, 8]
    assert payload["azimuth_sizes"]                         == [8, 16, 24, 32]
    assert payload["range_sizes"]                           == [8, 16]
    assert payload["datasets"]["w20_10"]["dataset_path"]    == str(tmp_path / "datasets" / "w20_10")


def test_report_writes_one_heatmap_per_dataset_and_the_summary_figure(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10", "w20_20"])
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    written = PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    assert (out_dir / "heatmaps" / "w20_10.png").exists()
    assert (out_dir / "heatmaps" / "w20_20.png").exists()
    assert (out_dir / "best_patch_vs_dataset.png").exists()
    assert (out_dir / "report.md").exists()
    assert all(path.exists() for path in written)


def test_report_falls_back_to_a_curve_when_only_one_axis_varies(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(32, 8))
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    assert (out_dir / "curves" / "w20_10.png").exists()
    assert not (out_dir / "heatmaps" / "w20_10.png").exists()


def test_report_survives_a_dataset_with_no_metrics(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10", "w20_20"])
    populate_runs(tmp_path, planner, drop_metrics={unit.name for unit in planner.units() if unit.dataset == "w20_20"})

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    payload = json.loads((out_dir / "patch_sweep.json").read_text())

    assert payload["datasets"]["w20_20"]["best_patch_size"] is None
    assert payload["datasets"]["w20_10"]["best_patch_size"] is not None
    assert not (out_dir / "best_patch_vs_dataset.png").exists()


def test_markdown_reports_the_ranking_metric_without_the_removed_prediction(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10"])
    populate_runs(tmp_path, planner)

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    text = (out_dir / "report.md").read_text()

    assert "seed-mean test avg_loss" in text
    assert "Dataset w20_10" in text
    assert "W*" not in text
    assert "sqrt(N/n)" not in text
    assert "predicted" not in text.lower()


def populate_seeded_runs(root: Path, planner: PatchSweepPlanner, seeds: list[int]) -> None:
    results = []
    for unit in planner.units():
        for offset, seed in enumerate(seeds):
            run_dir = root / "training" / unit.name / f"seed{seed}"
            loss    = synthetic_loss(unit) + 0.01 * offset

            meta = run_dir / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "test_metrics.json").write_text(json.dumps({"avg_loss": loss, "num_batches": 4}))

            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save({"best_val_loss": loss * 1.05, "best_epoch": 12}, checkpoints / "best_model.pt")

            results.append({"name": f"{unit.name}/seed{seed}", "status": "DONE", "duration_s": 60.0, "gpu": 0, "returncode": 0, "log_file": ""})

    pipeline_dir = root / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "training_results.json").write_text(json.dumps(results))


def test_collector_aggregates_nested_seed_runs(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(16, 16))
    populate_seeded_runs(tmp_path, planner, seeds=[0, 1])

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()

    assert all(record.n_seeds == 2 for record in records)
    assert all(record.status == "DONE" for record in records)
    assert all(record.test_loss_std is not None for record in records)

    unit     = records[0].unit
    expected = synthetic_loss(unit) + 0.005
    assert records[0].test_loss == pytest.approx(expected)
    assert [run["name"] for run in records[0].seed_runs] == [f"{unit.name}/seed0", f"{unit.name}/seed1"]


def test_n_seeds_counts_only_the_seeds_that_reported_a_test_loss(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(16, 16))
    populate_seeded_runs(tmp_path, planner, seeds=[0, 1])

    for unit in planner.units():
        (tmp_path / "training" / unit.name / "seed1" / "meta" / "test_metrics.json").unlink()

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()

    assert all(record.n_seeds     == 1 for record in records)
    assert all(record.n_seed_runs == 2 for record in records)
    assert all(record.test_loss_std is None for record in records)


def test_report_annotates_dispersion_for_seeded_runs(tmp_path, logger):
    planner = make_planner(tmp_path, ["w20_10"], maximum=(16, 16))
    populate_seeded_runs(tmp_path, planner, seeds=[0, 1])

    records = SweepCollector(run_dir=tmp_path, planner=planner, logger=logger).collect()
    out_dir = tmp_path / "report"
    PatchSweepReport(records=records, planner=planner, out_dir=out_dir, logger=logger).write_all()

    markdown = (out_dir / "report.md").read_text()
    payload  = json.loads((out_dir / "patch_sweep.json").read_text())

    assert "±" in markdown
    unit_payload = payload["datasets"]["w20_10"]["units"][0]
    assert unit_payload["n_seeds"] == 2
    assert unit_payload["test_loss_std"] is not None
    assert len(unit_payload["seed_runs"]) == 2
