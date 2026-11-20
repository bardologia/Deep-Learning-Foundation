
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pipelines.benchmark.results                   import BenchmarkSeedCollector
from pipelines.shared.comparison.comparison_report import ComparisonReport
from pipelines.shared.comparison.trial_collection  import TrialCollector, TrialRecord

from tools.data.io             import FileIO
from tools.runtime.completion  import CompletionMarker

from tests.conftest import RecordingLogger


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trial_record_has_inference_reflects_inference_dir(tmp_path):
    record = TrialRecord(name="unet", run_dir=tmp_path)
    assert record.has_inference is False

    record.inference_dir = tmp_path / "inf"
    assert record.has_inference is True


def test_collector_raises_without_training_dir(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    with pytest.raises(FileNotFoundError, match="training"):
        TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()


def test_collector_warns_about_unfinished_runs_and_missing_artifacts(tmp_path):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])
    (tmp_path / "training" / "unet").mkdir(parents=True)

    logger  = RecordingLogger()
    records = TrialCollector(run_dir=tmp_path, logger=logger, inference_split="test").collect()

    warnings = [message for level, message in logger.messages if level == "warning"]
    infos    = [message for level, message in logger.messages if level == "info"]
    oks      = [message for level, message in logger.messages if level == "ok"]

    assert len(records) == 1
    assert any("no best_model.pt" in message for message in warnings)
    assert any("no complete.json" in message and "did not finish" in message for message in warnings)
    assert any("absent from training_results.json" in message for message in warnings)
    assert any("no docs/trainer_config.json" in message for message in warnings)
    assert any("no meta/run_summary.json" in message for message in warnings)
    assert any(message.startswith("unet") and message.endswith("no inference, epochs ?, complete no") for message in infos)
    assert oks == ["1 run(s) collected: 0 with inference on split 'test', 1 without complete.json, 1 without best_model.pt"]


def test_collector_reads_the_checkpoint_at_the_run_root_only(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    trial_dir = tmp_path / "training" / "unet"
    (trial_dir / "snapshots").mkdir(parents=True)
    torch.save({"epoch": 1, "best_val_loss": 9.0, "best_epoch": 1}, trial_dir / "snapshots" / "best_model.pt")
    torch.save({"epoch": 9, "best_val_loss": 0.5, "best_epoch": 7}, trial_dir / "best_model.pt")

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()

    assert records[0].checkpoint["best_val_loss"] == pytest.approx(0.5)


def test_collector_reads_the_parameter_count_from_the_run_summary(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])
    _write_json(tmp_path / "training" / "unet" / "meta" / "run_summary.json", {"model_name": "unet", "parameters": 12345678})

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()

    assert len(records) == 1
    assert records[0].parameters == 12345678


def test_collector_reads_checkpoint_with_weights_only_false(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    trial_dir = tmp_path / "training" / "unet"
    trial_dir.mkdir(parents=True)

    checkpoint = {
        "epoch"         : 9,
        "params"        : {"encoder.weight": torch.zeros(64, 64)},
        "grid"          : np.zeros(4, dtype=np.float32),
        "best_val_loss" : 0.123,
        "best_epoch"    : 7,
    }
    torch.save(checkpoint, trial_dir / "best_model.pt")
    _write_json(trial_dir / CompletionMarker.FILENAME, {"stage": "training", "epochs_completed": 10, "epochs_total": 20})

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()

    assert records[0].checkpoint["best_val_loss"]  == pytest.approx(0.123)
    assert records[0].checkpoint["best_epoch"]     == 8
    assert records[0].checkpoint["epoch"]          == 10
    assert records[0].checkpoint["n_train_epochs"] == 10


def test_collector_omits_epoch_count_without_completion_marker(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    trial_dir = tmp_path / "training" / "unet"
    trial_dir.mkdir(parents=True)

    torch.save({"epoch": 4, "params": {}, "best_val_loss": 0.5, "best_epoch": 2}, trial_dir / "best_model.pt")

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()

    assert "n_train_epochs" not in records[0].checkpoint


def test_collector_attaches_inference_metrics_and_media(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    (tmp_path / "training" / "unet").mkdir(parents=True)

    inference_dir = tmp_path / "training" / "unet" / "inference" / "run_a"
    (inference_dir / "figures" / "pointwise").mkdir(parents=True)
    _write_json(inference_dir / "metrics.json", {"rmse_mean": 0.5})
    _write_json(inference_dir / CompletionMarker.FILENAME, {"stage": "inference", "split": "test"})
    (inference_dir / "figures" / "pointwise" / "error.png").write_bytes(b"x")
    (inference_dir / "report.md").write_text("ok", encoding="utf-8")

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()

    record = records[0]
    assert record.has_inference
    assert record.metrics["rmse_mean"] == 0.5
    assert [p.relative_to(inference_dir / "figures").as_posix() for p in record.figures] == ["pointwise/error.png"]
    assert record.report_path is not None


def test_collector_ignores_inference_of_another_split(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    (tmp_path / "training" / "unet").mkdir(parents=True)
    inference_root = tmp_path / "training" / "unet" / "inference"

    for name, split, rmse in (("20260101_000000", "test", 0.5), ("region_override", "val", 9.9)):
        _write_json(inference_root / name / "metrics.json", {"rmse_mean": rmse})
        _write_json(inference_root / name / CompletionMarker.FILENAME, {"stage": "inference", "split": split})

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()

    assert records[0].inference_dir.name          == "20260101_000000"
    assert records[0].metrics["rmse_mean"]    == 0.5


def test_collector_ignores_inference_without_completion_marker(tmp_path, logger_stub):
    (tmp_path / "pipeline").mkdir(parents=True)
    _write_json(tmp_path / "pipeline" / "training_results.json", [])

    (tmp_path / "training" / "unet").mkdir(parents=True)
    _write_json(tmp_path / "training" / "unet" / "inference" / "20260101_000000" / "metrics.json", {"rmse_mean": 0.5})

    records = TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()

    assert records[0].has_inference is False


def test_seed_collector_aggregates_runs_per_model(tmp_path, logger_stub):
    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True)
    _write_json(pipe / "training_results.json", [
        {"name": "unet/seed1", "status": "DONE", "duration_s": 10.0, "log_file": str(tmp_path / "training" / "unet" / "seed1" / "worker.log")},
        {"name": "unet/seed2", "status": "DONE", "duration_s": 20.0, "log_file": str(tmp_path / "training" / "unet" / "seed2" / "worker.log")},
    ])

    for seed, rmse in ((1, 2.0), (2, 4.0)):
        inference_dir = tmp_path / "training" / "unet" / f"seed{seed}" / "inference" / "run_a"
        inference_dir.mkdir(parents=True)
        _write_json(inference_dir / "metrics.json", {"rmse_mean": rmse})
        _write_json(inference_dir / CompletionMarker.FILENAME, {"stage": "inference", "split": "test"})

    collector = BenchmarkSeedCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test")
    records   = collector.collect()

    assert [record.name for record in records] == ["unet"]

    record = records[0]
    assert record.metrics["rmse_mean"]      == 3.0
    assert record.training_result["duration_s"] == pytest.approx(15.0)

    dispersion = collector.seed_dispersion["unet"]
    assert dispersion["n_seeds"]                  == 2
    assert dispersion["n_used"]                   == {"rmse_mean": 2}
    assert dispersion["metrics"]["rmse_mean"] == pytest.approx(float(np.std([2.0, 4.0], ddof=1)))


def test_seed_collector_single_run_is_identity(tmp_path, logger_stub):
    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True)
    _write_json(pipe / "training_results.json", [])
    (tmp_path / "training" / "unet").mkdir(parents=True)

    collector = BenchmarkSeedCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test")
    records   = collector.collect()

    assert [record.name for record in records] == ["unet"]
    assert collector.seed_dispersion           == {}


def test_metric_table_annotates_seed_dispersion(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = ComparisonReport(
        records         = records,
        out_dir         = tmp_path,
        embed_images    = False,
        logger          = logger_stub,
        seed_dispersion = {"unet": {"n_seeds": 2, "n_used": {"r2_mean": 2}, "best_val_loss_std": 0.01, "metrics": {"r2_mean": 0.05}}},
    )

    assert report.has_seed_sweep is True

    lines = "\n".join(report._metric_table(["r2_mean"], records))

    assert "± " in lines


def test_summary_json_includes_seed_dispersion(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = ComparisonReport(
        records         = records,
        out_dir         = tmp_path / "out",
        embed_images    = False,
        logger          = logger_stub,
        seed_dispersion = {"unet": {"n_seeds": 2, "n_used": {}, "best_val_loss_std": None, "metrics": {}}},
    )

    report.write_all()

    payload = json.loads((report.out_dir / "comparison_summary.json").read_text())
    by_name = {entry["name"]: entry for entry in payload}

    assert by_name["unet"]["seed_dispersion"]["n_seeds"] == 2
    assert "seed_dispersion" not in by_name["resunet"]


def _records_with_metrics(out_dir):
    a = TrialRecord(name="unet", run_dir=out_dir / "unet")
    b = TrialRecord(name="resunet", run_dir=out_dir / "resunet")

    a.inference_dir = out_dir / "unet" / "inf"
    b.inference_dir = out_dir / "resunet" / "inf"

    a.metrics = {"rmse_mean": 0.1, "r2_mean": 0.9, "mae_mean": 0.08}
    b.metrics = {"rmse_mean": 0.5, "r2_mean": 0.5, "mae_mean": 0.40}

    return [a, b]


def _report(records, out_dir, logger_stub, rank=True):
    return ComparisonReport(
        records         = records,
        out_dir         = out_dir,
        embed_images    = False,
        logger          = logger_stub,
        rank_models     = rank,
    )


def test_leaderboard_ranks_better_model_first(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub)

    lines = "\n".join(report._leaderboard())

    unet_pos    = lines.index("`unet`")
    resunet_pos = lines.index("`resunet`")
    assert unet_pos < resunet_pos


def test_leaderboard_dominant_model_has_lower_mean_rank(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub)

    lines = [line for line in report._leaderboard() if line.startswith("| ")]
    data  = [line for line in lines if "`unet`" in line or "`resunet`" in line]

    unet_rank    = float([line for line in data if "`unet`" in line][0].split("|")[4])
    resunet_rank = float([line for line in data if "`resunet`" in line][0].split("|")[4])

    assert unet_rank < resunet_rank


def test_leaderboard_empty_without_metrics(tmp_path, logger_stub):
    records = [TrialRecord(name="unet", run_dir=tmp_path / "unet")]
    report  = _report(records, tmp_path, logger_stub)

    lines = report._leaderboard()

    assert any("No inference metrics" in line for line in lines)


def test_metric_table_bolds_best_value(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub)

    lines = "\n".join(report._metric_table(["r2_mean"], records))

    assert "**0.9**" in lines


def test_metric_table_does_not_bold_in_fold_mode(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = _report(records, tmp_path, logger_stub, rank=False)

    lines = "\n".join(report._metric_table(["r2_mean"], records))

    assert "**" not in lines


def test_capacity_table_lists_the_parameter_count(tmp_path, logger_stub):
    record = TrialRecord(name="unet", run_dir=tmp_path / "unet", parameters=1000)
    report = _report([record], tmp_path, logger_stub)

    lines = "\n".join(report._capacity_table())

    assert "| `unet` | 1,000      |" in lines


def test_write_all_round_trips_summary_json(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    out_dir = tmp_path / "comparison"
    report  = _report(records, out_dir, logger_stub)

    written = report.write_all()

    summary_path = out_dir / "comparison_summary.json"
    assert summary_path in written

    payload = FileIO.load_json(summary_path)
    names   = {entry["name"] for entry in payload}
    assert names == {"unet", "resunet"}
    assert (out_dir / "benchmark_overview.md").exists()
    assert (out_dir / "metrics_comparison.md").exists()


def test_training_table_shows_seed_counts_under_a_sweep(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = ComparisonReport(
        records         = records,
        out_dir         = tmp_path,
        embed_images    = False,
        logger          = logger_stub,
        seed_dispersion = {"unet": {"n_seeds": 3, "n_used": {"rmse_mean": 3}, "best_val_loss_std": 0.01, "metrics": {}}},
    )

    lines = "\n".join(report._training_table())

    assert "Seeds" in lines
    assert "| 3" in lines.replace("  ", " ")


def test_training_table_shows_used_over_total_when_a_seed_dropped_out(tmp_path, logger_stub):
    records = _records_with_metrics(tmp_path)
    report  = ComparisonReport(
        records         = records,
        out_dir         = tmp_path,
        embed_images    = False,
        logger          = logger_stub,
        seed_dispersion = {"unet": {"n_seeds": 3, "n_used": {"rmse_mean": 3, "r2_mean": 2}, "best_val_loss_std": 0.01, "metrics": {}}},
    )

    lines = "\n".join(report._training_table())

    assert "| 2/3" in lines.replace("  ", " ")


def test_collector_rebuilds_training_results_from_markers_when_file_absent(tmp_path, logger_stub):
    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True)

    done = tmp_path / "training" / "unet" / "seed1"
    done.mkdir(parents=True)
    CompletionMarker.stamp(done, {"stage": "training", "epochs_completed": 3})

    (tmp_path / "training" / "unet" / "seed2").mkdir(parents=True)

    records = {record.name: record for record in TrialCollector(run_dir=tmp_path, logger=logger_stub, inference_split="test").collect()}

    assert records["unet/seed1"].training_result["status"] == "DONE"
    assert records["unet/seed2"].training_result["status"] == "PARTIAL"


def _seeded_benchmark_run(tmp_path: Path, seed_metrics: dict) -> Path:
    pipe = tmp_path / "pipeline"
    pipe.mkdir(parents=True)
    _write_json(pipe / "training_results.json", [{"name": f"unet/{seed}", "status": "DONE", "duration_s": 10.0, "log_file": str(tmp_path / "training" / "unet" / seed / "worker.log")} for seed in seed_metrics])

    for seed, metrics in seed_metrics.items():
        (tmp_path / "training" / "unet" / seed).mkdir(parents=True)
        if metrics is None:
            continue
        inference_dir = tmp_path / "training" / "unet" / seed / "inference" / "run_a"
        inference_dir.mkdir(parents=True)
        _write_json(inference_dir / "metrics.json", metrics)
        _write_json(inference_dir / CompletionMarker.FILENAME, {"stage": "inference", "split": "test"})

    return tmp_path


def test_seed_collector_warns_about_a_seed_without_metrics_and_records_the_seeds_used(tmp_path, recording_logger):
    run_dir   = _seeded_benchmark_run(tmp_path, {"seed1": {"rmse_mean": 2.0}, "seed2": {"rmse_mean": 4.0}, "seed3": None})
    collector = BenchmarkSeedCollector(run_dir=run_dir, logger=recording_logger, inference_split="test")

    records = collector.collect()

    assert records[0].metrics["rmse_mean"] == 3.0
    assert collector.seed_dispersion["unet"]["n_seeds"]       == 3
    assert collector.seed_dispersion["unet"]["n_used"]        == {"rmse_mean": 2}
    assert collector.seed_dispersion["unet"]["missing_seeds"] == ["seed3"]
    assert any(level == "warning" and message.startswith("unet:") and "seed3" in message and "cover 2 of 3 seeds" in message for level, message in recording_logger.messages)
    assert ("warning", "1 of 1 seed-swept models average fewer seeds than they trained: ['unet']") in recording_logger.messages


def test_seed_collector_names_a_metric_that_is_non_finite_in_one_seed(tmp_path, recording_logger):
    run_dir   = _seeded_benchmark_run(tmp_path, {"seed1": {"rmse_mean": 2.0, "r2_mean": 0.5}, "seed2": {"rmse_mean": 4.0, "r2_mean": float("nan")}})
    collector = BenchmarkSeedCollector(run_dir=run_dir, logger=recording_logger, inference_split="test")

    records = collector.collect()

    assert records[0].metrics["r2_mean"] == 0.5
    assert collector.seed_dispersion["unet"]["n_used"] == {"rmse_mean": 2, "r2_mean": 1}
    assert any(level == "warning" and message.startswith("unet:") and "r2_mean" in message and "cover 1 to 2 of 2 seeds" in message for level, message in recording_logger.messages)


def test_seed_collector_names_a_metric_that_is_non_finite_in_every_seed(tmp_path, recording_logger):
    run_dir   = _seeded_benchmark_run(tmp_path, {"seed1": {"rmse_mean": 2.0, "r2_mean": float("nan")}, "seed2": {"rmse_mean": 4.0, "r2_mean": float("nan")}})
    collector = BenchmarkSeedCollector(run_dir=run_dir, logger=recording_logger, inference_split="test")

    records = collector.collect()

    assert "r2_mean" not in records[0].metrics
    assert any(level == "warning" and message.startswith("unet:") and "r2_mean" in message and "cover 0 to 2 of 2 seeds" in message for level, message in recording_logger.messages)


def test_seed_collector_stays_quiet_when_every_seed_scored(tmp_path, recording_logger):
    run_dir   = _seeded_benchmark_run(tmp_path, {"seed1": {"rmse_mean": 2.0}, "seed2": {"rmse_mean": 4.0}})
    collector = BenchmarkSeedCollector(run_dir=run_dir, logger=recording_logger, inference_split="test")

    collector.collect()

    assert collector.seed_dispersion["unet"]["n_used"] == {"rmse_mean": 2}
    assert [message for level, message in recording_logger.messages if level == "warning" and "seed" in message and "cover" in message] == []


def test_seed_collector_keeps_the_per_run_records_beside_the_aggregates(tmp_path, recording_logger):
    run_dir   = _seeded_benchmark_run(tmp_path, {"seed1": {"rmse_mean": 2.0}, "seed2": None})
    collector = BenchmarkSeedCollector(run_dir=run_dir, logger=recording_logger, inference_split="test")

    records = collector.collect()

    assert [record.name for record in records]        == ["unet"]
    assert [record.name for record in collector.runs] == ["unet/seed1", "unet/seed2"]
    assert [record.has_inference for record in collector.runs] == [True, False]


def test_seed_collector_keeps_the_training_result_of_a_unit_in_a_renamed_directory(tmp_path, recording_logger):
    run_dir = _seeded_benchmark_run(tmp_path, {"seed1": {"rmse_mean": 2.0}, "seed2": {"rmse_mean": 4.0}})

    results = FileIO.load_json(run_dir / "pipeline" / "training_results.json")
    results[1]["name"] = "unet-mse-hv/seed2"
    FileIO.save_json(results, run_dir / "pipeline" / "training_results.json")

    collector = BenchmarkSeedCollector(run_dir=run_dir, logger=recording_logger, inference_split="test")
    records   = collector.collect()

    assert records[0].training_result["status"]     == "DONE"
    assert records[0].training_result["duration_s"] == pytest.approx(10.0)
    assert {run.name: run.training_result["status"] for run in collector.runs} == {"unet/seed1": "DONE", "unet/seed2": "DONE"}
    assert ("info", "1 unit result(s) matched to the renamed run directory holding their worker log: ['unet-mse-hv/seed2 -> unet/seed2']") in recording_logger.messages
