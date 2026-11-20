from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from configuration.benchmark                                 import BenchmarkConfig
from configuration.inference                                 import InferenceConfig, TargetDatasetConfig
from pipelines.backbone.inference.analysis.seed_disagreement import SeedMaps
from pipelines.backbone.inference.pipeline                   import InferencePipeline
from pipelines.shared.config.config_factory                  import ConfigFactory
from pipelines.shared.dataset.dataset_files                  import DatasetFiles
from pipelines.shared.inference.metadata                     import CompletedInference
from tools.runtime.completion                                import CompletionMarker

from tests.backbone_inference._helpers import build_run_directory


def _config(run_dir: Path, **overrides) -> InferenceConfig:
    return InferenceConfig(run_directory=run_dir, output_subdir="stamp", device="cpu", num_workers=0, **overrides)


def test_inference_writes_maps_metrics_figures_report_and_marker(tmp_path, test_data_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    report  = InferencePipeline(_config(run_dir)).run()
    stamp   = report.parent
    region  = DatasetFiles.read(test_data_dir).splits.regions("test")[0]

    metrics     = json.loads((stamp / "metrics.json").read_text())
    predictions = np.load(stamp / "maps" / "predictions.npy")
    targets     = np.load(stamp / "maps" / "targets.npy")

    assert predictions.shape == targets.shape == (2, region.row_size, region.col_size)
    assert {"rmse_pointwise", "r2_neighbourhood", "rmse_mean", "r2_mean"} <= set(metrics)
    assert metrics["split"] == "test"
    assert metrics["checkpoint_epoch"] == 2
    assert sorted(path.name for path in (stamp / "figures" / "pointwise").iterdir()) == ["error.png", "prediction.png", "target.png"]
    assert "| pointwise" in report.read_text()
    assert CompletionMarker.is_complete(stamp)
    assert CompletedInference.latest(run_dir / "inference", "test") == stamp


def test_inference_skips_maps_and_figures_when_asked(tmp_path, test_data_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    stamp   = InferencePipeline(_config(run_dir, save_maps=False, save_plots=False)).run().parent

    assert not (stamp / "maps").exists()
    assert not (stamp / "figures").exists()
    assert (stamp / "metrics.json").is_file()


def test_the_benchmark_inference_config_writes_the_maps_seed_comparison_reads(tmp_path, test_data_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    config  = ConfigFactory(BenchmarkConfig()).inference_config(run_dir)

    config.device, config.num_workers, config.output_subdir = "cpu", 0, "stamp"
    stamp = InferencePipeline(config).run().parent
    maps  = SeedMaps([run_dir], [stamp], "maps")

    assert maps.predictions().shape[1:] == maps.targets().shape
    assert maps.channels == ["pointwise", "neighbourhood"]


def test_a_run_scored_on_a_target_dataset_stamps_under_its_cross_dataset_tree(tmp_path, test_data_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    other   = tmp_path / "other_dataset"
    shutil.copytree(test_data_dir, other)

    stamp   = InferencePipeline(_config(run_dir, target=TargetDatasetConfig(enabled=True, dataset_path=other))).run().parent
    metrics = json.loads((stamp / "metrics.json").read_text())

    assert stamp.parent.parent.name == "cross_dataset"
    assert stamp.parent.name == "other_dataset"
    assert metrics["target_dataset"] == {"dataset_path": str(other)}
    assert metrics["training_dataset"] == str(test_data_dir)
