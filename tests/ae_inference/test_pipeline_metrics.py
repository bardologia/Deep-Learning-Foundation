"""Tests for the metric, summary and completion bookkeeping of the shared autoencoder inference pipeline.

Covers the scored checkpoint and epoch written into metrics.json and the completion payload,
and the active-latent-dimension row and collapse warning of the logged summary, using a
pipeline subclass with stubbed metrics and region hooks so no run is loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

from pipelines.autoencoder_common.inference.pipeline  import AeInferencePipeline
from pipelines.profile_autoencoder.inference.pipeline import ProfileAeInferencePipeline
from tools.data.io                                    import FileIO
from tools.runtime.completion                         import CompletionMarker
from tests.conftest                                   import SilentLogger


class _RecordingLogger(SilentLogger):
    """Silent logger keeping every kv_table mapping and warning it is handed.

    Attributes:
        tables   : Mappings passed to kv_table, in call order.
        warnings : Warning messages, in call order.
    """

    def __init__(self):
        """Starts with no tables or warnings."""
        self.tables   = []
        self.warnings = []

    def kv_table(self, data, *args, **kwargs):
        """Records the table mapping."""
        self.tables.append(dict(data))

    def warning(self, message, *args, **kwargs):
        """Records the warning."""
        self.warnings.append(message)


class _Metrics:
    """Metrics object stub returning a fixed mapping.

    Attributes:
        values : Mapping compute returns a copy of.
    """

    def __init__(self, values):
        """Stores the mapping compute returns."""
        self.values = values

    def compute(self):
        """Returns a copy of the stored mapping."""
        return dict(self.values)


class _Pipeline(AeInferencePipeline):
    """Pipeline stub with the family hooks reduced to fixed values.

    Attributes:
        values : Metric mapping the stubbed metrics object returns.
    """

    FAMILY = "Stub AE"

    def __init__(self, config, values):
        """Stores the config and the metric mapping."""
        super().__init__(config)
        self.values = values

    def _metrics_object(self, result, run):
        """Returns the stubbed metrics object."""
        return _Metrics(self.values)

    def _region_metrics(self, run):
        """Returns one region field."""
        return {"split_region": [0, 8, 0, 8]}

    def _metrics_summary(self, metrics):
        """Returns the MSE row."""
        return {"MSE": f"{metrics['mse_mean']:.3g}"}


def _run(epoch=4):
    """Returns a run stub with the split name and the zero-based checkpoint epoch."""
    return SimpleNamespace(split_name="test", checkpoint_meta={"epoch": epoch, "best_epoch": epoch, "best_val_loss": 0.1})


def _values():
    """Returns a metric mapping with a reconstruction error and a fully active latent."""
    return {"mse_mean": 0.5, "embedding_dim": 8, "embedding_active_dim_fraction": 1.0}


def test_metrics_record_the_scored_checkpoint_and_one_based_epoch(tmp_path):
    """metrics.json carries the checkpoint file name and its one-based epoch next to the split and region."""
    pipeline = _Pipeline(SimpleNamespace(checkpoint_name="final_model.pt"), _values())
    meta     = SimpleNamespace(metrics_path=tmp_path / "metrics.json")

    metrics, _ = pipeline._evaluate_metrics(result=None, run=_run(epoch=4), meta=meta, logger=_RecordingLogger())
    written    = FileIO.load_json(meta.metrics_path)

    assert metrics["checkpoint"]       == "final_model.pt"
    assert metrics["checkpoint_epoch"] == 5
    assert written["checkpoint"]       == "final_model.pt"
    assert written["checkpoint_epoch"] == 5
    assert written["split"]            == "test"
    assert written["split_region"]     == [0, 8, 0, 8]


def test_summary_reports_a_fully_active_latent_without_warning(tmp_path):
    """With every latent dimension active the summary states the count and no warning is raised."""
    pipeline = _Pipeline(SimpleNamespace(checkpoint_name="best_model.pt"), _values())
    logger   = _RecordingLogger()

    pipeline._evaluate_metrics(result=None, run=_run(), meta=SimpleNamespace(metrics_path=tmp_path / "metrics.json"), logger=logger)

    assert logger.tables[-1]["Active latent dims"] == "8 of 8 (1.00)"
    assert logger.warnings == []


def test_summary_warns_and_counts_inactive_latent_dimensions(tmp_path):
    """A partly collapsed latent shows the active count in the summary and warns naming the inactive dimensions."""
    values   = {"mse_mean": 0.5, "embedding_dim": 8, "embedding_active_dim_fraction": 0.25}
    pipeline = _Pipeline(SimpleNamespace(checkpoint_name="best_model.pt"), values)
    logger   = _RecordingLogger()

    pipeline._evaluate_metrics(result=None, run=_run(), meta=SimpleNamespace(metrics_path=tmp_path / "metrics.json"), logger=logger)

    assert logger.tables[-1]["Active latent dims"] == "2 of 8 (0.25)"
    assert len(logger.warnings) == 1
    assert "6 of the 8 latent dimensions" in logger.warnings[0]


def test_profile_metrics_record_the_threshold_and_the_sampling_behind_them():
    """The profile metrics carry the activity threshold and the sampling settings plus kept counts next to the scored checkpoint."""
    config   = SimpleNamespace(checkpoint_name="best_model.pt", pixel_subsample=0.25, keep_empty_frac=0.05, seed=3)
    run      = SimpleNamespace(checkpoint_meta={"epoch": 1}, activity_threshold=1e-3, dataset=SimpleNamespace(n_kept_active=120, n_kept_empty=30))
    recorded = ProfileAeInferencePipeline(config)._provenance_metrics(run)

    assert recorded["checkpoint"]         == "best_model.pt"
    assert recorded["checkpoint_epoch"]   == 2
    assert recorded["activity_threshold"] == 1e-3
    assert recorded["sampling"]           == {"pixel_subsample": 0.25, "keep_empty_frac": 0.05, "seed": 3, "n_kept_active": 120, "n_kept_empty": 30}


def test_profile_summary_names_the_active_subset_behind_the_shape_metrics():
    """The profile summary states how many curves the Pearson, relative L2 and peak metrics were computed over."""
    metrics = {"mse_mean": 0.5, "r2": 0.9, "n_active_curves": 1200, "n_curves": 5000, "pearson_mean": 0.8, "relative_l2_mean": 0.3, "peak_location_mae": 0.1}

    summary = ProfileAeInferencePipeline(SimpleNamespace())._metrics_summary(metrics)

    assert summary["Active curves"].startswith("1,200 of 5,000")
    assert list(summary).index("Active curves") < list(summary).index("Pearson")


def test_completion_payload_names_the_scored_weights(tmp_path, monkeypatch):
    """The completion marker records the checkpoint name and epoch alongside the stage and split."""
    pipeline = _Pipeline(SimpleNamespace(checkpoint_name="best_model.pt", save_plots=False), _values())
    meta     = SimpleNamespace(output_dir=tmp_path, metrics_path=tmp_path / "metrics.json", report_path=tmp_path / "report.md")
    run      = _run(epoch=2)

    monkeypatch.setattr(pipeline, "_load_run",           lambda logger: run)
    monkeypatch.setattr(pipeline, "_predict",            lambda run, logger: None)
    monkeypatch.setattr(pipeline, "_persist_embeddings", lambda result, run, meta, logger: None)
    monkeypatch.setattr(pipeline, "_build_report",       lambda meta, run, metrics, figures: meta.report_path)

    pipeline._infer(meta, _RecordingLogger())

    payload = CompletionMarker.payload(tmp_path)

    assert payload["stage"]            == "inference"
    assert payload["split"]            == "test"
    assert payload["checkpoint"]       == "best_model.pt"
    assert payload["checkpoint_epoch"] == 3
