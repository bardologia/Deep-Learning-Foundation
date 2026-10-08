"""Tests for the JEPA training pipeline's validation gates and run bookkeeping.

Covers autoencoder run-directory resolution, the head restriction under profile coupling,
checkpoint presence and run-completion checks, the head passed into the overfit gate, z-axis
agreement and the checkpoint record kept when loading a pretrained profile autoencoder, the
persistence of that record, and closing the run metadata when dataset preparation fails."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from pipelines.jepa.training          import pipeline as jepa_pipeline
from pipelines.jepa.training.pipeline import TrainingPipeline
from tools.data.io                    import FileIO
from tools.runtime.completion         import CompletionMarker
from tests.conftest                   import SilentLogger


class _RecordingLogger(SilentLogger):
    """Silent logger that keeps every kv_table mapping and warning it is handed.

    Attributes:
        tables   : Mappings passed to kv_table, in call order.
        warnings : Warning messages, in call order.
    """

    def __init__(self):
        """Starts with no recorded tables or warnings."""
        self.tables   = []
        self.warnings = []

    def kv_table(self, data, *args, **kwargs):
        """Records the table mapping."""
        self.tables.append(dict(data))

    def warning(self, message, *args, **kwargs):
        """Records the warning message."""
        self.warnings.append(message)


def test_resolve_ae_run_returns_none_when_no_run():
    """An empty or absent autoencoder run name resolves to None."""
    assert TrainingPipeline._resolve_ae_run("/anywhere", None, "profile") is None
    assert TrainingPipeline._resolve_ae_run("/anywhere", "", "profile")   is None


def test_resolve_ae_run_raises_for_missing_directory(tmp_path):
    """A named autoencoder run that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TrainingPipeline._resolve_ae_run(tmp_path, "does_not_exist", "profile")


def test_resolve_ae_run_returns_existing_directory(tmp_path):
    """An existing autoencoder run name resolves to its directory."""
    run_dir = tmp_path / "ae_run_01"
    run_dir.mkdir()

    resolved = TrainingPipeline._resolve_ae_run(tmp_path, "ae_run_01", "profile")

    assert resolved == run_dir


def test_a_profile_coupled_jepa_rejects_parameter_structured_heads():
    """Profile coupling rejects every parameter-structured head."""
    for head in ("multihead", "per_gaussian", "set_pred"):
        with pytest.raises(ValueError, match="only the 'conv' head"):
            TrainingPipeline._validate_head(head, profile_coupled=True)


def test_head_validation_allows_conv_and_the_param_regime():
    """The conv head passes under profile coupling and set_pred passes in the parameter regime."""
    TrainingPipeline._validate_head("conv", profile_coupled=True)
    TrainingPipeline._validate_head("set_pred", profile_coupled=False)


def test_validate_checkpoint_raises_when_missing(tmp_path):
    """A missing autoencoder checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TrainingPipeline.validate_checkpoint(tmp_path / "best_model.pt", "profile")


def test_validate_checkpoint_rejects_an_unfinished_run(tmp_path):
    """A checkpoint whose run directory carries no completion marker is refused as partial."""
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"x")

    with pytest.raises(RuntimeError, match="partial checkpoint"):
        TrainingPipeline.validate_checkpoint(ckpt, "profile")


def test_validate_checkpoint_passes_when_present_and_complete(tmp_path):
    """An existing checkpoint of a run that stamped its completion marker passes validation."""
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"x")
    CompletionMarker.stamp(tmp_path, {"stage": "training"})

    TrainingPipeline.validate_checkpoint(ckpt, "profile")


class _GateStop(Exception):
    """Sentinel exception raised to abort the overfit gate once its arguments are captured."""
    pass


def test_overfit_gate_backbone_config_carries_entry_head(tmp_path, monkeypatch):
    """The overfit gate builds its module with the head named in the entry config."""
    from types import SimpleNamespace

    from configuration.training                  import JepaEntryConfig, OverfitCheckConfig
    from pipelines.shared.training.overfit_check import OverfitCheck

    captured = {}

    def fake_sanitized_trainer_config(self, trainer_config):
        return SimpleNamespace(param_loss=SimpleNamespace(use_active_normalization=True))

    def fake_build_module(self, datasets, z_len, z_axis, logger, backbone_config=None):
        captured["config"] = backbone_config
        raise _GateStop()

    monkeypatch.setattr(OverfitCheck, "sanitized_trainer_config", fake_sanitized_trainer_config)
    monkeypatch.setattr(TrainingPipeline, "_build_module", fake_build_module)

    pipe                 = TrainingPipeline.__new__(TrainingPipeline)
    pipe.entry           = JepaEntryConfig(backbone_head="set_pred", overfit_check=OverfitCheckConfig(enabled=True))
    pipe.backbone_name   = pipe.entry.backbone_name
    pipe.trainer_config  = SimpleNamespace()
    pipe.autoencoder_cfg = None

    run_meta = SimpleNamespace(run_directory=tmp_path)
    logger   = SimpleNamespace(section=lambda *a, **k: None, subsection=lambda *a, **k: None)

    with pytest.raises(_GateStop):
        pipe._run_overfit_check(run_meta, logger, {"train": None}, 128, None, None)

    assert captured["config"].head == "set_pred"


def _axis_pipeline(tmp_path):
    """Returns a pipeline stub coupled to a finished MLP autoencoder run on a unit z axis.

    The run directory carries the checkpoint with its epoch bookkeeping, the
    run summary naming the commit, and the completion marker of a nine-epoch
    run stopped early.
    """
    import torch
    from types import SimpleNamespace

    from configuration.architectures import MlpAutoencoderConfig
    from models.profile_autoencoder  import get_profile_autoencoder

    config          = MlpAutoencoderConfig(profile_length=8, embedding_dim=4, hidden_dim=8, depth=1)
    autoencoder, _  = get_profile_autoencoder("mlp_ae", config)
    checkpoint_path = tmp_path / "best_model.pt"

    torch.save({"params": autoencoder.state_dict(), "z_axis": np.linspace(0.0, 1.0, 8, dtype=np.float32), "epoch": 6, "best_epoch": 6, "best_val_loss": 0.125}, checkpoint_path)
    FileIO.save_json({"model_name": "profile_ae", "git_commit": "abc1234"}, tmp_path / "meta" / "run_summary.json")
    CompletionMarker.stamp(tmp_path, {"stage": "training", "epochs_completed": 9, "epochs_total": 20, "early_stopped": True, "best_val_loss": 0.125, "best_epoch": 7})

    pipe                 = TrainingPipeline.__new__(TrainingPipeline)
    pipe.ae_model_name   = "mlp_ae"
    pipe.autoencoder_cfg = config
    pipe.trainer_config  = SimpleNamespace(profile_autoencoder_checkpoint=str(checkpoint_path))

    pipe.profile_ae_state     = None
    pipe.coupled_autoencoders = {}

    return pipe


def test_load_profile_autoencoder_rejects_axis_range_mismatch(tmp_path):
    """An z axis spanning a different range than the checkpoint is rejected."""
    pipe = _axis_pipeline(tmp_path)

    with pytest.raises(ValueError, match="z axis"):
        pipe._load_profile_autoencoder(np.linspace(0.0, 2.0, 8, dtype=np.float32), SilentLogger())


def test_load_profile_autoencoder_rejects_axis_length_mismatch(tmp_path):
    """An z axis of a different length than the checkpoint is rejected."""
    pipe = _axis_pipeline(tmp_path)

    with pytest.raises(ValueError, match="z axis"):
        pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 16, dtype=np.float32), SilentLogger())


def test_load_profile_autoencoder_accepts_matching_axis(tmp_path):
    """A matching z axis loads the pretrained autoencoder."""
    pipe   = _axis_pipeline(tmp_path)
    loaded = pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 8, dtype=np.float32), SilentLogger())

    assert loaded is not None


def test_load_profile_autoencoder_refuses_an_unfinished_run(tmp_path):
    """Without the run's completion marker the checkpoint is refused before any weight is read."""
    pipe = _axis_pipeline(tmp_path)
    CompletionMarker.clear(tmp_path)

    with pytest.raises(RuntimeError, match="partial checkpoint"):
        pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 8, dtype=np.float32), SilentLogger())


def test_load_profile_autoencoder_records_and_logs_the_imported_checkpoint(tmp_path):
    """The first load records one-based epochs, the best loss, the training length and the commit, and logs them once."""
    pipe   = _axis_pipeline(tmp_path)
    logger = _RecordingLogger()

    pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 8, dtype=np.float32), logger)
    pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 8, dtype=np.float32), logger)

    record = pipe.coupled_autoencoders["profile"]

    assert record["run_directory"]     == str(tmp_path)
    assert record["checkpoint"]        == str(tmp_path / "best_model.pt")
    assert record["epoch"]             == 7
    assert record["best_epoch"]        == 7
    assert record["best_val_loss"]     == 0.125
    assert record["epochs_completed"]  == 9
    assert record["epochs_total"]      == 20
    assert record["early_stopped"]     is True
    assert record["marker_backfilled"] is False
    assert record["git_commit"]        == "abc1234"

    assert len(logger.tables) == 1
    assert logger.tables[0]["Checkpoint epoch"] == 7
    assert logger.tables[0]["Git commit"]       == "abc1234"
    assert "early stopped" in logger.tables[0]["Epochs completed"]
    assert logger.warnings == []


def test_load_profile_autoencoder_records_a_backfilled_marker_without_epoch_counts(tmp_path):
    """A run whose completion marker was backfilled loads, records no epoch counts and names the backfilled marker in its table."""
    pipe   = _axis_pipeline(tmp_path)
    logger = _RecordingLogger()

    CompletionMarker.clear(tmp_path)
    CompletionMarker.stamp(tmp_path, {"stage": "training", "backfilled": True})

    pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 8, dtype=np.float32), logger)

    record = pipe.coupled_autoencoders["profile"]

    assert record["marker_backfilled"] is True
    assert record["epochs_completed"]  is None
    assert record["epochs_total"]      is None
    assert record["early_stopped"]     is None
    assert logger.tables[0]["Epochs completed"] == "not recorded (backfilled completion marker)"


def test_save_metadata_persists_the_coupled_autoencoder_record(tmp_path):
    """The run's meta directory receives the imported-checkpoint record next to the other metadata files."""
    pipe = _axis_pipeline(tmp_path)
    pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 8, dtype=np.float32), SilentLogger())

    pipe.trainer_config        = types.SimpleNamespace(gaussian=types.SimpleNamespace(n_default_gaussians=2))
    pipe.image_ae_cfg          = None
    pipe.backbone_name         = "resunet"
    pipe.backbone_in_channels  = 2
    pipe.backbone_out_channels = 4
    pipe.entry                 = types.SimpleNamespace(seed=0)

    meta_directory = tmp_path / "jepa_run" / "meta"
    meta_directory.mkdir(parents=True)

    run_meta = types.SimpleNamespace(
        metadata_directory  = meta_directory,
        save_trainer_config = lambda: None,
        save_model_config   = lambda *a, **k: None,
        save_run_summary    = lambda *a, **k: None,
    )

    pipe._save_metadata(run_meta, backbone_cfg=None, z_len=8)

    saved = FileIO.load_json(meta_directory / TrainingPipeline.COUPLED_RECORD_FILENAME)

    assert saved["profile"]["best_epoch"] == 7
    assert saved["profile"]["git_commit"] == "abc1234"


class _RecordingRunMetadata:
    """Run-metadata stand-in that records every instance and whether it was closed.

    Attributes:
        instances          : Class-level list of every instance constructed.
        run_directory      : Directory the run would write to.
        metadata_directory : The meta subdirectory of the run directory.
        closed             : Whether close was called on this instance.
    """
    instances = []

    def __init__(self, trainer_config, run_label, logdir, run_name):
        """Builds the run paths from the logdir and run name and registers the instance."""
        self.run_directory      = Path(logdir) / str(run_name)
        self.metadata_directory = self.run_directory / "meta"
        self.logger             = SilentLogger()
        self.closed             = False

        type(self).instances.append(self)

    def save_dataset_fingerprint(self, dataset_path, parameters_path, enabled=True):
        """Accepts the provenance call without touching the filesystem."""
        return None

    def __enter__(self):
        """Returns the instance itself as the context value."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes the instance on context exit."""
        self.close()

    def close(self):
        """Marks the instance closed."""
        self.closed = True


def test_run_closes_the_run_metadata_when_dataset_preparation_fails(tmp_path, monkeypatch):
    """A failure during dataset preparation still closes the run metadata before propagating."""
    _RecordingRunMetadata.instances = []

    pipeline                = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.backbone_name  = "resunet"
    pipeline.entry          = types.SimpleNamespace(run_name="jepa_test", seed=0, backbone_head="conv", paths=types.SimpleNamespace(dataset_path=str(tmp_path), parameters_path=None, fingerprint_dataset=False))
    pipeline.trainer_config = types.SimpleNamespace(io=types.SimpleNamespace(logdir=str(tmp_path)))

    def _failing_prepare(run_directory, logger, seed):
        raise RuntimeError("dataset shape mismatch")

    pipeline._prepare_datasets = _failing_prepare
    monkeypatch.setattr(jepa_pipeline, "TrainingRunMetadata", _RecordingRunMetadata)

    with pytest.raises(RuntimeError, match="dataset shape mismatch"):
        pipeline.run()

    assert len(_RecordingRunMetadata.instances) == 1
    assert _RecordingRunMetadata.instances[0].closed
