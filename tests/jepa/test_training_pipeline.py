from __future__ import annotations

import numpy as np
import pytest

from pipelines.jepa.training.pipeline import TrainingPipeline


def test_resolve_ae_run_returns_none_when_no_run():
    assert TrainingPipeline._resolve_ae_run("/anywhere", None, "profile") is None
    assert TrainingPipeline._resolve_ae_run("/anywhere", "", "profile")   is None


def test_resolve_ae_run_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainingPipeline._resolve_ae_run(tmp_path, "does_not_exist", "profile")


def test_resolve_ae_run_returns_existing_directory(tmp_path):
    run_dir = tmp_path / "ae_run_01"
    run_dir.mkdir()

    resolved = TrainingPipeline._resolve_ae_run(tmp_path, "ae_run_01", "profile")

    assert resolved == run_dir


def test_validate_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainingPipeline.validate_checkpoint(tmp_path / "best_model.pt", "profile")


def test_validate_checkpoint_passes_when_present(tmp_path):
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"x")

    TrainingPipeline.validate_checkpoint(ckpt, "profile")


class _GateStop(Exception):
    pass


def test_overfit_gate_backbone_config_carries_entry_head(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from configuration.training                  import JepaEntryConfig, OverfitCheckConfig
    from pipelines.shared.training.overfit_check import OverfitCheck

    captured = {}

    def fake_sanitized_trainer_config(self, trainer_config):
        return SimpleNamespace(param_loss=SimpleNamespace(use_active_normalization=True))

    def fake_build_module(self, datasets, x_len, x_axis, logger, backbone_config=None):
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
        pipe._run_overfit_check(run_meta, logger, {"train": None}, 128, None)

    assert captured["config"].head == "set_pred"


def _axis_pipeline(tmp_path):
    import torch
    from types import SimpleNamespace

    from configuration.architectures import MlpAutoencoderConfig
    from models.profile_autoencoder  import get_profile_autoencoder

    config          = MlpAutoencoderConfig(profile_length=8, embedding_dim=4, hidden_dim=8, depth=1)
    autoencoder, _  = get_profile_autoencoder("mlp_ae", config)
    checkpoint_path = tmp_path / "best_model.pt"

    torch.save({"params": autoencoder.state_dict(), "x_axis": np.linspace(0.0, 1.0, 8, dtype=np.float32)}, checkpoint_path)

    pipe                 = TrainingPipeline.__new__(TrainingPipeline)
    pipe.ae_model_name   = "mlp_ae"
    pipe.autoencoder_cfg = config
    pipe.trainer_config  = SimpleNamespace(profile_autoencoder_checkpoint=str(checkpoint_path))
    return pipe


def test_load_profile_autoencoder_rejects_axis_range_mismatch(tmp_path):
    pipe = _axis_pipeline(tmp_path)

    with pytest.raises(ValueError, match="elevation axis"):
        pipe._load_profile_autoencoder(np.linspace(0.0, 2.0, 8, dtype=np.float32))


def test_load_profile_autoencoder_rejects_axis_length_mismatch(tmp_path):
    pipe = _axis_pipeline(tmp_path)

    with pytest.raises(ValueError, match="elevation axis"):
        pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 16, dtype=np.float32))


def test_load_profile_autoencoder_accepts_matching_axis(tmp_path):
    pipe   = _axis_pipeline(tmp_path)
    loaded = pipe._load_profile_autoencoder(np.linspace(0.0, 1.0, 8, dtype=np.float32))

    assert loaded is not None
