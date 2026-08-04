from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from configuration.architectures.profile_autoencoder import MlpAutoencoderConfig
from configuration.dataset                           import ProfileDatasetConfig, SplitRegions
from configuration.training                          import ProfileAeEntryConfig
from configuration.training.general.runtime          import ResourceConfig, TrainingLoopConfig
from configuration.training.profile_autoencoder      import ProfileAeLossConfig, ProfileAeTrainerConfig
from models.profile_autoencoder                      import PROFILE_AE_CONFIG_REGISTRY
from pipelines.profile_autoencoder.training          import pipeline as profile_pipeline
from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline
from pipelines.shared.config.run_metadata            import TrainingRunMetadata
from tools.data.regions                              import CropRegion


pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("force_cpu")]


PROFILE_LENGTH = 16
EMBEDDING_DIM  = 8


def _trainer_config(cfg):
    tc                         = ProfileAeTrainerConfig(gaussian=None, autoencoder=cfg, ae_loss=ProfileAeLossConfig(curve_kind="mse"))
    tc.training                = TrainingLoopConfig(epochs=1, validation_frequency=1, use_amp=False)
    tc.resources               = ResourceConfig(enabled=False)
    tc.early_stopping.patience = 1000
    tc.warmup.warmup_steps     = 0
    return tc


def _pipeline(cfg):
    pipe                 = profile_pipeline.TrainingPipeline.__new__(profile_pipeline.TrainingPipeline)
    pipe.trainer_config  = _trainer_config(cfg)
    pipe.autoencoder_cfg = cfg
    pipe.ae_model_name   = "mlp_ae"
    pipe.entry           = types.SimpleNamespace(run_name="profile_ae_test", seed=0)
    return pipe


def _build_cfg():
    cfg            = MlpAutoencoderConfig(profile_length=PROFILE_LENGTH, embedding_dim=EMBEDDING_DIM)
    cfg.hidden_dim = 16
    cfg.depth      = 2
    return cfg


def _loader(n=8, batch_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, PROFILE_LENGTH, generator=g)
    return DataLoader(x, batch_size=batch_size)


def _profile_config(tmp_path):
    region = CropRegion(0, 8, 0, 8)
    return ProfileDatasetConfig(preprocessing_run_directory=tmp_path, split_regions=SplitRegions(train=region, val=region, test=region))


def test_build_model_sets_profile_length():
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    model = pipe._build_model(PROFILE_LENGTH, cfg)

    assert isinstance(model, torch.nn.Module)
    assert cfg.profile_length == PROFILE_LENGTH


def test_orchestration_produces_checkpoint_and_metadata(tmp_path):
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    run_meta = TrainingRunMetadata(pipe.trainer_config, "profile_ae", tmp_path, "profile_ae_test")
    model    = pipe._build_model(PROFILE_LENGTH, cfg)
    x_axis   = np.linspace(0.0, 1.0, PROFILE_LENGTH, dtype=np.float32)

    pipe._save_metadata(run_meta, _profile_config(tmp_path), PROFILE_LENGTH)

    loader            = _loader()
    results, run_dir  = pipe._train(run_meta, run_meta.logger, model, x_axis, loader, loader, loader)

    train_losses, val_losses, best = results

    assert np.isfinite(best)
    assert len(train_losses) == 1

    run_dir = Path(run_dir)
    assert (run_dir / "best_model.pt").is_file()
    assert (run_meta.metadata_directory / "profile_autoencoder_config.json").is_file()
    assert (run_meta.metadata_directory / "run_summary.json").is_file()
    assert (run_meta.docs_directory / "trainer_config.json").is_file()


def test_saved_checkpoint_is_loadable(tmp_path):
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    run_meta = TrainingRunMetadata(pipe.trainer_config, "profile_ae", tmp_path, "profile_ae_test")
    model    = pipe._build_model(PROFILE_LENGTH, cfg)
    x_axis   = np.linspace(0.0, 1.0, PROFILE_LENGTH, dtype=np.float32)

    loader            = _loader()
    _results, run_dir = pipe._train(run_meta, run_meta.logger, model, x_axis, loader, loader, loader)

    ckpt = torch.load(Path(run_dir) / "best_model.pt", map_location="cpu", weights_only=False)

    fresh = pipe._build_model(PROFILE_LENGTH, cfg)
    fresh.load_state_dict(ckpt["params"])


def test_autoencoder_config_follows_ae_model_name():
    entry = ProfileAeEntryConfig(ae_model_name="gru_ae", model_overrides={"embedding_dim": 48})
    cfg   = TrainingPipeline._autoencoder_config(TrainingPipeline.__new__(TrainingPipeline), entry)

    assert type(cfg) is type(PROFILE_AE_CONFIG_REGISTRY["gru_ae"]())
    assert cfg.embedding_dim == 48

    for name in PROFILE_AE_CONFIG_REGISTRY:
        cfg = TrainingPipeline._autoencoder_config(TrainingPipeline.__new__(TrainingPipeline), ProfileAeEntryConfig(ae_model_name=name))
        assert type(cfg) is type(PROFILE_AE_CONFIG_REGISTRY[name]())


class _CurveDataset(torch.utils.data.Dataset):
    def __init__(self, n=8):
        g = torch.Generator().manual_seed(0)
        self.curves    = torch.randn(n, PROFILE_LENGTH, generator=g)
        self.augmenter = None

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        return self.curves[idx]


def test_overfit_check_gate_trains_and_reports(tmp_path):
    import json

    from configuration.training import OverfitCheckConfig

    cfg         = _build_cfg()
    cfg.dropout = 0.25
    pipe        = _pipeline(cfg)

    pipe.entry.overfit_check = OverfitCheckConfig(enabled=True, n_examples=2, max_steps=4, steps_per_epoch=2, pass_loss_ratio=1.0)

    run_meta = TrainingRunMetadata(pipe.trainer_config, "profile_ae", tmp_path, "profile_ae_overfit_check")
    x_axis   = np.linspace(0.0, 1.0, PROFILE_LENGTH, dtype=np.float32)

    pipe._run_overfit_check(run_meta, run_meta.logger, PROFILE_LENGTH, x_axis, {"train": _CurveDataset()})

    report_path = run_meta.run_directory / "meta" / "overfit_report.json"
    report      = json.loads(report_path.read_text())

    assert report["passed"] is True
    assert report["epochs_run"] <= 2
    assert report["sanitized_overrides"]["model.dropout"] == 0.0
    assert cfg.dropout == 0.25

    assert not (run_meta.run_directory / "overfit_check" / "best_model.pt").exists()
    assert not (run_meta.run_directory / "overfit_check" / "last.pt").exists()
