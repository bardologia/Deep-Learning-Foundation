from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.architectures.image_autoencoder import Conv2dImageAutoencoderConfig
from configuration.training                        import ImageAeEntryConfig
from configuration.training.general.runtime        import ResourceConfig, TrainingLoopConfig
from configuration.training.image_autoencoder      import ImageAeLossConfig, ImageAeTrainerConfig
from models.image_autoencoder                      import IMAGE_AE_CONFIG_REGISTRY
from pipelines.image_autoencoder.training          import pipeline as image_pipeline
from pipelines.shared.config.run_metadata          import TrainingRunMetadata


pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("force_cpu")]


IN_CHANNELS   = 2
EMBEDDING_DIM = 8
PATCH         = 16


def _trainer_config(cfg):
    tc                         = ImageAeTrainerConfig(gaussian=None, image_autoencoder=cfg, ae_loss=ImageAeLossConfig(recon_kind="mse"))
    tc.training                = TrainingLoopConfig(epochs=1, validation_frequency=1, use_amp=False)
    tc.resources               = ResourceConfig(enabled=False)
    tc.early_stopping.patience = 1000
    tc.warmup.warmup_steps     = 0
    return tc


def _pipeline(cfg):
    pipe                 = image_pipeline.TrainingPipeline.__new__(image_pipeline.TrainingPipeline)
    pipe.trainer_config  = _trainer_config(cfg)
    pipe.autoencoder_cfg = cfg
    pipe.ae_model_name   = "conv2d_ae"
    pipe.entry           = types.SimpleNamespace(run_name="image_ae_test", seed=0)
    return pipe


def _build_cfg():
    cfg               = Conv2dImageAutoencoderConfig(in_channels=IN_CHANNELS, embedding_dim=EMBEDDING_DIM)
    cfg.base_channels = 8
    cfg.depth         = 1
    return cfg


def _loader(n=8, batch_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_CHANNELS, PATCH, PATCH, generator=g)
    return DataLoader(TensorDataset(x), batch_size=batch_size)


def test_build_model_sets_in_channels():
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    model = pipe._build_model(IN_CHANNELS, cfg)

    assert isinstance(model, torch.nn.Module)
    assert cfg.in_channels == IN_CHANNELS


def test_orchestration_produces_checkpoint_and_metadata(tmp_path):
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    run_meta = TrainingRunMetadata(pipe.trainer_config, "image_ae", tmp_path, "image_ae_test")
    model    = pipe._build_model(IN_CHANNELS, cfg)
    x_axis   = np.arange(PATCH, dtype=np.float32)

    pipe._save_metadata(run_meta, IN_CHANNELS, PATCH)

    loader            = _loader()
    results, run_dir  = pipe._train(run_meta, run_meta.logger, model, x_axis, loader, loader, loader)

    train_losses, val_losses, best = results

    assert np.isfinite(best)
    assert len(train_losses) == 1

    run_dir = Path(run_dir)
    assert (run_dir / "best_model.pt").is_file()
    assert (run_meta.metadata_directory / "image_autoencoder_config.json").is_file()
    assert (run_meta.metadata_directory / "run_summary.json").is_file()
    assert (run_meta.docs_directory / "trainer_config.json").is_file()


def test_saved_checkpoint_is_loadable(tmp_path):
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    run_meta = TrainingRunMetadata(pipe.trainer_config, "image_ae", tmp_path, "image_ae_test")
    model    = pipe._build_model(IN_CHANNELS, cfg)
    x_axis   = np.arange(PATCH, dtype=np.float32)

    loader            = _loader()
    _results, run_dir = pipe._train(run_meta, run_meta.logger, model, x_axis, loader, loader, loader)

    ckpt = torch.load(Path(run_dir) / "best_model.pt", map_location="cpu", weights_only=False)

    fresh = pipe._build_model(IN_CHANNELS, cfg)
    fresh.load_state_dict(ckpt["params"])


def test_autoencoder_config_follows_ae_model_name():
    entry = ImageAeEntryConfig(ae_model_name="conv2d_ae", model_overrides={"embedding_dim": 48})
    cfg   = image_pipeline.TrainingPipeline._autoencoder_config(image_pipeline.TrainingPipeline.__new__(image_pipeline.TrainingPipeline), entry)

    assert type(cfg) is type(IMAGE_AE_CONFIG_REGISTRY["conv2d_ae"]())
    assert cfg.embedding_dim == 48

    for name in IMAGE_AE_CONFIG_REGISTRY:
        cfg = image_pipeline.TrainingPipeline._autoencoder_config(image_pipeline.TrainingPipeline.__new__(image_pipeline.TrainingPipeline), ImageAeEntryConfig(ae_model_name=name))
        assert type(cfg) is type(IMAGE_AE_CONFIG_REGISTRY[name]())
