"""Tests for the image autoencoder training pipeline.

Covers model construction from the entry's architecture name, the artifacts a training run
leaves behind and their reloadability, the refusal of input-noise augmentation for an
autoencoder, and that the run metadata is closed when data preparation fails.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.architectures.image_autoencoder import Conv2dImageAutoencoderConfig
from configuration.dataset                         import AugmentationConfig
from configuration.training                        import ImageAeEntryConfig
from configuration.resources                       import ResourceConfig
from configuration.training.general.runtime        import TrainingLoopConfig
from configuration.training.image_autoencoder      import ImageAeLossConfig, ImageAeTrainerConfig
from models.image_autoencoder                      import IMAGE_AE_CONFIG_REGISTRY
from pipelines.autoencoder_common                  import training as autoencoder_training
from pipelines.image_autoencoder.training          import pipeline as image_pipeline
from pipelines.shared.config.run_metadata          import TrainingRunMetadata
from tests.conftest                                import SilentLogger


pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("force_cpu")]


IN_CHANNELS   = 2
EMBEDDING_DIM = 8
PATCH         = 16


def _trainer_config(cfg):
    """Returns a one-epoch CPU trainer config with early stopping and warmup disabled."""
    tc                         = ImageAeTrainerConfig(gaussian=None, image_autoencoder=cfg, ae_loss=ImageAeLossConfig(recon_kind="mse"))
    tc.training                = TrainingLoopConfig(epochs=1, validation_frequency=1, use_amp=False)
    tc.resources               = ResourceConfig(enabled=False)
    tc.early_stopping.patience = 1000
    tc.warmup.warmup_fraction  = 0.0
    return tc


def _pipeline(cfg):
    """Returns a training pipeline with the trainer config and architecture fields set, bypassing its constructor."""
    pipe                 = image_pipeline.TrainingPipeline.__new__(image_pipeline.TrainingPipeline)
    pipe.trainer_config  = _trainer_config(cfg)
    pipe.autoencoder_cfg = cfg
    pipe.ae_model_name   = "conv2d_ae"
    pipe.entry           = types.SimpleNamespace(run_name="image_ae_test", seed=0, paths=types.SimpleNamespace(dataset_path=".", parameters_path=None, fingerprint_dataset=False))
    pipe.dataset_config  = types.SimpleNamespace(batch_size=4)
    return pipe


def _build_cfg():
    """Returns a tiny conv2d image autoencoder config: 8 base channels at depth one."""
    cfg               = Conv2dImageAutoencoderConfig(in_channels=IN_CHANNELS, embedding_dim=EMBEDDING_DIM)
    cfg.base_channels = 8
    cfg.depth         = 1
    return cfg


def _loader(n=8, batch_size=4, seed=0):
    """Returns a loader over random patches of shape (n, channels, 16, 16)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_CHANNELS, PATCH, PATCH, generator=g)
    return DataLoader(TensorDataset(x), batch_size=batch_size)


def test_build_model_sets_in_channels():
    """Building the model installs the dataset channel count on the architecture config."""
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    model = pipe._build_model(IN_CHANNELS, cfg)

    assert isinstance(model, torch.nn.Module)
    assert cfg.in_channels == IN_CHANNELS


def test_orchestration_produces_checkpoint_and_metadata(tmp_path):
    """A training run writes the best checkpoint plus the architecture, summary and trainer config metadata."""
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    run_meta = TrainingRunMetadata(pipe.trainer_config, "image_ae", tmp_path, "image_ae_test")
    model    = pipe._build_model(IN_CHANNELS, cfg)
    z_axis   = np.arange(PATCH, dtype=np.float32)

    pipe._save_metadata(run_meta, IN_CHANNELS, PATCH)

    loader            = _loader()
    results, run_dir  = pipe._train(run_meta, run_meta.logger, model, z_axis, loader, loader, loader)

    train_losses, val_losses, best = results

    assert np.isfinite(best)
    assert len(train_losses) == 1

    run_dir = Path(run_dir)
    assert (run_dir / "best_model.pt").is_file()
    assert (run_meta.metadata_directory / "image_autoencoder_config.json").is_file()
    assert (run_meta.metadata_directory / "run_summary.json").is_file()
    assert (run_meta.docs_directory / "trainer_config.json").is_file()


def test_saved_checkpoint_is_loadable(tmp_path):
    """The saved parameters load into a freshly built model of the same configuration."""
    cfg  = _build_cfg()
    pipe = _pipeline(cfg)

    run_meta = TrainingRunMetadata(pipe.trainer_config, "image_ae", tmp_path, "image_ae_test")
    model    = pipe._build_model(IN_CHANNELS, cfg)
    z_axis   = np.arange(PATCH, dtype=np.float32)

    loader            = _loader()
    _results, run_dir = pipe._train(run_meta, run_meta.logger, model, z_axis, loader, loader, loader)

    ckpt = torch.load(Path(run_dir) / "best_model.pt", map_location="cpu", weights_only=False)

    fresh = pipe._build_model(IN_CHANNELS, cfg)
    fresh.load_state_dict(ckpt["params"])


def test_autoencoder_config_follows_ae_model_name():
    """Every registered architecture name resolves to its own config class, with model overrides applied."""
    entry = ImageAeEntryConfig(ae_model_name="conv2d_ae", model_overrides={"embedding_dim": 48})
    cfg   = image_pipeline.TrainingPipeline._autoencoder_config(image_pipeline.TrainingPipeline.__new__(image_pipeline.TrainingPipeline), entry)

    assert type(cfg) is type(IMAGE_AE_CONFIG_REGISTRY["conv2d_ae"]())
    assert cfg.embedding_dim == 48

    for name in IMAGE_AE_CONFIG_REGISTRY:
        cfg = image_pipeline.TrainingPipeline._autoencoder_config(image_pipeline.TrainingPipeline.__new__(image_pipeline.TrainingPipeline), ImageAeEntryConfig(ae_model_name=name))
        assert type(cfg) is type(IMAGE_AE_CONFIG_REGISTRY[name]())


def _dataset_config_pipeline(p_noise: float):
    """Returns a pipeline whose dataset factory reports an augmentation config with the given noise probability."""
    dataset_config = types.SimpleNamespace(split_regions=None, augmentation=AugmentationConfig(p_noise=p_noise))

    pipe               = image_pipeline.TrainingPipeline.__new__(image_pipeline.TrainingPipeline)
    pipe.factory       = types.SimpleNamespace(training_dataset_config=lambda: dataset_config)
    pipe.split_regions = "split-regions"

    return pipe


def test_image_ae_refuses_input_noise_augmentation():
    """Input noise augmentation raises, since the autoencoder reconstructs its own input."""
    pipe = _dataset_config_pipeline(p_noise=0.5)

    with pytest.raises(ValueError, match="p_noise"):
        pipe._build_dataset_config()


def test_image_ae_accepts_disabled_input_noise():
    """With input noise disabled the dataset config is built and keeps the pipeline split regions."""
    pipe   = _dataset_config_pipeline(p_noise=0.0)
    config = pipe._build_dataset_config()

    assert config.split_regions == "split-regions"


class _RecordingRunMetadata:
    """Run metadata stub recording every instance created and whether it was closed."""

    instances = []

    def __init__(self, trainer_config, run_label, logdir, run_name):
        """Derives the run directory and registers the instance on the class."""
        self.run_directory = Path(logdir) / str(run_name)
        self.logger        = SilentLogger()
        self.closed        = False

        type(self).instances.append(self)

    def save_dataset_fingerprint(self, dataset_path, parameters_path, enabled=True):
        """Accepts the provenance call without touching the filesystem."""
        return None

    def __enter__(self):
        """Returns the stub itself as the context manager value."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes the stub on context exit, however the block ended."""
        self.close()

    def close(self):
        """Marks the stub as closed."""
        self.closed = True


def test_run_closes_the_run_metadata_when_data_preparation_fails(tmp_path, monkeypatch):
    """A failure inside data preparation still closes the single run metadata that was opened."""
    _RecordingRunMetadata.instances = []

    pipe                          = _pipeline(_build_cfg())
    pipe.trainer_config.io.logdir = str(tmp_path)

    def _failing_prepare(run_directory, logger):
        raise RuntimeError("dataset shape mismatch")

    pipe._prepare_data = _failing_prepare
    monkeypatch.setattr(autoencoder_training, "TrainingRunMetadata", _RecordingRunMetadata)

    with pytest.raises(RuntimeError, match="dataset shape mismatch"):
        pipe.run()

    assert len(_RecordingRunMetadata.instances) == 1
    assert _RecordingRunMetadata.instances[0].closed
