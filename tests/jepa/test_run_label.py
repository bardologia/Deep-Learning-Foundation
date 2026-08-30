"""Tests that JEPA run labels name the coupled autoencoders, their modes and the active loss.

Covers the profile-coupled label built from the embedding loss, the image-only label that keeps
the parameter loss, the label naming both autoencoders with their coupling modes, and the
refusals when no autoencoder is configured or its persisted config is missing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from configuration.architectures                import Conv2dImageAutoencoderConfig, GruAutoencoderConfig
from configuration.training                     import JepaEntryConfig
from pipelines.jepa.training                    import pipeline as jepa_pipeline
from pipelines.jepa.training.pipeline           import SingleTrainRunner
from pipelines.shared.config.config_persistence import ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO
from pipelines.shared.training.run_naming       import JepaNamingSpec


def _entry(tmp_path, **fields) -> JepaEntryConfig:
    """Returns a JEPA entry config pointed at temporary autoencoder logdirs with `fields` applied."""
    config = JepaEntryConfig()

    config.profile_autoencoder_logdir = tmp_path / "profile_runs"
    config.image_autoencoder_logdir   = tmp_path / "image_runs"

    for name, value in fields.items():
        setattr(config, name, value)

    return config


def _save_profile_ae(tmp_path) -> None:
    """Persists a default GRU profile-autoencoder config into the gru_run directory."""
    ProfileAutoencoderConfigIO.save(GruAutoencoderConfig(), "gru_ae", tmp_path / "profile_runs" / "gru_run" / "meta")


def _save_image_ae(tmp_path) -> None:
    """Persists a default conv2d image-autoencoder config into the conv_run directory."""
    ImageAutoencoderConfigIO.save(Conv2dImageAutoencoderConfig(), "conv2d_ae", tmp_path / "image_runs" / "conv_run" / "meta")


def _runner(monkeypatch, config: JepaEntryConfig) -> SingleTrainRunner:
    """Returns a SingleTrainRunner with the config factory stubbed to five Gaussians."""
    monkeypatch.setattr(jepa_pipeline, "ConfigFactory", lambda entry: SimpleNamespace(gaussian_config=lambda: SimpleNamespace(n_default_gaussians=5)))
    return SingleTrainRunner(config)


def test_label_profile_coupled_carries_target_ae_mode_and_embedding_loss(monkeypatch, tmp_path):
    """A profile-coupled run is labelled with the target provider, the pae tag and the embedding loss terms."""
    _save_profile_ae(tmp_path)
    runner = _runner(monkeypatch, _entry(tmp_path, profile_autoencoder_run="gru_run"))

    assert runner.label == "resunet-conv-stopgrad-K_5-hv-none-pae_gru.frozen-emb_mse_1-curve_mse_1"


def test_label_image_only_carries_the_param_loss_and_the_image_ae(monkeypatch, tmp_path):
    """An image-only run keeps the parameter-loss label and gains the iae tag."""
    _save_image_ae(tmp_path)
    runner = _runner(monkeypatch, _entry(tmp_path, image_autoencoder_run="conv_run"))

    assert runner.label == "resunet-conv-sorted_gt-K_5-hv-A-iae_conv2d.frozen-param_l1_1"


def test_label_names_both_autoencoders_with_their_modes(monkeypatch, tmp_path):
    """With both autoencoders coupled the label carries both tags and both coupling modes."""
    _save_profile_ae(tmp_path)
    _save_image_ae(tmp_path)
    fields = dict(profile_autoencoder_run="gru_run", profile_autoencoder_mode="finetune", target_provider="live", image_autoencoder_run="conv_run", image_autoencoder_mode="finetune")
    runner = _runner(monkeypatch, _entry(tmp_path, **fields))

    assert runner.label == "resunet-conv-live-K_5-hv-none-pae_gru.finetune-iae_conv2d.finetune-emb_mse_1-curve_mse_1"


def test_resolve_rejects_a_config_without_any_autoencoder(tmp_path):
    """Resolving a naming spec with neither autoencoder configured raises ValueError."""
    with pytest.raises(ValueError):
        JepaNamingSpec.resolve(_entry(tmp_path))


def test_resolve_fails_loudly_when_the_ae_run_lacks_its_config(tmp_path):
    """An autoencoder run directory without a persisted config raises FileNotFoundError."""
    (tmp_path / "profile_runs" / "gru_run" / "meta").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        JepaNamingSpec.resolve(_entry(tmp_path, profile_autoencoder_run="gru_run"))
