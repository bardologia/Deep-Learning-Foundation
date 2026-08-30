"""Tests that RunRenamer migrates legacy JEPA run directories to the current naming scheme.

Covers profile-coupled runs renamed to the embedding-loss name, image-only runs keeping the
parameter-loss name while gaining the iae tag, runs naming both autoencoders, already-migrated
directories left untouched, and coupled runs skipped when their autoencoder config is absent."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib     import Path

import pytest

from configuration.dataset                import AugmentationConfig
from configuration.training               import EmbeddingLossConfig, LossConfig, ParamMatching
from configuration.training.jepa          import JepaMaskingConfig
from pipelines.shared.training.run_naming import RunNaming
from scripts.rename_runs                  import RunRenamer


def _param_loss() -> LossConfig:
    """Returns a parameter loss config using L1 at weight one with sorted-ground-truth matching."""
    return LossConfig(use_param_l1=True, weight_param_l1=1.0, use_active_normalization=True, param_matching=ParamMatching.SORTED_GT)


def _augmentation() -> AugmentationConfig:
    """Returns an augmentation config with horizontal and vertical flips only."""
    return AugmentationConfig(p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.0, p_noise=0.0)


def _trainer(profile: bool, image: bool) -> dict:
    """Returns a serialised trainer config payload with the requested autoencoders coupled.

    Args:
        profile : Whether a profile-autoencoder section is present.
        image   : Whether an image-autoencoder section is present.

    Returns:
        Trainer config dict as it is written to docs/trainer_config.json.
    """
    loss_payload                   = asdict(_param_loss())
    loss_payload["param_matching"] = str(_param_loss().param_matching)

    return {
        "gaussian"                 : {"n_default_gaussians": 5},
        "param_loss"               : loss_payload,
        "embedding_loss"           : asdict(EmbeddingLossConfig()),
        "masking"                  : asdict(JepaMaskingConfig()),
        "profile_autoencoder_mode" : "frozen",
        "target_provider"          : "stopgrad",
        "image_autoencoder_mode"   : "frozen",
        "autoencoder"              : {"embedding_dim": 24} if profile else None,
        "image_autoencoder"        : {"embedding_dim": 24} if image else None,
    }


def _write_metadata(run_dir: Path, profile: bool, image: bool) -> None:
    """Writes the meta and docs files a legacy run needs for the renamer to read it."""
    (run_dir / "meta").mkdir(parents=True)
    (run_dir / "docs").mkdir()

    (run_dir / "meta" / "run_summary.json").write_text(json.dumps({"model_name": "resunet", "n_gaussians": 5, "run_directory": str(run_dir)}))
    (run_dir / "meta" / "model_config.json").write_text(json.dumps({"model_name": "resunet", "config": {"head": "conv"}}))
    (run_dir / "meta" / "dataset_creation_config.json").write_text(json.dumps({"augmentation": asdict(_augmentation())}))
    (run_dir / "docs" / "trainer_config.json").write_text(json.dumps(_trainer(profile, image)))

    if profile:
        (run_dir / "meta" / "profile_autoencoder_config.json").write_text(json.dumps({"model_name": "gru_ae", "config": {}}))
    if image:
        (run_dir / "meta" / "image_autoencoder_config.json").write_text(json.dumps({"model_name": "conv2d_ae", "config": {}}))


def _renamer(root: Path, apply: bool, monkeypatch, tmp_path: Path) -> RunRenamer:
    """Returns a RunRenamer over `root`, with the working directory moved to tmp_path."""
    monkeypatch.chdir(tmp_path)
    return RunRenamer([root], apply=apply)


def _old_name() -> str:
    """Returns the legacy run tag built from the parameter loss and augmentation configs."""
    return RunNaming.tag("resunet", "conv", _param_loss(), 5, _augmentation())


def test_profile_coupled_jepa_run_is_renamed_to_the_embedding_loss_name(monkeypatch, tmp_path):
    """A profile-coupled run is renamed to the embedding-loss name and its summary is rewritten."""
    root     = tmp_path / "jepa"
    old_name = _old_name() + "_seed0"
    _write_metadata(root / old_name, profile=True, image=False)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    new_name = "resunet-conv-stopgrad-K_5-hv-none-pae_gru.frozen-emb_mse_1-curve_mse_1_seed0"
    assert not (root / old_name).exists()
    assert (root / new_name).is_dir()
    assert old_name not in (root / new_name / "meta" / "run_summary.json").read_text()


def test_image_only_jepa_run_keeps_the_param_loss_and_gains_the_iae_tag(monkeypatch, tmp_path):
    """An image-only run keeps the parameter-loss name and gains the image-autoencoder tag."""
    root     = tmp_path / "jepa"
    old_name = _old_name()
    _write_metadata(root / old_name, profile=False, image=True)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    new_name = "resunet-conv-sorted_gt-K_5-hv-A-iae_conv2d.frozen-param_l1_1"
    assert not (root / old_name).exists()
    assert (root / new_name).is_dir()


def test_jepa_run_with_both_autoencoders_names_both(monkeypatch, tmp_path):
    """A run coupling both autoencoders is renamed to a directory naming both."""
    root     = tmp_path / "jepa"
    old_name = _old_name()
    _write_metadata(root / old_name, profile=True, image=True)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    new_name = "resunet-conv-stopgrad-K_5-hv-none-pae_gru.frozen-iae_conv2d.frozen-emb_mse_1-curve_mse_1"
    assert (root / new_name).is_dir()


def test_already_migrated_jepa_run_is_left_alone(monkeypatch, tmp_path):
    """A directory already carrying the new name is not renamed again."""
    root     = tmp_path / "jepa"
    new_name = "resunet-conv-stopgrad-K_5-hv-none-pae_gru.frozen-emb_mse_1-curve_mse_1"
    _write_metadata(root / new_name, profile=True, image=False)

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    assert (root / new_name).is_dir()


def test_coupled_run_without_persisted_ae_config_is_skipped(monkeypatch, tmp_path):
    """A coupled run missing its autoencoder config keeps its old directory name."""
    root     = tmp_path / "jepa"
    old_name = _old_name()
    _write_metadata(root / old_name, profile=True, image=False)
    (root / old_name / "meta" / "profile_autoencoder_config.json").unlink()

    _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()

    assert (root / old_name).is_dir()


def test_run_predating_the_masking_block_breaks_loudly(monkeypatch, tmp_path):
    """A run whose trainer config has no masking section aborts the rename with a ValueError."""
    root     = tmp_path / "jepa"
    old_name = _old_name()
    _write_metadata(root / old_name, profile=True, image=False)

    payload = json.loads((root / old_name / "docs" / "trainer_config.json").read_text())
    payload.pop("masking")
    (root / old_name / "docs" / "trainer_config.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="predates the input-masking block"):
        _renamer(root, apply=True, monkeypatch=monkeypatch, tmp_path=tmp_path).run()
