from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from configuration.inference                         import InferenceConfig
from models                                          import BACKBONE_CONFIG_REGISTRY, BACKBONE_MODEL_REGISTRY, get_backbone
from pipelines.backbone.inference.loader             import AbsentTomogram, RunLoader
from pipelines.backbone.inference.model_wrapper      import ModelWrapper
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.shared.config.config_persistence      import BackboneModelConfigIO

from tests.models_backbone._helpers import SMALL_OVERRIDES, WINDOW


def _write_fake_checkpoint(path: Path, x_axis: np.ndarray) -> dict:
    state = {
        "weight" : torch.zeros(2, 2),
        "bias"   : torch.zeros(2),
    }
    ckpt = {
        "params"        : state,
        "x_axis"        : x_axis,
        "epoch"         : 7,
        "best_val_loss" : 0.125,
        "best_epoch"    : 5,
    }
    torch.save(ckpt, str(path))
    return ckpt


def test_load_checkpoint_numpy_x_axis(tmp_path):
    x_axis    = np.linspace(-20.0, 80.0, 32).astype(np.float32)
    ckpt_path = tmp_path / "best_model.pt"
    _write_fake_checkpoint(ckpt_path, x_axis)

    loader = RunLoader.__new__(RunLoader)
    ckpt, loaded_axis, meta = loader._load_checkpoint(ckpt_path, "cpu")

    assert isinstance(loaded_axis, np.ndarray)
    assert loaded_axis.dtype == np.float32
    assert np.allclose(loaded_axis, x_axis)
    assert meta["epoch"]         == 7
    assert meta["best_epoch"]    == 5
    assert meta["best_val_loss"] == pytest.approx(0.125)
    assert "params" in ckpt


def test_load_checkpoint_torch_tensor_x_axis(tmp_path):
    x_axis    = torch.linspace(-20.0, 80.0, 16)
    ckpt_path = tmp_path / "ckpt.pt"
    _write_fake_checkpoint(ckpt_path, x_axis)

    loader = RunLoader.__new__(RunLoader)
    _, loaded_axis, meta = loader._load_checkpoint(ckpt_path, "cpu")

    assert isinstance(loaded_axis, np.ndarray)
    assert loaded_axis.dtype == np.float32
    assert loaded_axis.size  == 16
    assert np.allclose(loaded_axis, x_axis.numpy())


def test_load_checkpoint_uses_weights_only_false(tmp_path):
    x_axis    = np.arange(8, dtype=np.float32)
    ckpt_path = tmp_path / "obj.pt"
    ckpt      = {
        "params"        : {"w": torch.ones(1)},
        "x_axis"        : x_axis,
        "epoch"         : 0,
        "best_val_loss" : 1.0,
        "best_epoch"    : 0,
        "extra"         : np.array([1, 2, 3]),
    }
    torch.save(ckpt, str(ckpt_path))

    loader = RunLoader.__new__(RunLoader)
    loaded, axis, _ = loader._load_checkpoint(ckpt_path, "cpu")

    assert np.array_equal(loaded["extra"], np.array([1, 2, 3]))
    assert axis.size == 8


def test_model_wrapper_identity_no_normalizer():
    class _Echo(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    wrapper = ModelWrapper(_Echo(), "cpu")
    out     = wrapper(np.ones((1, 3, 4, 4), dtype=np.float32))

    assert isinstance(out, np.ndarray)
    assert np.allclose(out, 2.0)


def test_model_wrapper_denormalize_passthrough_without_clamp():
    class _Echo(torch.nn.Module):
        def forward(self, x):
            return x

    wrapper = ModelWrapper(_Echo(), "cpu", x_axis=None, amp_max=None, normalizer=None)
    t       = torch.arange(6.0).reshape(1, 6, 1, 1)

    assert torch.allclose(wrapper.denormalize_output(t), t)


def test_inference_metadata_paths_with_output_subdir(tmp_path):
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="my_run", device="cpu")
    meta = InferenceMetadata(cfg)

    assert meta.output_dir == tmp_path / "inference" / "my_run"
    assert meta.figures_dir.parent    == meta.output_dir
    assert meta.figure_path("x", "png") == meta.figures_dir / "x.png"


def test_inference_metadata_create_dirs(tmp_path):
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="run0", device="cpu")
    meta = InferenceMetadata(cfg)
    meta.create_dirs()

    assert meta.output_dir.is_dir()
    assert meta.figures_dir.is_dir()
    assert meta.animations_dir.is_dir()
    assert meta.logs_dir.is_dir()
    assert meta.cube_dir.is_dir()


@pytest.mark.parametrize("name", sorted(BACKBONE_MODEL_REGISTRY))
def test_build_model_reconstructs_every_registry_backbone(name, tmp_path):
    in_channels  = 2
    out_channels = 6
    meta_dir     = tmp_path / "meta"
    meta_dir.mkdir()

    config = BACKBONE_CONFIG_REGISTRY[name]()
    for key, value in SMALL_OVERRIDES.get(name, {}).items():
        setattr(config, key, value)

    torch.manual_seed(0)
    trained, config = get_backbone(name, config=config, in_channels=in_channels, out_channels=out_channels)
    trained.eval()

    BackboneModelConfigIO.save(config, name, meta_dir)

    x_axis    = np.linspace(-20.0, 80.0, 32).astype(np.float32)
    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"params": trained.state_dict(), "x_axis": x_axis, "epoch": 1, "best_val_loss": 0.1, "best_epoch": 1}, str(ckpt_path))

    loader  = RunLoader(tmp_path, logger=None)
    rebuilt = loader._build_model(name, in_channels, out_channels, (WINDOW, WINDOW))

    ckpt, _, _ = loader._load_checkpoint(ckpt_path, "cpu")
    rebuilt.load_state_dict(ckpt["params"])
    rebuilt.eval()

    torch.manual_seed(1)
    x = torch.randn(1, in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        expected = trained(x)
        actual   = ModelWrapper(rebuilt, "cpu")(x.numpy())

    assert loader.model_head == config.head
    assert actual.shape == (1, out_channels, WINDOW, WINDOW)
    assert np.allclose(actual, expected.numpy(), atol=1e-6)


def test_persisted_n_gaussians_is_authoritative(tmp_path):
    loader = RunLoader(tmp_path, logger=None)

    assert loader._persisted_n_gaussians({"out_channels": 15, "n_gaussians": 5}) == 5

    loader._validate_out_channels(15, 5)


def test_persisted_n_gaussians_missing_raises(tmp_path):
    loader = RunLoader(tmp_path, logger=None)

    with pytest.raises(KeyError, match="n_gaussians"):
        loader._persisted_n_gaussians({"out_channels": 15})


def test_out_channels_inconsistent_with_n_gaussians_raises(tmp_path):
    loader = RunLoader(tmp_path, logger=None)

    with pytest.raises(ValueError, match="24"):
        loader._validate_out_channels(24, 5)


def test_absent_tomogram_refuses_every_use(tmp_path):
    absent = AbsentTomogram(tmp_path)

    with pytest.raises(ValueError, match="load_tomogram=False"):
        _ = absent.shape

    with pytest.raises(ValueError, match="load_tomogram=False"):
        _ = absent[0]

    with pytest.raises(ValueError, match="load_tomogram=False"):
        np.asarray(absent)


def test_full_run_load_skips_without_run_dir():
    candidate = Path("/ste/rnd/User/vice_vi")
    if candidate.is_dir():
        pytest.skip("server run directory present; full RunLoader.load covered elsewhere")
    pytest.skip("no real trained run directory available locally; RunLoader.load requires meta/ + checkpoint + preprocessing run")
