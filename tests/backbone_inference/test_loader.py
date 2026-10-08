"""Tests for loading a trained backbone run back into an inference session.

Covers checkpoint reading and z-axis coercion, the numpy-facing model wrapper,
the inference output-path layout, rebuilding every registered backbone from its saved
configuration, slot-count validation, the refusal of an unloaded cube, and a full
real-data run assembly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from configuration.inference                         import InferenceConfig
from models                                          import BACKBONE_CONFIG_REGISTRY, BACKBONE_MODEL_REGISTRY, get_backbone
from pipelines.backbone.inference.loader             import AbsentCube, RunLoader
from pipelines.backbone.inference.model_wrapper      import ModelWrapper
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.shared.config.config_persistence      import BackboneModelConfigIO
from tools.data.regions                              import CropRegion

from tests.backbone_inference._helpers import N_Z, build_run_directory
from tests.conftest                    import SilentLogger
from tests.models_backbone._helpers    import SMALL_OVERRIDES, WINDOW


def _write_fake_checkpoint(path: Path, z_axis: np.ndarray) -> dict:
    """Writes a minimal checkpoint holding two dummy tensors and the z axis.

    Args:
        path   : Destination file of the checkpoint.
        z_axis : Z axis in metres, as numpy array or torch tensor.

    Returns:
        The checkpoint dictionary that was written.
    """
    state = {
        "weight" : torch.zeros(2, 2),
        "bias"   : torch.zeros(2),
    }
    ckpt = {
        "params"        : state,
        "z_axis"        : z_axis,
        "epoch"         : 7,
        "best_val_loss" : 0.125,
        "best_epoch"    : 5,
    }
    torch.save(ckpt, str(path))
    return ckpt


def test_load_checkpoint_missing_file_raises(tmp_path):
    """A missing checkpoint file raises FileNotFoundError."""
    loader = RunLoader.__new__(RunLoader)

    with pytest.raises(FileNotFoundError, match="Checkpoint not found for inference"):
        loader._load_checkpoint(tmp_path / "absent.pt", "cpu")


def test_load_checkpoint_numpy_x_axis(tmp_path):
    """A numpy z axis is returned as float32 alongside the epoch and loss metadata."""
    z_axis    = np.linspace(-20.0, 80.0, 32).astype(np.float32)
    ckpt_path = tmp_path / "best_model.pt"
    _write_fake_checkpoint(ckpt_path, z_axis)

    loader = RunLoader.__new__(RunLoader)
    ckpt, loaded_axis, meta = loader._load_checkpoint(ckpt_path, "cpu")

    assert isinstance(loaded_axis, np.ndarray)
    assert loaded_axis.dtype == np.float32
    assert np.allclose(loaded_axis, z_axis)
    assert meta["epoch"]         == 7
    assert meta["best_epoch"]    == 5
    assert meta["best_val_loss"] == pytest.approx(0.125)
    assert "params" in ckpt


def test_load_checkpoint_torch_tensor_x_axis(tmp_path):
    """A torch z axis is converted to a float32 numpy array."""
    z_axis    = torch.linspace(-20.0, 80.0, 16)
    ckpt_path = tmp_path / "ckpt.pt"
    _write_fake_checkpoint(ckpt_path, z_axis)

    loader = RunLoader.__new__(RunLoader)
    _, loaded_axis, meta = loader._load_checkpoint(ckpt_path, "cpu")

    assert isinstance(loaded_axis, np.ndarray)
    assert loaded_axis.dtype == np.float32
    assert loaded_axis.size  == 16
    assert np.allclose(loaded_axis, z_axis.numpy())


def test_load_checkpoint_uses_weights_only_false(tmp_path):
    """Non-tensor checkpoint entries survive the load, so unpickling is not restricted to weights."""
    z_axis    = np.arange(8, dtype=np.float32)
    ckpt_path = tmp_path / "obj.pt"
    ckpt      = {
        "params"        : {"w": torch.ones(1)},
        "z_axis"        : z_axis,
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
    """Without a normalizer the wrapper returns the raw model output as a numpy array."""
    class _Echo(torch.nn.Module):
        """Module returning twice its input."""
        def forward(self, x):
            """Returns 2 * x, of the same shape as x."""
            return x * 2.0

    wrapper = ModelWrapper(_Echo(), "cpu")
    out     = wrapper(np.ones((1, 3, 4, 4), dtype=np.float32))

    assert isinstance(out, np.ndarray)
    assert np.allclose(out, 2.0)


def test_model_wrapper_denormalize_passthrough_without_clamp():
    """Denormalization is the identity when no normalizer and no amplitude ceiling are configured."""
    class _Echo(torch.nn.Module):
        def forward(self, x):
            """Returns x unchanged."""
            return x

    wrapper = ModelWrapper(_Echo(), "cpu", z_axis=None, amp_max=None, normalizer=None)
    t       = torch.arange(6.0).reshape(1, 6, 1, 1)

    assert torch.allclose(wrapper.denormalize_output(t), t)


def test_inference_metadata_paths_with_output_subdir(tmp_path):
    """Output, figures and figure-file paths hang off the configured inference subdirectory."""
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="my_run", device="cpu")
    meta = InferenceMetadata(cfg)

    assert meta.output_dir == tmp_path / "inference" / "my_run"
    assert meta.figures_dir.parent    == meta.output_dir
    assert meta.figure_path("x", "png") == meta.figures_dir / "x.png"


def test_inference_metadata_create_dirs(tmp_path):
    """Creating the directories materialises the output, figures, animations, logs and cube folders."""
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
    """Every registered backbone rebuilds from its saved configuration and reproduces the trained output."""
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

    z_axis    = np.linspace(-20.0, 80.0, 32).astype(np.float32)
    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"params": trained.state_dict(), "z_axis": z_axis, "epoch": 1, "best_val_loss": 0.1, "best_epoch": 1}, str(ckpt_path))

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
    """The persisted slot count is taken from the run summary and validated against the output channels."""
    loader = RunLoader(tmp_path, logger=None)

    assert loader._persisted_n_gaussians({"out_channels": 15, "n_gaussians": 5}) == 5

    loader._validate_out_channels(15, 5)


def test_persisted_n_gaussians_missing_raises(tmp_path):
    """A run summary without a slot count raises KeyError."""
    loader = RunLoader(tmp_path, logger=None)

    with pytest.raises(KeyError, match="n_gaussians"):
        loader._persisted_n_gaussians({"out_channels": 15})


def test_out_channels_inconsistent_with_n_gaussians_raises(tmp_path):
    """An output-channel count that is not three times the slot count raises ValueError."""
    loader = RunLoader(tmp_path, logger=None)

    with pytest.raises(ValueError, match="24"):
        loader._validate_out_channels(24, 5)


def test_absent_cube_refuses_every_use(tmp_path):
    """Shape access, indexing and array conversion all raise when the cube was not loaded."""
    absent = AbsentCube(tmp_path)

    with pytest.raises(ValueError, match="load_cube=False"):
        _ = absent.shape

    with pytest.raises(ValueError, match="load_cube=False"):
        _ = absent[0]

    with pytest.raises(ValueError, match="load_cube=False"):
        np.asarray(absent)


@pytest.mark.dataset
def test_load_refuses_a_checkpoint_axis_that_differs_from_the_dataset_axis(tmp_path, test_data_dir, params_dir):
    """A checkpoint trained on another z range with the same bin count is refused, so no z metric is reported in the wrong units."""
    run_dir   = build_run_directory(tmp_path, test_data_dir, params_dir)
    ckpt_path = run_dir / "best_model.pt"

    ckpt           = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt["z_axis"] = np.linspace(-30.0, 70.0, N_Z).astype(np.float32)
    torch.save(ckpt, str(ckpt_path))

    with pytest.raises(ValueError, match="z axis"):
        RunLoader(run_dir, logger=SilentLogger()).load(split="test", batch_size=None, num_workers=0, device="cpu", checkpoint_name="best_model.pt")


class _TableLogger(SilentLogger):
    """Logger stub recording every key-value table it is asked to print.

    Attributes:
        tables : Field-to-value mappings in emission order.
    """

    def __init__(self):
        """Starts with no recorded table."""
        self.tables = []

    def kv_table(self, data, title=None, **kwargs):
        """Records the table fields, discarding the title."""
        self.tables.append(dict(data))


@pytest.mark.dataset
@pytest.mark.slow
def test_full_run_load_logs_the_effective_checkpoint_and_split_values(tmp_path, test_data_dir, params_dir):
    """The [Model] and [Split] tables name the one-based checkpoint epoch, both z axes, the region, its source and the effective batch size before any pass runs."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    logger  = _TableLogger()

    RunLoader(run_dir, logger=logger).load(split="test", batch_size=None, num_workers=0, device="cpu", checkpoint_name="best_model.pt", region_override=CropRegion(1180, 1296, 500, 596))

    model_table = next(table for table in logger.tables if "Checkpoint epoch" in table)
    split_table = next(table for table in logger.tables if "Region source" in table)

    assert model_table["Checkpoint epoch"]    == "2 (best epoch 2, best val loss 0.1)"
    assert model_table["Z axis (checkpoint)"] == model_table["Z axis (dataset)"] == f"[-20.000, 80.000] m, {N_Z} bins"
    assert split_table["Region"]              == (1180, 1296, 500, 596)
    assert split_table["Region source"]       == "region override; stored core region (1200, 1296, 500, 596)"
    assert split_table["Batch size"]          == "4 (from training)"
    assert split_table["Loader workers"]      == 0
