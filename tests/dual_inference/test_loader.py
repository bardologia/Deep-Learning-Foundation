from __future__ import annotations

import numpy as np
import pytest
import torch

from models.dual                                import DUAL_CONFIG_REGISTRY, DUAL_MODEL_REGISTRY, get_dual
from pipelines.backbone.inference.model_wrapper import ModelWrapper
from pipelines.dual.inference.loader            import DualRunLoader
from pipelines.shared.config.config_persistence import DualModelConfigIO


WINDOW = 32

SMALL_OVERRIDES = {
    "params_features"     : [8, 16],
    "existence_features"  : [8],
    "params_overrides"    : {"bottleneck_factor": 1, "dropout": 0.0},
    "existence_overrides" : {"bottleneck_factor": 1, "dropout": 0.0},
    "params_channels"     : (0, 1, 2, 3, 4),
    "existence_channels"  : (3, 4),
}


@pytest.mark.parametrize("name", sorted(DUAL_MODEL_REGISTRY))
def test_build_model_reconstructs_every_registry_dual_model(name, tmp_path):
    in_channels  = 5
    out_channels = 6
    meta_dir     = tmp_path / "meta"
    meta_dir.mkdir()

    config = DUAL_CONFIG_REGISTRY[name]()
    for key, value in SMALL_OVERRIDES.items():
        setattr(config, key, value)

    torch.manual_seed(0)
    trained, config = get_dual(name, config=config, in_channels=in_channels, out_channels=out_channels)
    trained.eval()

    DualModelConfigIO.save(config, name, meta_dir)

    x_axis    = np.linspace(-20.0, 80.0, 32).astype(np.float32)
    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"params": trained.state_dict(), "x_axis": x_axis, "epoch": 1, "best_val_loss": 0.1, "best_epoch": 1}, str(ckpt_path))

    loader  = DualRunLoader(tmp_path, logger=None)
    rebuilt = loader._build_model(name, in_channels, out_channels, WINDOW)

    ckpt, _, _ = loader._load_checkpoint(ckpt_path, "cpu")
    rebuilt.load_state_dict(ckpt["params"])
    rebuilt.eval()

    torch.manual_seed(1)
    x = torch.randn(1, in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        expected = trained(x)
        actual   = ModelWrapper(rebuilt, "cpu")(x.numpy())

    assert loader.model_head == "set_pred"
    assert tuple(rebuilt.config.params_channels)    == (0, 1, 2, 3, 4)
    assert tuple(rebuilt.config.existence_channels) == (3, 4)
    assert actual.shape == (1, out_channels, WINDOW, WINDOW)
    assert np.allclose(actual, expected.numpy(), atol=1e-6)
