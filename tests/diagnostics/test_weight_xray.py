"""Tests for the weight x-ray orchestration: checkpoint resolution, module types, provenance and the batch.

Covers which checkpoint entry the tensors come from and the refusal to guess
between unconfigured entries, the module types rebuilt from a persisted
architecture so transposed convolutions are read on dim 1, the provenance block
and unchecked-check warning of one x-ray, and the numbered sections and closing
summary of a batch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configuration.diagnostics                  import WeightXrayConfig, WeightXrayEntryConfig
from models                                     import BACKBONE_CONFIG_REGISTRY, get_backbone
from pipelines.shared.config.config_persistence import BackboneModelConfigIO
from tools.diagnostics.weight_xray               import ModuleTypeResolver, StateDictResolver, WeightXray, WeightXrayBatch

from tests.conftest import RecordingLogger


def _state_dict(wide_units: int = 4) -> dict:
    """Returns a small state dict with one wide and one narrow weight plus a bias."""
    torch.manual_seed(0)
    return {
        "encoder.wide.weight"   : torch.randn(wide_units, 3) * 0.1,
        "encoder.narrow.weight" : torch.randn(2, 3) * 0.1,
        "encoder.narrow.bias"   : torch.randn(2) * 0.01,
    }


def _write_run(root, name: str, wide_units: int = 4):
    """Writes a run directory holding a best_model.pt in the trainer's layout and returns it."""
    run_dir = root / name
    run_dir.mkdir(parents=True)

    payload = {"epoch": 3, "best_epoch": 2, "best_val_loss": 0.1, "params": _state_dict(wide_units), "z_axis": np.linspace(-20.0, 80.0, 8)}
    torch.save(payload, run_dir / "best_model.pt")
    return run_dir


def _write_unet_run(root, name: str):
    """Writes a run directory with a persisted two-level U-Net config and its checkpoint, as training does.

    Returns:
        Pair of the run directory and the model the checkpoint was taken from.
    """
    run_dir  = root / name
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    config                   = BACKBONE_CONFIG_REGISTRY["unet"]()
    config.in_channels       = 2
    config.out_channels      = 3
    config.features          = [4, 8]
    config.bottleneck_factor = 1

    BackboneModelConfigIO.save(config, "unet", meta_dir)

    torch.manual_seed(0)
    model, _ = get_backbone("unet", config=config)
    torch.save({"epoch": 0, "best_epoch": 0, "params": model.state_dict(), "z_axis": np.linspace(-20.0, 80.0, 8)}, run_dir / "best_model.pt")

    return run_dir, model


def _messages(logger: RecordingLogger, level: str) -> list[str]:
    """Returns the recorded messages of one level."""
    return [message for recorded, message in logger.messages if recorded == level]


class _TableLogger(RecordingLogger):
    """Recording logger that also keeps the titles of the key-value tables it is asked to print."""

    def kv_table(self, data, title=None, **kwargs):
        """Records only the title of a key-value table, discarding the rows."""
        self.messages.append(("kv_table", title))


def test_resolver_reads_the_configured_entry_and_the_stored_epochs(tmp_path):
    """The tensors come from the configured entry and the stored epochs travel with them."""
    run_dir  = _write_run(tmp_path, "run_a")
    resolved = StateDictResolver(WeightXrayConfig(checkpoint_path=run_dir)).resolve()

    assert resolved.key        == "params"
    assert resolved.epoch      == 3
    assert resolved.best_epoch == 2
    assert set(resolved.state_dict) == set(_state_dict())


def test_resolver_accepts_a_bare_state_dict(tmp_path):
    """A checkpoint that is the mapping itself resolves at the top level with no epochs."""
    path = tmp_path / "weights.pt"
    torch.save(_state_dict(), path)

    resolved = StateDictResolver(WeightXrayConfig(checkpoint_path=path)).resolve()

    assert resolved.key   == StateDictResolver.TOP_LEVEL
    assert resolved.epoch is None


def test_resolver_refuses_to_guess_between_unconfigured_entries(tmp_path):
    """A trainer state without the configured entry raises and names the entries that hold tensors."""
    path = tmp_path / "last.pt"
    torch.save({"epoch": 1, "model": _state_dict(), "ema": _state_dict()}, path)

    with pytest.raises(ValueError, match=r"\['model', 'ema'\]"):
        StateDictResolver(WeightXrayConfig(checkpoint_path=path)).resolve()


def test_resolver_names_the_entries_when_none_holds_tensors(tmp_path):
    """A checkpoint without any tensor mapping raises and lists its entries."""
    path = tmp_path / "odd.pt"
    torch.save({"epoch": 1, "note": "nothing here"}, path)

    with pytest.raises(ValueError, match="its entries are"):
        StateDictResolver(WeightXrayConfig(checkpoint_path=path)).resolve()


def test_xray_logs_the_entry_and_writes_provenance_into_the_json(tmp_path):
    """One x-ray names the checkpoint entry in the log and stores entry, epochs, config and commit in the JSON."""
    run_dir = _write_run(tmp_path, "run_a")
    config  = WeightXrayConfig(checkpoint_path=run_dir, output_dir=run_dir / "weight_xray", make_plots=False, duplicate_max_units=3)
    logger  = RecordingLogger()

    result  = WeightXray(config, logger).run()
    payload = json.loads(config.report_json_path.read_text())

    assert any("checkpoint entry 'params' (epoch index 3, best epoch index 2)" in message for message in _messages(logger, "ok"))
    assert payload["provenance"]["checkpoint"] == {"path": str(run_dir / "best_model.pt"), "key": "params", "epoch": 3, "best_epoch": 2}
    assert payload["provenance"]["config"]["duplicate_max_units"] == 3
    assert payload["provenance"]["config"]["thresholds.duplicate_cosine"] == 0.999
    assert payload["provenance"]["git_commit"]
    assert payload["provenance"]["created"]
    assert result["provenance"] == payload["provenance"]


def test_xray_warns_about_the_tensors_a_limit_left_unchecked(tmp_path):
    """A weight above duplicate_max_units is named in a warning and counted as unchecked in the markdown."""
    run_dir = _write_run(tmp_path, "run_a", wide_units=8)
    config  = WeightXrayConfig(checkpoint_path=run_dir, output_dir=run_dir / "weight_xray", make_plots=False, duplicate_max_units=4)
    logger  = RecordingLogger()

    WeightXray(config, logger).run()
    markdown = config.report_markdown_path.read_text()

    warnings = [message for message in _messages(logger, "warning") if "check skipped" in message]
    assert len(warnings) == 1
    assert "duplicates check skipped for 1 of 3 tensors" in warnings[0]
    assert "encoder.wide.weight (8 output units > duplicate_max_units 4)" in warnings[0]
    assert "spectral 0, duplicates 1" in markdown
    assert "skipped" in markdown
    assert "Thresholds and limits" in markdown
    assert "duplicate_max_units" in markdown


def test_resolver_finds_the_transposed_convolutions_of_a_persisted_unet(tmp_path):
    """The rebuilt U-Net names every checkpoint tensor's module and its transposed-convolution weights."""
    run_dir, model = _write_unet_run(tmp_path, "unet_run")

    types    = ModuleTypeResolver(run_dir / "best_model.pt").resolve()
    expected = {f"{name}.weight" for name, module in model.named_modules() if isinstance(module, torch.nn.ConvTranspose2d)}

    assert types.architecture == "unet"
    assert expected
    assert types.transposed == expected
    assert set(types.by_key) == set(model.state_dict())


def test_xray_reads_transposed_weights_on_dim_1_and_records_the_architecture(tmp_path):
    """With a persisted architecture the upsampler weights report dim-1 output units and the JSON names the architecture."""
    run_dir, _model = _write_unet_run(tmp_path, "unet_run")
    config          = WeightXrayConfig(checkpoint_path=run_dir, output_dir=run_dir / "weight_xray", make_plots=False)
    logger          = RecordingLogger()

    WeightXray(config, logger).run()
    payload = json.loads(config.report_json_path.read_text())

    transposed = [tensor for tensor in payload["tensors"] if tensor["module_type"] == "ConvTranspose2d" and tensor["role"] == "weight"]
    assert transposed
    for tensor in transposed:
        assert tensor["unit_axis"] == 1
        assert tensor["out_units"] == tensor["shape"][1]

    assert payload["provenance"]["architecture"] == "unet"
    assert any("transposed-convolution weights read their output units on dim 1" in message for message in _messages(logger, "ok"))


def test_xray_without_a_persisted_architecture_warns_and_reads_dim_0(tmp_path):
    """A bare run directory without model_config.json is x-rayed on dim 0 with a warning and no architecture recorded."""
    run_dir = _write_run(tmp_path, "run_a")
    config  = WeightXrayConfig(checkpoint_path=run_dir, output_dir=run_dir / "weight_xray", make_plots=False)
    logger  = RecordingLogger()

    WeightXray(config, logger).run()
    payload = json.loads(config.report_json_path.read_text())

    assert any(BackboneModelConfigIO.FILENAME in message and "dim 0" in message for message in _messages(logger, "warning"))
    assert payload["provenance"]["architecture"] is None
    assert all(tensor["unit_axis"] == 0 for tensor in payload["tensors"])


def test_batch_logs_its_configuration_numbers_the_runs_and_closes_with_a_summary(tmp_path):
    """A batch opens one numbered section per run and ends with a summary table and verdict counts."""
    for name in ("run_a", "run_b"):
        _write_run(tmp_path, name)

    entry  = WeightXrayEntryConfig(runs_dir=tmp_path, run_filter=["run_a", "run_b"], make_plots=False)
    logger = _TableLogger()

    results  = WeightXrayBatch(entry, logger).run()
    sections = _messages(logger, "section")

    assert len(results) == 2
    assert logger.messages[0] == ("kv_table", "Weight x-ray configuration")
    assert sections == ["Weight x-ray [1/2] run_a", "Weight x-ray [2/2] run_b"]
    assert ("metrics_table", "Weight x-ray summary") in logger.messages
    assert any(message.startswith("X-rayed 2 run(s): ") and "clean" in message for message in _messages(logger, "ok"))
