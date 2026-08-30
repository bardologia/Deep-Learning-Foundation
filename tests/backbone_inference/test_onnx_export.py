"""Tests for the ONNX export of trained backbone runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx", reason="the export extra is not installed")

from configuration.diagnostics                import OnnxExportEntryConfig
from pipelines.backbone.inference.onnx_export import OnnxRunExporter

from tests.backbone_inference._helpers import build_run_directory
from tests.conftest                    import SilentLogger


@pytest.fixture
def run_root(tmp_path, test_data_dir, params_dir) -> Path:
    """Returns a runs root holding one complete backbone run."""
    root = tmp_path / "runs"
    root.mkdir()
    build_run_directory(root, test_data_dir, params_dir)

    return root


def _entry(root: Path, **overrides) -> OnnxExportEntryConfig:
    """Returns an export entry pointed at the runs root."""
    entry          = OnnxExportEntryConfig(**overrides)
    entry.runs_dir = root

    return entry


@pytest.mark.dataset
def test_export_writes_a_checkable_graph_and_manifest(run_root):
    """The exported graph passes the ONNX checker and the manifest ties it to the checkpoint."""
    paths = OnnxRunExporter(_entry(run_root), SilentLogger()).run()

    assert len(paths) == 1
    model = onnx.load(str(paths[0]))
    onnx.checker.check_model(model)

    manifest = json.loads((paths[0].parent / "manifest.json").read_text())

    assert manifest["model_name"]    == "unet"
    assert manifest["dynamic_batch"] is True
    assert manifest["onnx"]["sha256"]
    assert model.graph.input[0].name == "input"


@pytest.mark.dataset
def test_export_dynamic_batch_axis(run_root):
    """With dynamic_batch on, the input's leading dimension is symbolic."""
    paths = OnnxRunExporter(_entry(run_root), SilentLogger()).run()

    model = onnx.load(str(paths[0]))
    dim0  = model.graph.input[0].type.tensor_type.shape.dim[0]

    assert dim0.dim_param == "batch"


def test_export_raises_without_runs(tmp_path):
    """An empty runs root fails loudly."""
    root = tmp_path / "runs"
    root.mkdir()

    with pytest.raises(FileNotFoundError):
        OnnxRunExporter(_entry(root), SilentLogger()).run()
