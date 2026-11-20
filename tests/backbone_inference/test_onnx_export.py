
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
def run_root(tmp_path, test_data_dir) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    build_run_directory(root, test_data_dir)

    return root


def _entry(root: Path, **overrides) -> OnnxExportEntryConfig:
    entry          = OnnxExportEntryConfig(**overrides)
    entry.runs_dir = root

    return entry


@pytest.mark.dataset
def test_export_writes_a_checkable_graph_and_manifest(run_root):
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
    paths = OnnxRunExporter(_entry(run_root), SilentLogger()).run()

    model = onnx.load(str(paths[0]))
    dim0  = model.graph.input[0].type.tensor_type.shape.dim[0]

    assert dim0.dim_param == "batch"

