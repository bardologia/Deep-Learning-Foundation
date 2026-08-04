from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from pipelines.backbone.training.docs import LayerRecord, ModelInspector, TrainingDocs
from tests.backbone_training._helpers import tiny_model
from tools.monitoring.logger          import Logger


def _loader(in_channels: int = 2, n_gaussians: int = 2, hw: int = 16) -> DataLoader:
    imgs = torch.randn(2, in_channels, hw, hw)
    tgt  = torch.randn(2, n_gaussians * 3, hw, hw)
    return DataLoader(TensorDataset(imgs, tgt), batch_size=1)


def test_inspector_totals_report_all_parameters(tmp_path):
    model, model_cfg = tiny_model()
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="docs", level="ERROR")

    inspector = ModelInspector(model, logger, tmp_path / "docs", include_types=model_cfg.shape_logger_types)
    inspector.run(torch.randn(1, 2, 16, 16))

    totals = inspector.totals()

    assert set(totals.keys())  == {"total", "trainable", "frozen", "size_mb"}
    assert totals["total"]     == sum(p.numel() for p in model.parameters())
    assert totals["trainable"] == totals["total"]
    assert totals["size_mb"]   > 0.0


def test_inspector_records_shapes_after_run(tmp_path):
    model, model_cfg = tiny_model()
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="docs", level="ERROR")

    inspector = ModelInspector(model, logger, tmp_path / "docs", include_types=model_cfg.shape_logger_types)
    inspector.run(torch.randn(1, 2, 16, 16))

    visited = [record for record in inspector.records if record.visited]

    assert len(visited) > 0
    assert all(isinstance(record, LayerRecord) for record in inspector.records)
    assert any(record.in_shape is not None for record in visited)


def test_inspector_to_markdown_structure(tmp_path):
    model, model_cfg = tiny_model()
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="docs", level="ERROR")

    inspector = ModelInspector(model, logger, tmp_path / "docs", include_types=model_cfg.shape_logger_types)
    inspector.run(torch.randn(1, 2, 16, 16))

    text = inspector.to_markdown(title="Tiny Model").render()

    assert "Tiny Model"             in text
    assert "Total Parameters"       in text
    assert model.__class__.__name__ in text


def test_inspector_save_markdown_writes_file(tmp_path):
    model, model_cfg = tiny_model()
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="docs", level="ERROR")

    docs_dir  = tmp_path / "docs"
    inspector = ModelInspector(model, logger, docs_dir, include_types=model_cfg.shape_logger_types)
    inspector.run(torch.randn(1, 2, 16, 16))

    path = inspector.save_markdown(filename="model_doc.md")

    assert path.exists()
    assert path.read_text().strip() != ""


def test_training_docs_emit_writes_documentation(tmp_path):
    model, model_cfg = tiny_model()
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="docs", level="ERROR")

    docs = TrainingDocs(model, model_cfg, logger, tmp_path, enabled=True)
    docs.emit(_loader(), torch.device("cpu"))

    assert (tmp_path / "docs" / "model_doc.md").exists()


def test_training_docs_disabled_writes_nothing(tmp_path):
    model, model_cfg = tiny_model()
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="docs", level="ERROR")

    docs = TrainingDocs(model, model_cfg, logger, tmp_path, enabled=False)
    docs.emit(_loader(), torch.device("cpu"))

    assert not (tmp_path / "docs").exists()


def test_layer_record_own_params_sums_trainable_and_frozen():
    conv   = torch.nn.Conv2d(2, 4, 3)
    record = LayerRecord("conv", conv)

    expected = sum(p.numel() for p in conv.parameters())

    assert record.own_params == expected
    assert record.frozen     == 0
