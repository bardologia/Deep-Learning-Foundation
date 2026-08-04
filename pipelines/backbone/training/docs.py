from __future__ import annotations

from pathlib import Path
from typing  import Optional

import torch
import torch.nn as nn
from rich.tree        import Tree
from torch.utils.data import DataLoader

from tools.reporting.markdown import MarkdownDoc, MarkdownTable


class LayerRecord:
    def __init__(self, name: str, module: nn.Module) -> None:
        self.name      = name
        self.module    = module
        self.type_name = module.__class__.__name__
        self.depth     = name.count(".") + 1
        self.trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        self.frozen    = sum(p.numel() for p in module.parameters(recurse=False) if not p.requires_grad)
        self.in_shape  = None
        self.out_shape = None
        self.visited   = False

    @property
    def own_params(self) -> int:
        return self.trainable + self.frozen


class ModelInspector:
    def __init__(self, model: nn.Module, logger, docs_dir, include_types=None) -> None:
        self.model         = model
        self.logger        = logger
        self.docs_dir      = Path(docs_dir)
        self.include_types = include_types
        self.records       = []
        self.hooks         = []

    def _shape_of(self, value):
        if hasattr(value, "shape"):
            return tuple(value.shape)
        if isinstance(value, (tuple, list)):
            head = value[0] if value else None
            return tuple(head.shape) if hasattr(head, "shape") else f"{type(value).__name__}[{len(value)}]"
        return type(value).__name__

    def _hook(self, record: LayerRecord):
        def fn(module, inputs, output):
            record.in_shape  = self._shape_of(inputs[0]) if inputs else None
            record.out_shape = self._shape_of(output)
            record.visited   = True
        return fn

    def _build(self) -> None:
        self.records = [LayerRecord(name, module) for name, module in self.model.named_modules() if name != ""]

    def _attach(self) -> None:
        self.hooks = [record.module.register_forward_hook(self._hook(record)) for record in self.records]

    def _detach(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def run(self, sample: torch.Tensor) -> "ModelInspector":
        self._build()
        self._attach()

        was_training = self.model.training
        self.model.eval()

        try:
            self.model(sample)
        finally:
            self._detach()
            if was_training:
                self.model.train()

        return self

    def totals(self) -> dict:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        size_mb   = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 ** 2

        return {
            "total"     : trainable + frozen,
            "trainable" : trainable,
            "frozen"    : frozen,
            "size_mb"   : size_mb,
        }

    def _selected(self) -> list[LayerRecord]:
        if self.include_types is None:
            return list(self.records)
        return [r for r in self.records if isinstance(r.module, self.include_types) or r.own_params > 0]

    def _shape_cell(self, shape) -> Optional[str]:
        return str(shape) if shape is not None else None

    def _node_label(self, record: LayerRecord) -> str:
        leaf  = record.name.rpartition(".")[2]
        label = f"[key]{leaf}[/key] [muted]{record.type_name}[/muted]"

        if record.visited:
            label += f"  {record.in_shape} [muted]->[/muted] {record.out_shape}"
        if record.own_params > 0:
            label += f"  [value]{record.own_params:,}[/value]"

        return label

    def render_console(self) -> None:
        totals = self.totals()
        tree   = Tree(f"[section]{self.model.__class__.__name__}[/section]")
        nodes  = {"": tree}

        for record in self.records:
            parent             = nodes.get(record.name.rpartition(".")[0], tree)
            nodes[record.name] = parent.add(self._node_label(record))

        self.logger.render(tree)
        self.logger.kv_table({
            "Total parameters"     : f"{totals['total']:,}",
            "Trainable parameters" : f"{totals['trainable']:,}",
            "Frozen parameters"    : f"{totals['frozen']:,}",
            "Parameter size"       : f"{totals['size_mb']:.2f} MB",
        }, title="Parameter Totals")

    def to_markdown(self, title: str = "Model Documentation") -> MarkdownDoc:
        totals = self.totals()
        rows   = self._selected()

        doc = MarkdownDoc(title)
        doc.bold_kv("Model",                self.model.__class__.__name__)
        doc.bold_kv("Total Parameters",     f"{totals['total']:,}")
        doc.bold_kv("Trainable Parameters", f"{totals['trainable']:,}")
        doc.bold_kv("Frozen Parameters",    f"{totals['frozen']:,}")
        doc.bold_kv("Parameter Size",       f"{totals['size_mb']:.2f} MB")
        doc.blank()

        columns = ["Layer", "Type", "Input shape", "Output shape", "Params", "Trainable", "Share %"]
        align   = ["left", "left", "left", "left", "right", "right", "right"]
        table   = MarkdownTable(columns, align=align)

        for r in rows:
            share = 100.0 * r.own_params / totals["total"] if totals["total"] else 0.0
            table.add_row(
                f"`{r.name}`",
                r.type_name,
                self._shape_cell(r.in_shape),
                self._shape_cell(r.out_shape),
                f"{r.own_params:,}",
                f"{r.trainable:,}",
                f"{share:.2f}",
            )

        doc.table(table)
        doc.bold_kv("Layers Documented", len(rows))
        return doc

    def save_markdown(self, filename: str = "model_doc.md", title: str = "Model Documentation") -> Path:
        path = self.to_markdown(title=title).save(self.docs_dir / filename)
        self.logger.subsection(f"Model documentation saved to {path}")
        return path


class TrainingDocs:
    def __init__(self, model, model_cfg, logger, run_dir, enabled=True):
        self.model     = model
        self.model_cfg = model_cfg
        self.logger    = logger
        self.run_dir   = Path(run_dir)
        self.enabled   = enabled

    @torch.no_grad()
    def emit(self, data_loader: DataLoader, device: torch.device) -> None:
        if not self.enabled:
            return

        self.logger.section("[Model Documentation]")

        inspector = ModelInspector(
            model         = self.model,
            logger        = self.logger,
            docs_dir      = self.run_dir / "docs",
            include_types = self.model_cfg.shape_logger_types,
        )

        batch  = next(iter(data_loader))
        sample = batch[0].to(device)

        inspector.run(sample)
        inspector.render_console()
        inspector.save_markdown()
