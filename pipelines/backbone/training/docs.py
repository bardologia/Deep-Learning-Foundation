"""Model documentation emitted at the start of a training run.

Traces one real batch through the model with forward hooks to record per-layer input and
output shapes and parameter counts, then renders the result both as a console tree and as
a markdown document in the run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Optional

import torch
import torch.nn as nn
from rich.tree        import Tree
from torch.utils.data import DataLoader

from tools.reporting.markdown import MarkdownDoc, MarkdownTable


class LayerRecord:
    """One submodule of the traced model and the shapes observed through it.

    Attributes:
        name      : Dotted module path inside the model.
        module    : The submodule itself.
        type_name : Class name of the submodule.
        depth     : Nesting depth of the module path.
        trainable : Parameters owned directly by this module that require gradients.
        frozen    : Parameters owned directly by this module that do not require gradients.
        in_shape  : Shape of the first positional input seen, or None before the trace.
        out_shape : Shape of the output seen, or None before the trace.
        visited   : Whether the forward hook fired during the trace.
    """

    def __init__(self, name: str, module: nn.Module) -> None:
        """Records the module's identity and its own (non-recursive) parameter counts."""
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
        """Total parameters owned directly by this module."""
        return self.trainable + self.frozen


class ModelInspector:
    """Traces a model on one sample and documents its layers, shapes and parameter counts.

    Attributes:
        model         : Model being documented.
        logger        : Logger receiving the console tree and the totals table.
        docs_dir      : Directory the markdown document is written into.
        include_types : Module types kept in the markdown table; parameterless others are dropped when set.
        records       : Per-layer records, populated by the trace.
        hooks         : Active forward-hook handles during the trace.
    """

    def __init__(self, model: nn.Module, logger, docs_dir, include_types=None) -> None:
        """Stores the model, the logger, the documentation directory and the table filter."""
        self.model         = model
        self.logger        = logger
        self.docs_dir      = Path(docs_dir)
        self.include_types = include_types
        self.records       = []
        self.hooks         = []

    def _shape_of(self, value):
        """Returns a tensor's shape tuple, a sequence's head shape or length, or its type name."""
        if hasattr(value, "shape"):
            return tuple(value.shape)
        if isinstance(value, (tuple, list)):
            head = value[0] if value else None
            return tuple(head.shape) if hasattr(head, "shape") else f"{type(value).__name__}[{len(value)}]"
        return type(value).__name__

    def _hook(self, record: LayerRecord):
        """Returns a forward hook writing the observed input and output shapes into a record."""
        def fn(module, inputs, output):
            record.in_shape  = self._shape_of(inputs[0]) if inputs else None
            record.out_shape = self._shape_of(output)
            record.visited   = True
        return fn

    def _build(self) -> None:
        """Creates one record per named submodule, excluding the model root."""
        self.records = [LayerRecord(name, module) for name, module in self.model.named_modules() if name != ""]

    def _attach(self) -> None:
        """Registers the forward hook of every record."""
        self.hooks = [record.module.register_forward_hook(self._hook(record)) for record in self.records]

    def _detach(self) -> None:
        """Removes every registered forward hook."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def run(self, sample: torch.Tensor) -> "ModelInspector":
        """Traces the model in eval mode on one sample, restoring the training mode afterwards.

        Args:
            sample : Model input batch of shape (batch, in_channels, height, width).

        Returns:
            This inspector, with the per-layer records filled in.
        """
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
        """Returns the model's total, trainable and frozen parameter counts and its size in MB."""
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
        """Returns the records for the markdown table, filtered by type unless none is configured."""
        if self.include_types is None:
            return list(self.records)
        return [r for r in self.records if isinstance(r.module, self.include_types) or r.own_params > 0]

    def _shape_cell(self, shape) -> Optional[str]:
        """Returns a shape rendered for a table cell, or None when the layer was never visited."""
        return str(shape) if shape is not None else None

    def _node_label(self, record: LayerRecord) -> str:
        """Returns the console-tree label of a layer with its shapes and own parameter count."""
        leaf  = record.name.rpartition(".")[2]
        label = f"[key]{leaf}[/key] [muted]{record.type_name}[/muted]"

        if record.visited:
            label += f"  {record.in_shape} [muted]->[/muted] {record.out_shape}"
        if record.own_params > 0:
            label += f"  [value]{record.own_params:,}[/value]"

        return label

    def render_console(self) -> None:
        """Renders the module hierarchy as a tree and logs the parameter totals."""
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
        """Builds the markdown document with the parameter totals and the per-layer table.

        Args:
            title : Document title.

        Returns:
            The assembled document, one row per selected layer with its share of all parameters.
        """
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
        """Writes the model document into the documentation directory and returns its path."""
        path = self.to_markdown(title=title).save(self.docs_dir / filename)
        self.logger.subsection(f"Model documentation saved to {path}")
        return path


class TrainingDocs:
    """Emits the model documentation of a training run from its first real batch.

    Attributes:
        model     : Model being documented.
        model_cfg : Model configuration supplying the module types kept in the table.
        logger    : Logger receiving the section heading and the console tree.
        run_dir   : Run directory whose docs subdirectory receives the markdown file.
        enabled   : Whether documentation is emitted at all.
    """

    def __init__(self, model, model_cfg, logger, run_dir, enabled=True):
        """Stores the model, its configuration, the logger, the run directory and the enable flag."""
        self.model     = model
        self.model_cfg = model_cfg
        self.logger    = logger
        self.run_dir   = Path(run_dir)
        self.enabled   = enabled

    @torch.no_grad()
    def emit(self, data_loader: DataLoader, device: torch.device) -> None:
        """Traces the model on the loader's first batch and writes the console and markdown docs.

        Args:
            data_loader : Loader whose first batch supplies the tracing sample.
            device      : Device the sample is moved to before the forward pass.
        """
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
