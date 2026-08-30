"""Capture and analysis of the attention a trained backbone puts on its inputs.

Collects spatial attention gates, the weights of the shared attention blocks and of
any torch MultiheadAttention module, then summarizes their entropy, per-head
redundancy and attention reach and renders the figures and markdown report.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.blocks                                   import AttentionTap
from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class AttentionCapture:
    """Hooks a model to record its attention gates and attention weight tensors.

    Gates are captured with forward hooks, the shared attention blocks through the
    module-level AttentionTap, and torch MultiheadAttention modules by temporarily
    replacing their forward so the weights are returned per head.

    Attributes:
        model : Model being instrumented.
    """

    GATE_SUFFIX = "attention_score"
    TAP_TYPES   = ("MultiHeadSelfAttention", "WindowAttention")

    def __init__(self, model: torch.nn.Module) -> None:
        """Prepares the empty hook, patch and record stores.

        Args:
            model : Model to instrument.
        """
        self.model = model

        self._handles     = []
        self._patched     = []
        self._tap_sink    = []
        self._tap_order   = []
        self._gate_maps   = {}
        self._mha_records = {}

    def _find_modules(self) -> tuple[dict, dict, dict]:
        """Returns the attention gate, tapped-block and MultiheadAttention modules by name."""
        gates = {name: m for name, m in self.model.named_modules() if name.endswith(self.GATE_SUFFIX)}
        taps  = {name: m for name, m in self.model.named_modules() if type(m).__name__ in self.TAP_TYPES}
        mhas  = {name: m for name, m in self.model.named_modules() if isinstance(m, torch.nn.MultiheadAttention)}

        return gates, taps, mhas

    def _gate_hook(self, name: str):
        """Returns a forward hook storing a gate's output map of shape (batch, 1, h, w)."""
        def hook(module, args, output):
            self._gate_maps.setdefault(name, []).append(output.detach().cpu())
        return hook

    def _tap_pre_hook(self, name: str):
        """Returns a forward pre-hook recording the call order of the tapped attention blocks."""
        def hook(module, args):
            self._tap_order.append(name)
        return hook

    def _patch_mha(self, name: str, module: torch.nn.MultiheadAttention) -> None:
        """Replaces a MultiheadAttention forward so it returns and records per-head weights.

        Args:
            name   : Qualified module name used as the record key.
            module : MultiheadAttention module to patch; restored by detach().
        """
        original = module.forward

        def wrapped(query, key, value, **kwargs):
            kwargs["need_weights"]         = True
            kwargs["average_attn_weights"] = False

            out, weights = original(query, key, value, **kwargs)
            self._mha_records.setdefault(name, []).append(weights.detach().cpu())

            return out, weights

        module.forward = wrapped
        self._patched.append(module)

    def attach(self) -> None:
        """Installs the gate hooks, the attention tap and the MultiheadAttention patches.

        Raises:
            ValueError : If the model exposes no attention of any capturable kind.
        """
        gates, taps, mhas = self._find_modules()

        if not gates and not taps and not mhas:
            raise ValueError(f"Model {type(self.model).__name__} has no capturable attention: no attention gates, no shared-block attention, no torch MultiheadAttention")

        AttentionTap.enable(self._tap_sink)

        for name, module in gates.items():
            self._handles.append(module.register_forward_hook(self._gate_hook(name)))

        for name, module in taps.items():
            self._handles.append(module.register_forward_pre_hook(self._tap_pre_hook(name)))

        for name, module in mhas.items():
            self._patch_mha(name, module)

    def detach(self) -> None:
        """Removes every hook and restores the patched MultiheadAttention forwards."""
        AttentionTap.disable()

        for handle in self._handles:
            handle.remove()
        self._handles = []

        for module in self._patched:
            del module.forward
        self._patched = []

    def records(self) -> dict:
        """Returns the captured attention, labelled by module.

        Returns:
            Dictionary with "gates", mapping each gate name to its concatenated maps of
            shape (calls * batch, 1, h, w), and "attention", mapping each attention module
            name to the list of its weight tensors of shape (batch, heads, queries, keys).

        Raises:
            RuntimeError : If the tap collected a different number of weight tensors than the number of tapped module calls, so the capture cannot be labelled.
        """
        if len(self._tap_order) != len(self._tap_sink):
            raise RuntimeError(f"Attention tap saw {len(self._tap_sink)} weight tensors but {len(self._tap_order)} module calls; the capture cannot be labelled")

        attention: dict[str, list] = {}
        for name, weights in zip(self._tap_order, self._tap_sink):
            attention.setdefault(name, []).append(weights)

        for name, weight_list in self._mha_records.items():
            attention.setdefault(name, []).extend(weight_list)

        gates = {name: torch.cat(maps, dim=0) for name, maps in self._gate_maps.items()}
        return {"gates": gates, "attention": attention}


class AttentionSummary:
    """Scalar and map summaries of captured attention weights."""

    @staticmethod
    def stack_calls(weight_list: list[torch.Tensor]) -> torch.Tensor:
        """Concatenates the calls of one attention module along the batch axis.

        Args:
            weight_list : Weight tensors of shape (batch, heads, queries, keys), one per call.

        Returns:
            Tensor of shape (total_batch, heads, queries, keys).

        Raises:
            ValueError : If the calls disagree on their head, query or key extent, or if the weights are not four-dimensional.
        """
        shapes = {tuple(w.shape[1:]) for w in weight_list}
        if len(shapes) != 1:
            raise ValueError(f"Attention calls of one module disagree on shape {sorted(shapes)}; the capture cannot be pooled")

        weights = torch.cat(weight_list, dim=0)
        if weights.ndim != 4:
            raise ValueError(f"Expected attention weights of shape (batch, heads, queries, keys), got {tuple(weights.shape)}")

        return weights

    @staticmethod
    def entropy_values(weights: torch.Tensor) -> np.ndarray:
        """Returns the log(keys)-normalized entropy of every query distribution.

        Args:
            weights : Attention weights of shape (batch, heads, queries, keys).

        Returns:
            Flat array of shape (batch * heads * queries,), 1 for uniform attention and
            0 for a single attended token.
        """
        w   = weights.clamp_min(1e-12)
        ent = -(w * w.log()).sum(dim=-1) / math.log(w.shape[-1])

        return ent.reshape(-1).numpy()

    @staticmethod
    def head_entropies(weights: torch.Tensor) -> np.ndarray:
        """Returns the normalized attention entropy of each head.

        Args:
            weights : Attention weights of shape (batch, heads, queries, keys).

        Returns:
            Array of shape (heads,) averaged over batch and queries.
        """
        w   = weights.clamp_min(1e-12)
        ent = -(w * w.log()).sum(dim=-1) / math.log(w.shape[-1])

        return ent.mean(dim=(0, 2)).numpy()

    @staticmethod
    def head_redundancy(weights: torch.Tensor) -> float | None:
        """Returns the mean off-diagonal correlation between the heads' attention maps.

        Args:
            weights : Attention weights of shape (batch, heads, queries, keys).

        Returns:
            The mean correlation, 1 when all heads attend identically, or None when the
            module has a single head or the correlation is not finite.
        """
        heads = weights.shape[1]
        if heads < 2:
            return None

        flat        = weights.permute(1, 0, 2, 3).reshape(heads, -1).numpy()
        correlation = np.corrcoef(flat)
        off_diag    = correlation[~np.eye(heads, dtype=bool)]

        value = float(off_diag.mean())
        return value if np.isfinite(value) else None

    @staticmethod
    def peak(weights: torch.Tensor) -> float:
        """Returns the mean largest attention weight a query assigns to a single key."""
        return float(weights.amax(dim=-1).mean())

    @staticmethod
    def token_grid_side(n_tokens: int) -> int | None:
        """Returns the side of the square token grid, or None when the count is not a square."""
        side = int(round(math.sqrt(n_tokens)))
        return side if side * side == n_tokens else None

    @classmethod
    def distance_matrix(cls, side: int) -> np.ndarray:
        """Returns the pairwise Euclidean token distances on a square grid.

        Args:
            side : Side of the token grid in tokens.

        Returns:
            Array of shape (side², side²) holding the distance in tokens between every
            pair of grid positions.
        """
        coords = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float64)
        return np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))

    @classmethod
    def attention_distance(cls, weights: torch.Tensor) -> dict | None:
        """Measures how far attention reaches on the token grid.

        Args:
            weights : Attention weights of shape (batch, heads, queries, keys).

        Returns:
            Dictionary with the attention-weighted mean query-key distance in "tokens",
            the same divided by the grid side as "relative", and the corresponding
            "uniform_tokens" and "uniform_relative" references for uniform attention.
            None when the keys do not form a square grid or queries and keys differ.
        """
        queries, keys = weights.shape[-2], weights.shape[-1]
        side          = cls.token_grid_side(keys)

        if side is None or queries != keys:
            return None

        distances = cls.distance_matrix(side)
        averaged  = weights.mean(dim=(0, 1)).numpy()
        measured  = float((averaged * distances).sum(axis=1).mean())
        uniform   = float(distances.mean())

        return {
            "tokens"           : measured,
            "relative"         : measured / side,
            "uniform_tokens"   : uniform,
            "uniform_relative" : uniform / side,
        }

    @classmethod
    def mean_attention_map(cls, weights: torch.Tensor) -> np.ndarray | None:
        """Averages the attention received by each key and folds it back onto the token grid.

        Args:
            weights : Attention weights of shape (batch, heads, queries, keys).

        Returns:
            Array of shape (side, side) with the mean weight each key receives, or None
            when the keys do not form a square grid.
        """
        side = cls.token_grid_side(weights.shape[-1])
        if side is None:
            return None

        return weights.mean(dim=(0, 1, 2)).numpy().reshape(side, side)

    @staticmethod
    def gate_stats(gate_map: torch.Tensor) -> dict:
        """Summarizes a spatial attention gate.

        Args:
            gate_map : Gate activations of shape (batch, channels, azimuth, range).

        Returns:
            Dictionary with the "mean" and "std" gate weight, the "frac_active" fraction
            of positions above 0.5, and the "spatial_shape" of the map.
        """
        values = gate_map.float()

        return {
            "mean"          : float(values.mean()),
            "std"           : float(values.std()),
            "frac_active"   : float((values > 0.5).float().mean()),
            "spatial_shape" : list(values.shape[-2:]),
        }


class AttentionCapturePlots(PlotBase):
    """Figures of the captured attention gates and attention weights."""

    @staticmethod
    def _short(name: str) -> str:
        """Returns a module name shortened to its last few dotted components for axis labels."""
        trimmed = name.removesuffix(".attention")
        parts   = trimmed.split(".")
        return ".".join(parts[-5:]) if len(parts) > 5 else trimmed

    def gate_map(self, gate: np.ndarray, name: str, stats: dict, path: Path) -> Path:
        """Draws the batch-mean map of one attention gate.

        Args:
            gate  : Mean gate weights of shape (azimuth, range), on the gate's own grid.
            name  : Qualified module name of the gate.
            stats : Gate statistics used for the overlay text.
            path  : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        return self._imshow_figure(
            gate,
            x_label        = "Col [px, gate scale]",
            y_label        = "Row [px, gate scale]",
            title          = f"Attention gate: {name} (mean over batch)",
            cmap           = self._cmap_with_bad("viridis"),
            vmin           = 0.0,
            vmax           = 1.0,
            colorbar_label = "Gate weight",
            text_overlay   = f"mean = {stats['mean']:.3f}\nstd = {stats['std']:.3f}\nactive (>0.5) = {stats['frac_active'] * 100.0:.1f}%",
            path           = path,
        )

    def entropy_by_layer(self, names: list[str], summary: dict, path: Path) -> Path:
        """Draws the median attention entropy per layer with its p10-p90 spread.

        Args:
            names   : Attention module names in capture order.
            summary : Per-module attention summary holding the entropy percentiles.
            path    : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        depth = np.arange(len(names))
        p10   = np.array([summary[name]["entropy_p10"] for name in names])
        p50   = np.array([summary[name]["entropy_p50"] for name in names])
        p90   = np.array([summary[name]["entropy_p90"] for name in names])

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.fill_between(depth, p10, p90, color=self.OKABE_ITO[0], alpha=0.25, label="p10–p90 across queries")
        ax.plot(depth, p50, marker="o", color=self.OKABE_ITO[0], linewidth=1.4, label="median")
        ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0, label="uniform attention")

        ax.set_xticks(depth)
        ax.set_xticklabels([self._short(name) for name in names], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Normalized attention entropy")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Attention spread by layer (1 = uniform, 0 = single token)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)

    def head_entropy_heatmap(self, names: list[str], summary: dict, path: Path) -> Path:
        """Draws a layer-by-head heatmap of the normalized attention entropy.

        Args:
            names   : Attention module names in capture order, one heatmap row each.
            summary : Per-module attention summary holding the per-head entropies.
            path    : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        max_heads = max(len(summary[name]["head_entropy"]) for name in names)
        matrix    = np.full((len(names), max_heads), np.nan)
        for i, name in enumerate(names):
            entropies                       = summary[name]["head_entropy"]
            matrix[i, : len(entropies)]     = entropies

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=max(0.35, 0.09 * len(names))))
        im      = ax.imshow(matrix, cmap=self._cmap_with_bad("viridis"), vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")

        ax.set_xticks(range(max_heads))
        ax.set_xticklabels([str(h) for h in range(max_heads)], fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([self._short(name) for name in names], fontsize=7)

        for i in range(len(names)):
            for j in range(max_heads):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if matrix[i, j] < 0.6 else "black")

        ax.set_xlabel("Head")
        ax.set_title("Per-head attention entropy (1 = uniform, 0 = single token)")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("Normalized entropy")
        fig.tight_layout()

        return self._save(fig, path)

    def distance_by_layer(self, names: list[str], summary: dict, path: Path) -> Path:
        """Draws the relative attention reach per layer against the uniform-attention reference.

        Args:
            names   : Attention module names in capture order.
            summary : Per-module attention summary holding the distance entries.
            path    : Output path of the figure.

        Returns:
            Path of the saved figure.

        Raises:
            ValueError : If no layer sits on a square token grid, leaving nothing to draw.
        """
        self._apply_style()

        entries = [(i, summary[name]["distance"]) for i, name in enumerate(names) if summary[name]["distance"] is not None]
        if not entries:
            raise ValueError("No attention layer sits on a square token grid; the attention-distance figure has nothing to draw")

        depth    = np.array([i for i, _entry in entries])
        measured = np.array([entry["relative"] for _i, entry in entries])
        uniform  = np.array([entry["uniform_relative"] for _i, entry in entries])

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(depth, measured, marker="o", color=self.OKABE_ITO[1], linewidth=1.4, label="measured")
        ax.plot(depth, uniform, marker=".", color="0.4", linewidth=1.0, linestyle="--", label="uniform attention")

        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([self._short(name) for name in names], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Mean attention distance / grid side")
        ax.set_ylim(0.0, None)
        ax.set_title("Attention reach by layer")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)

    def mean_attention_map(self, attention: np.ndarray, name: str, entropy: float, path: Path) -> Path:
        """Draws the mean attention received by each key on the token grid.

        Args:
            attention : Mean key attention of shape (side, side).
            name      : Qualified module name of the attention layer.
            entropy   : Mean normalized entropy of the layer, shown as an overlay.
            path      : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        return self._imshow_figure(
            attention,
            x_label        = "Key column [tokens]",
            y_label        = "Key row [tokens]",
            title          = f"Mean attention over keys: {name}",
            cmap           = self._cmap_with_bad("magma"),
            vmin           = 0.0,
            vmax           = float(attention.max()),
            colorbar_label = "Mean attention weight",
            text_overlay   = f"entropy = {entropy:.3f}",
            path           = path,
        )


class AttentionCaptureRun(AnalysisRun):
    """Captures, summarizes and reports the attention of a single training run."""

    SUMMARY_FILENAME = "attention_capture.json"
    REPORT_FILENAME  = "attention_capture.md"

    def _capture(self, run) -> dict:
        """Runs the model over the configured batches with the attention capture attached.

        Args:
            run : Loaded run bundling the model and the dataloader.

        Returns:
            The captured gate maps and attention weights, as returned by the capture.
        """
        capture = AttentionCapture(run.model.module)
        capture.attach()

        try:
            with torch.no_grad():
                for index, batch in enumerate(run.loader):
                    if index >= self.config.max_batches:
                        break
                    run.model(batch[0])
        finally:
            capture.detach()

        return capture.records()

    def _summarize(self, records: dict) -> tuple[dict, list[str]]:
        """Reduces the captured tensors to per-module scalar summaries.

        Args:
            records : Gate maps and attention weights returned by the capture.

        Returns:
            A pair of the summary, holding a "gates" and an "attention" section keyed by
            module name, and the attention module names in capture order.
        """
        gates = {name: AttentionSummary.gate_stats(gate) for name, gate in records["gates"].items()}

        attention = {}
        order     = list(records["attention"])
        for name in order:
            weights   = AttentionSummary.stack_calls(records["attention"][name])
            entropies = AttentionSummary.entropy_values(weights)

            attention[name] = {
                "calls"           : len(records["attention"][name]),
                "n_tokens"        : int(weights.shape[-1]),
                "heads"           : int(weights.shape[1]),
                "entropy"         : float(entropies.mean()),
                "entropy_p10"     : float(np.percentile(entropies, 10)),
                "entropy_p50"     : float(np.percentile(entropies, 50)),
                "entropy_p90"     : float(np.percentile(entropies, 90)),
                "peak"            : AttentionSummary.peak(weights),
                "head_entropy"    : AttentionSummary.head_entropies(weights).tolist(),
                "head_redundancy" : AttentionSummary.head_redundancy(weights),
                "distance"        : AttentionSummary.attention_distance(weights),
            }

        return {"gates": gates, "attention": attention}, order

    def _render_figures(self, records: dict, summary: dict, order: list[str]) -> dict[str, Path]:
        """Renders the gate maps, the entropy and reach profiles and the most focused attention maps.

        Args:
            records : Captured gate maps and attention weights.
            summary : Per-module summary of the capture.
            order   : Attention module names in capture order.

        Returns:
            Figure paths keyed by figure name.
        """
        plots   = AttentionCapturePlots()
        figures = {}

        for index, (name, gate) in enumerate(records["gates"].items()):
            if index >= self.config.max_gate_figures:
                break
            safe_name = name.replace(".", "_")
            mean_map  = gate.float().mean(dim=(0, 1)).numpy()
            figures[f"gate_{safe_name}"] = plots.gate_map(mean_map, name, summary["gates"][name], self.output_dir / "plots" / f"gate_{safe_name}.png")

        if summary["attention"]:
            figures["entropy_by_layer"]     = plots.entropy_by_layer(order, summary["attention"], self.output_dir / "plots" / "entropy_by_layer.png")
            figures["head_entropy_heatmap"] = plots.head_entropy_heatmap(order, summary["attention"], self.output_dir / "plots" / "head_entropy_heatmap.png")

            if any(summary["attention"][name]["distance"] is not None for name in order):
                figures["distance_by_layer"] = plots.distance_by_layer(order, summary["attention"], self.output_dir / "plots" / "distance_by_layer.png")

            focused = sorted(order, key=lambda name: summary["attention"][name]["entropy"])
            for name in focused[: self.config.max_attention_figures]:
                weights  = AttentionSummary.stack_calls(records["attention"][name])
                mean_map = AttentionSummary.mean_attention_map(weights)
                if mean_map is not None:
                    safe_name = name.replace(".", "_")
                    figures[f"attention_{safe_name}"] = plots.mean_attention_map(mean_map, name, summary["attention"][name]["entropy"], self.output_dir / "plots" / f"attention_{safe_name}.png")

        return figures

    def _write_report(self, run, summary: dict, order: list[str], figures: dict[str, Path]) -> Path:
        """Writes the markdown report of the capture.

        Args:
            run     : Loaded run, used for the backbone name in the title.
            summary : Per-module summary of the capture.
            order   : Attention module names in capture order.
            figures : Figure paths keyed by figure name.

        Returns:
            Path of the written report.
        """
        doc = MarkdownDoc(title=f"Attention capture: {run.backbone_name}")
        doc.paragraph(
            f"Attention gates and attention weights captured on {self.config.max_batches} real '{self.config.split}' batches of {self.config.batch_size} patches. "
            "Entropy is normalized by log(tokens): 1 means uniform attention, 0 a single attended token; the p10–p90 spread is taken across queries, heads and batches. "
            "Head redundancy is the mean correlation between the attention maps of different heads (1 = all heads identical). "
            "Attention distance is the attention-weighted mean query-key distance on the token grid, divided by the grid side, next to the uniform-attention reference; "
            "it is only defined for layers whose keys form a square token grid."
        )

        if summary["gates"]:
            doc.heading("Attention gates", level=2)
            table = MarkdownTable(("Gate", "Mean weight", "Std", "Fraction > 0.5", "Map shape"))
            for name, stats in summary["gates"].items():
                table.add_row(f"`{name}`", f"{stats['mean']:.3f}", f"{stats['std']:.3f}", f"{stats['frac_active'] * 100.0:.1f}%", "x".join(str(v) for v in stats["spatial_shape"]))
            doc.table(table)

        if summary["attention"]:
            doc.heading("Attention weights", level=2)
            table = MarkdownTable(("Layer", "Calls", "Tokens", "Heads", "Entropy p50", "p10–p90", "Peak", "Head redundancy", "Rel. distance"))
            for name in order:
                stats      = summary["attention"][name]
                redundancy = f"{stats['head_redundancy']:.3f}" if stats["head_redundancy"] is not None else "n/a"
                distance   = f"{stats['distance']['relative']:.3f}" if stats["distance"] is not None else "n/a"
                table.add_row(f"`{name}`", str(stats["calls"]), str(stats["n_tokens"]), str(stats["heads"]), f"{stats['entropy_p50']:.3f}", f"{stats['entropy_p10']:.2f}–{stats['entropy_p90']:.2f}", f"{stats['peak']:.3f}", redundancy, distance)
            doc.table(table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        """Captures the attention of this run, plots it and writes the report.

        Returns:
            The summary with its "gates" and "attention" sections, as written to JSON
            alongside the capture order.
        """
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run            = self._load_run()
        records        = self._capture(run)
        summary, order = self._summarize(records)

        figures = self._render_figures(records, summary, order)

        FileIO.save_json({**summary, "layer_order": order}, self.output_dir / self.SUMMARY_FILENAME)
        report_path = self._write_report(run, summary, order, figures)

        self.logger.ok(f"{self.run_dir.name}: {len(summary['gates'])} gates, {len(summary['attention'])} attention layers -> {report_path}")

        return summary


class AttentionCaptureBatch(RunBatch):
    """Runs the attention capture over every run matching the configured filter."""

    SELECTOR_ACTION = "capture"
    SECTION_TITLE   = "Attention capture"
    RUN_CLASS       = AttentionCaptureRun
