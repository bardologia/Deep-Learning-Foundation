from __future__ import annotations

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

    GATE_SUFFIX = "attention_score"
    TAP_TYPES   = ("MultiHeadSelfAttention", "WindowAttention")

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

        self._handles     = []
        self._patched     = []
        self._tap_sink    = []
        self._tap_order   = []
        self._gate_maps   = {}
        self._mha_records = {}

    def _find_modules(self) -> tuple[dict, dict, dict]:
        gates = {name: m for name, m in self.model.named_modules() if name.endswith(self.GATE_SUFFIX)}
        taps  = {name: m for name, m in self.model.named_modules() if type(m).__name__ in self.TAP_TYPES}
        mhas  = {name: m for name, m in self.model.named_modules() if isinstance(m, torch.nn.MultiheadAttention)}

        return gates, taps, mhas

    def _gate_hook(self, name: str):
        def hook(module, args, output):
            self._gate_maps[name] = output.detach().cpu()
        return hook

    def _tap_pre_hook(self, name: str):
        def hook(module, args):
            self._tap_order.append(name)
        return hook

    def _patch_mha(self, name: str, module: torch.nn.MultiheadAttention) -> None:
        original = module.forward

        def wrapped(query, key, value, **kwargs):
            kwargs["need_weights"]         = True
            kwargs["average_attn_weights"] = True

            out, weights = original(query, key, value, **kwargs)
            self._mha_records.setdefault(name, []).append(weights.detach().cpu())

            return out, weights

        module.forward = wrapped
        self._patched.append(module)

    def attach(self) -> None:
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
        AttentionTap.disable()

        for handle in self._handles:
            handle.remove()
        self._handles = []

        for module in self._patched:
            del module.forward
        self._patched = []

    def records(self) -> dict:
        if len(self._tap_order) != len(self._tap_sink):
            raise RuntimeError(f"Attention tap saw {len(self._tap_sink)} weight tensors but {len(self._tap_order)} module calls; the capture cannot be labelled")

        attention: dict[str, list] = {}
        for name, weights in zip(self._tap_order, self._tap_sink):
            attention.setdefault(name, []).append(weights)

        for name, weight_list in self._mha_records.items():
            attention.setdefault(name, []).extend(weight_list)

        return {"gates": dict(self._gate_maps), "attention": attention}


class AttentionSummary:

    @staticmethod
    def entropy(weights: torch.Tensor) -> float:
        w    = weights.clamp_min(1e-12)
        n_kv = w.shape[-1]
        ent  = -(w * w.log()).sum(dim=-1)

        return float(ent.mean() / np.log(n_kv))

    @staticmethod
    def peak(weights: torch.Tensor) -> float:
        return float(weights.amax(dim=-1).mean())

    @staticmethod
    def gate_stats(gate_map: torch.Tensor) -> dict:
        values = gate_map.float()

        return {
            "mean"          : float(values.mean()),
            "frac_active"   : float((values > 0.5).float().mean()),
            "spatial_shape" : list(values.shape[-2:]),
        }


class AttentionCapturePlots(PlotBase):

    def gate_map(self, gate: np.ndarray, name: str, path: Path) -> Path:
        return self._imshow_figure(
            gate,
            x_label        = "Range [px, gate scale]",
            y_label        = "Azimuth [px, gate scale]",
            title          = f"Attention gate: {name}",
            cmap           = self._cmap_with_bad("viridis"),
            vmin           = 0.0,
            vmax           = 1.0,
            colorbar_label = "Gate weight",
            path           = path,
        )

    def entropy_bars(self, names: list[str], entropies: list[float], path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.75))
        ax.bar(range(len(names)), entropies, color="#0072B2")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name for name in names], rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("Normalized attention entropy")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Attention spread by layer (1 = uniform, 0 = single token)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class AttentionCaptureRun(AnalysisRun):

    SUMMARY_FILENAME = "attention_capture.json"
    REPORT_FILENAME  = "attention_capture.md"

    def _capture(self, run) -> dict:
        capture = AttentionCapture(run.model.module)
        capture.attach()

        try:
            with torch.no_grad():
                run.model(next(iter(run.loader))[0])
        finally:
            capture.detach()

        return capture.records()

    def _summarize(self, records: dict) -> dict:
        gates = {name: AttentionSummary.gate_stats(gate) for name, gate in records["gates"].items()}

        attention = {}
        for name, weight_list in records["attention"].items():
            attention[name] = {
                "calls"    : len(weight_list),
                "entropy"  : float(np.mean([AttentionSummary.entropy(w) for w in weight_list])),
                "peak"     : float(np.mean([AttentionSummary.peak(w) for w in weight_list])),
                "n_tokens" : int(weight_list[0].shape[-1]),
            }

        return {"gates": gates, "attention": attention}

    def _render_figures(self, records: dict, summary: dict) -> dict[str, Path]:
        plots   = AttentionCapturePlots()
        figures = {}

        for index, (name, gate) in enumerate(records["gates"].items()):
            if index >= self.config.max_gate_figures:
                break
            safe_name = name.replace(".", "_")
            figures[f"gate_{safe_name}"] = plots.gate_map(gate[0, 0].numpy(), name, self.output_dir / "plots" / f"gate_{safe_name}.png")

        if summary["attention"]:
            names     = sorted(summary["attention"])
            entropies = [summary["attention"][name]["entropy"] for name in names]
            figures["entropy_by_layer"] = plots.entropy_bars(names, entropies, self.output_dir / "plots" / "entropy_by_layer.png")

        return figures

    def _write_report(self, run, summary: dict, figures: dict[str, Path]) -> Path:
        doc = MarkdownDoc(title=f"Attention capture: {run.backbone_name}")
        doc.paragraph(f"Attention gates and attention weights captured on one real '{self.config.split}' batch of {self.config.batch_size} patches.")

        if summary["gates"]:
            doc.heading("Attention gates", level=2)
            table = MarkdownTable(("Gate", "Mean weight", "Fraction > 0.5", "Map shape"))
            for name, stats in summary["gates"].items():
                table.add_row(f"`{name}`", f"{stats['mean']:.3f}", f"{stats['frac_active'] * 100.0:.1f}%", "x".join(str(v) for v in stats["spatial_shape"]))
            doc.table(table)

        if summary["attention"]:
            doc.heading("Attention weights", level=2)
            table = MarkdownTable(("Layer", "Calls", "Tokens", "Entropy", "Mean peak"))
            for name in sorted(summary["attention"]):
                stats = summary["attention"][name]
                table.add_row(f"`{name}`", str(stats["calls"]), str(stats["n_tokens"]), f"{stats['entropy']:.3f}", f"{stats['peak']:.3f}")
            doc.table(table)

        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run     = self._load_run()
        records = self._capture(run)
        summary = self._summarize(records)

        figures = self._render_figures(records, summary)

        FileIO.save_json(summary, self.output_dir / self.SUMMARY_FILENAME)
        report_path = self._write_report(run, summary, figures)

        self.logger.ok(f"{self.run_dir.name}: {len(summary['gates'])} gates, {len(summary['attention'])} attention layers -> {report_path}")

        return summary


class AttentionCaptureBatch(RunBatch):

    SELECTOR_ACTION = "capture"
    SECTION_TITLE   = "Attention capture"
    RUN_CLASS       = AttentionCaptureRun
