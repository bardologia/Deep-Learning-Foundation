from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisOutcome, AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import ModelDevice, ProbeWindows
from tools.data.io                                   import FileIO
from tools.metrics.scoring                           import RankCorrelation
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class GradientAttribution:

    def __init__(self, model, window: int, target_names: list[str]) -> None:
        self.model        = model
        self.window       = int(window)
        self.target_names = list(target_names)

    def channel_importance(self, windows: torch.Tensor) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, int]]]:
        half       = self.window // 2
        device     = ModelDevice.of(self.model)
        n_channels = windows.shape[1]

        grad_shares = {name: [] for name in self.target_names}
        gxi_shares  = {name: [] for name in self.target_names}

        for p in range(windows.shape[0]):
            x = windows[p:p + 1].clone().to(device).requires_grad_(True)
            y = self.model(x)

            for index, name in enumerate(self.target_names):
                grad = torch.autograd.grad(y[0, index, half, half], x, retain_graph=True)[0]

                grad_mass = grad.abs().sum(dim=(0, 2, 3)).cpu().numpy().astype(np.float64)
                gxi_mass  = (grad * x).abs().sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64)

                if grad_mass.sum() > 0.0:
                    grad_shares[name].append(grad_mass / grad_mass.sum())
                if gxi_mass.sum() > 0.0:
                    gxi_shares[name].append(gxi_mass / gxi_mass.sum())

        importance = {}
        for name in self.target_names:
            stacked      = np.stack(grad_shares[name]) if grad_shares[name] else np.full((1, n_channels), np.nan)
            grad_x_input = np.stack(gxi_shares[name]).mean(axis=0) if gxi_shares[name] else np.full(n_channels, np.nan)

            importance[name] = {"share": stacked.mean(axis=0), "share_std": stacked.std(axis=0), "grad_x_input": grad_x_input}

        kept = {name: {"share": len(grad_shares[name]), "grad_x_input": len(gxi_shares[name])} for name in self.target_names}

        return importance, kept

    @staticmethod
    def combined(importance: dict[str, dict[str, np.ndarray]], key: str) -> np.ndarray:
        stacked = np.stack([entry[key] for entry in importance.values()])
        live    = ~np.all(np.isnan(stacked), axis=1)

        return np.nanmean(stacked[live], axis=0)


class ChannelOcclusion:

    def __init__(self, model, on_pass) -> None:
        self.model   = model
        self.on_pass = on_pass

    def deltas(self, loader, max_batches: int, n_channels: int) -> np.ndarray:
        totals = np.zeros(n_channels, dtype=np.float64)
        count  = 0
        device = ModelDevice.of(self.model)

        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break

                images = batch[0].to(device)
                base   = self.model(images)

                for channel in range(n_channels):
                    occluded             = images.clone()
                    occluded[:, channel] = 0.0
                    totals[channel]     += float(((self.model(occluded) - base) ** 2).mean())
                    self.on_pass()

                count += 1

        return totals / count


class InputAttributionPlots(PlotBase):

    def channel_bars(self, values: np.ndarray, stds: np.ndarray | None, labels: list[str], xlabel: str, title: str, path: Path, percent: bool = True) -> Path:
        self._apply_style()

        finite = np.where(np.isfinite(values), values, 0.0)
        order  = np.argsort(finite)
        scale  = 100.0 if percent else 1.0
        errors = np.where(np.isfinite(stds[order]), stds[order], 0.0) * scale if stds is not None else None

        fig, ax = plt.subplots(figsize=(self.FULL_WIDTH * 1.15, max(2.4, 0.30 * len(labels) + 1.2)))
        ax.barh(range(len(order)), finite[order] * scale, xerr=errors, color=self.OKABE_ITO[0], error_kw={"elinewidth": 0.9, "ecolor": "0.3"})
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=8)

        span = float(finite.max() * scale) if finite.max() > 0 else 1.0
        for row, i in enumerate(order):
            text = "n/a" if not np.isfinite(values[i]) else (f"{values[i] * scale:.1f}%" if percent else f"{values[i]:.3g}")
            ax.text(finite[i] * scale + 0.01 * span, row, text, va="center", fontsize=7, color="0.25")

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(0.0, span * 1.14)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)

    def agreement_scatter(self, gradient: np.ndarray, occlusion: np.ndarray, labels: list[str], rho: float, path: Path) -> Path:
        self._apply_style()

        grad_rank = RankCorrelation.ranks(gradient)
        occl_rank = RankCorrelation.ranks(occlusion)

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.85))
        ax.scatter(grad_rank, occl_rank, s=26, color=self.OKABE_ITO[0], edgecolors="white", linewidths=0.5)
        ax.plot([0, len(labels) - 1], [0, len(labels) - 1], color="0.4", linestyle="--", linewidth=0.9, label="perfect agreement")

        for i in np.argsort(occlusion)[-3:]:
            ax.annotate(labels[i], xy=(grad_rank[i], occl_rank[i]), xytext=(4, 4), textcoords="offset points", fontsize=7, color="0.25")

        ax.set_xlabel("Channel rank by gradient attribution (0 = weakest)")
        ax.set_ylabel("Channel rank by occlusion impact (0 = weakest)")
        ax.set_title(f"Gradient and occlusion rank agreement (Spearman rho = {rho:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)


class InputAttributionRun(AnalysisRun):

    SUMMARY_FILENAME = "input_attribution.json"
    REPORT_FILENAME  = "input_attribution.md"

    def _gradient_importance(self, run) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
        probes  = ProbeWindows(run, self.config.window)
        centers = probes.centers(self.config.n_row_probes, self.config.n_col_probes)
        windows = probes.assemble(centers)

        with self.logger.timer(f"Input attribution: gradient pass ({len(centers)} probe windows)"):
            importance, kept = GradientAttribution(run.model.module, self.config.window, run.target_channels).channel_importance(windows)

        self._warn_dropped_probes(kept, len(centers))

        return importance, {"n_probes": len(centers), "kept": kept}

    def _warn_dropped_probes(self, kept: dict[str, dict[str, int]], n_probes: int) -> None:
        for name, counts in kept.items():
            if counts["share"] < n_probes or counts["grad_x_input"] < n_probes:
                self.logger.warning(f"{self.run_dir.name}: gradient attribution of {name}: {n_probes - counts['share']}/{n_probes} probes had a zero gradient (share over {counts['share']}, gradient x input over {counts['grad_x_input']} probes)")

    def _occlusion(self, run) -> np.ndarray | None:
        if not self.config.occlude:
            return None

        n_batches = min(self.config.max_batches, len(run.loader))
        n_passes  = run.in_channels * n_batches

        self.logger.info(f"{self.run_dir.name}: occluding {run.in_channels} channels over {n_batches} batches ({n_passes} forward passes)")

        with self.logger.timer(f"Input attribution: occlusion ({n_passes} passes)"), self.logger.track() as progress:
            task      = progress.add_task("[section]Channel occlusion[/section]", total=n_passes)
            occlusion = ChannelOcclusion(run.model.module, functools.partial(progress.advance, task))

            return occlusion.deltas(run.loader, self.config.max_batches, run.in_channels)

    def _render_figures(self, labels: list[str], importance: dict, occlusion: np.ndarray | None, spearman: float | None) -> dict[str, Path]:
        plots   = InputAttributionPlots()
        figures = {}

        for name, entry in importance.items():
            figures[f"grad_{name}"] = plots.channel_bars(entry["share"], entry["share_std"], labels, xlabel="Gradient attribution share (%, with the std across probes)", title=f"Input-channel attribution of target '{name}'", path=self.output_dir / "plots" / f"grad_{name}.png")

        figures["grad_x_input"] = plots.channel_bars(GradientAttribution.combined(importance, "grad_x_input"), None, labels, xlabel="Gradient x input attribution share (%, mean over target channels)", title="Input-channel attribution weighted by the input signal", path=self.output_dir / "plots" / "grad_x_input.png")

        if occlusion is not None:
            figures["occlusion"] = plots.channel_bars(occlusion, None, labels, xlabel="Mean squared prediction shift (normalized target units)", title="Prediction shift under single-channel occlusion", path=self.output_dir / "plots" / "occlusion.png", percent=False)
            figures["agreement"] = plots.agreement_scatter(GradientAttribution.combined(importance, "share"), occlusion, labels, spearman, self.output_dir / "plots" / "agreement.png")

        return figures

    def _write_report(self, run, importance: dict, probes: dict, occlusion: np.ndarray | None, spearman: float | None, figures: dict[str, Path]) -> Path:
        kept = ", ".join(f"{name} {probes['kept'][name]['share']}" for name in run.target_channels)

        doc = MarkdownDoc(title=f"Input attribution: {run.backbone_name}")
        doc.paragraph(
            f"Gradient attribution on {probes['n_probes']} probe windows of the '{self.config.split}' split (share over {kept} probes; a probe with a zero gradient for a target carries no share): "
            "the share of |d output / d input| at the window centre falling on each input channel, per target channel, with the spread across probe windows. "
            "Gradient x input weights the same sensitivities by the input signal, so channels that are sensitive but quiet rank lower. "
            "Occlusion, when computed, is the mean squared shift of the normalized prediction when one input channel is set to zero in normalized units, which puts it at that channel's normalization location. "
            "Spearman rho measures how well the gradient ranking predicts the occlusion ranking."
        )

        header = ("Input channel", *[f"{name} %" for name in run.target_channels], "g x i %")
        if occlusion is not None:
            header += ("occlusion",)

        table = MarkdownTable(header)
        gxi   = GradientAttribution.combined(importance, "grad_x_input")
        for c, label in enumerate(run.input_channels):
            row = [f"`{label}`"]
            for name in run.target_channels:
                value = importance[name]["share"][c]
                row.append(f"{value * 100.0:.2f}" if np.isfinite(value) else "n/a")
            row.append(f"{gxi[c] * 100.0:.2f}" if np.isfinite(gxi[c]) else "n/a")
            if occlusion is not None:
                row.append(f"{occlusion[c]:.4g}")
            table.add_row(*row)
        doc.table(table)

        if spearman is not None:
            doc.paragraph(f"Gradient and occlusion rank agreement: Spearman rho = {spearman:.3f}.")

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def _analyse(self, run) -> AnalysisOutcome:
        importance, probes = self._gradient_importance(run)
        occlusion          = self._occlusion(run)
        combined           = GradientAttribution.combined(importance, "share")
        spearman           = RankCorrelation.spearman(combined, occlusion) if occlusion is not None else None
        figures            = self._render_figures(run.input_channels, importance, occlusion, spearman)

        payload = {
            "backbone"        : run.backbone_name,
            "split"           : self.config.split,
            "input_channels"  : run.input_channels,
            "target_channels" : run.target_channels,
            "gradient"        : {name: {key: values.tolist() for key, values in entry.items()} for name, entry in importance.items()},
            "gradient_probes" : probes,
            "occlusion"       : occlusion.tolist() if occlusion is not None else None,
            "spearman"        : spearman,
        }
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, importance, probes, occlusion, spearman, figures)
        headline    = f"strongest attribution on '{run.input_channels[int(np.nanargmax(combined))]}'" + (f", gradient-occlusion rho {spearman:.2f}" if spearman is not None else "")

        return AnalysisOutcome(payload, headline, report_path)


class InputAttributionBatch(RunBatch):

    SELECTOR_ACTION = "attribute"
    SECTION_TITLE   = "Input attribution"
    RUN_CLASS       = InputAttributionRun
