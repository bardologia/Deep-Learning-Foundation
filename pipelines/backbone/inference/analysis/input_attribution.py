from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import ModelDevice, PredictionCurves, ProbeWindows
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class ChannelLabeler:

    @staticmethod
    def build(run) -> list[str]:
        input_config = run.dataset.input_config
        labels       = []

        if input_config.use_primary:
            per_pass = input_config.primary_channels_per_pass
            labels  += ["primary"] if per_pass == 1 else [f"primary[{j}]" for j in range(per_pass)]

        if input_config.use_secondaries:
            per_pass = input_config.secondaries_channels_per_pass
            for track in run.secondary_labels or [f"S{i}" for i in range(run.n_secondaries)]:
                labels += [f"sec {track}"] if per_pass == 1 else [f"sec {track}[{j}]" for j in range(per_pass)]

        if input_config.use_interferograms:
            per_pass = input_config.interferograms_channels_per_pass
            for track in run.secondary_labels or [f"S{i}" for i in range(run.n_secondaries)]:
                labels += [f"ifg {track}"] if per_pass == 1 else [f"ifg {track}[{j}]" for j in range(per_pass)]

        if input_config.use_dem:
            labels += ["dem"]

        if len(labels) != run.in_channels:
            raise ValueError(f"Derived {len(labels)} channel labels but the run has {run.in_channels} input channels; the input configuration does not match the label derivation")

        return labels


class TrackChannels:

    @staticmethod
    def build(run) -> list[list[int]]:
        input_config = run.dataset.input_config
        offset       = 0

        if input_config.use_primary:
            offset += input_config.primary_channels_per_pass

        n_tracks  = run.n_secondaries
        per_track = [[] for _ in range(n_tracks)]

        if input_config.use_secondaries:
            per_pass = input_config.secondaries_channels_per_pass
            for track in range(n_tracks):
                per_track[track] += list(range(offset + track * per_pass, offset + (track + 1) * per_pass))
            offset += n_tracks * per_pass

        if input_config.use_interferograms:
            per_pass = input_config.interferograms_channels_per_pass
            for track in range(n_tracks):
                per_track[track] += list(range(offset + track * per_pass, offset + (track + 1) * per_pass))
            offset += n_tracks * per_pass

        if not any(per_track):
            raise ValueError("The input configuration holds neither secondaries nor interferograms; there is no track channel to drop")

        return per_track


class GradientAttribution:

    FAMILIES = ("amp", "mu", "sigma")

    def __init__(self, model, window: int) -> None:
        self.model  = model
        self.window = int(window)

    def channel_importance(self, windows: torch.Tensor) -> dict[str, np.ndarray]:
        half       = self.window // 2
        device     = ModelDevice.of(self.model)
        n_channels = windows.shape[1]
        totals     = {family: np.zeros(n_channels, dtype=np.float64) for family in self.FAMILIES}

        for p in range(windows.shape[0]):
            x = windows[p:p + 1].clone().to(device).requires_grad_(True)
            y = self.model(x)

            for f, family in enumerate(self.FAMILIES):
                target = y[0, f::3, half, half].abs().sum()
                grad   = torch.autograd.grad(target, x, retain_graph=True)[0]

                totals[family] += grad.abs().sum(dim=(0, 2, 3)).cpu().numpy()

        live = [family for family in self.FAMILIES if totals[family].sum() > 0.0]
        if not live:
            raise ValueError("Gradient attribution is all zero for every output family; the model output does not depend on the input")

        for family in self.FAMILIES:
            total = totals[family].sum()
            totals[family] = totals[family] / total if total > 0.0 else np.full_like(totals[family], np.nan)

        return totals


class ChannelOcclusion:

    def __init__(self, model, renderer: PredictionCurves) -> None:
        self.model    = model
        self.renderer = renderer

    def deltas(self, loader, max_batches: int, n_channels: int) -> np.ndarray:
        totals = np.zeros(n_channels, dtype=np.float64)
        count  = 0

        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break

                images      = batch[0]
                base_curves = self.renderer.render(self.model(images))

                for c in range(n_channels):
                    occluded       = images.clone()
                    occluded[:, c] = 0.0
                    curves         = self.renderer.render(self.model(occluded))
                    totals[c]     += float(((curves - base_curves) ** 2).mean())

                count += 1

        if count == 0:
            raise ValueError("Channel occlusion saw no batches; the loader is empty or max_batches is zero")

        return totals / count


class InputAttributionPlots(PlotBase):

    def channel_bars(self, values: np.ndarray, labels: list[str], ylabel: str, title: str, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.75))
        ax.bar(range(values.size), values, color="#0072B2")
        ax.set_xticks(range(values.size))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class InputAttributionRun(AnalysisRun):

    SUMMARY_FILENAME = "input_attribution.json"
    REPORT_FILENAME  = "input_attribution.md"

    def _gradient_importance(self, run) -> dict[str, np.ndarray]:
        probes  = ProbeWindows(run, self.config.window)
        centers = probes.centers(self.config.n_azimuth_probes, self.config.n_range_probes)
        windows = probes.assemble(centers)

        return GradientAttribution(run.model.module, self.config.window).channel_importance(windows)

    def _occlusion_deltas(self, run) -> np.ndarray | None:
        if not self.config.occlude:
            return None

        renderer = PredictionCurves(run.n_gaussians, run.x_axis, self.config.render_amp_floor)
        return ChannelOcclusion(run.model, renderer).deltas(run.loader, self.config.max_batches, run.in_channels)

    def _render_figures(self, labels: list[str], importance: dict[str, np.ndarray], occlusion: np.ndarray | None) -> dict[str, Path]:
        plots   = InputAttributionPlots()
        figures = {}

        for family, values in importance.items():
            figures[f"grad_{family}"] = plots.channel_bars(
                values, labels,
                ylabel = "Gradient attribution share",
                title  = f"Input-channel attribution of predicted {family}",
                path   = self.output_dir / "plots" / f"grad_{family}.png",
            )

        if occlusion is not None:
            figures["occlusion"] = plots.channel_bars(
                occlusion, labels,
                ylabel = "Curve MSE vs baseline prediction",
                title  = "Prediction shift when one input channel is zeroed",
                path   = self.output_dir / "plots" / "occlusion.png",
            )

        return figures

    def _write_report(self, run, labels: list[str], importance: dict[str, np.ndarray], occlusion: np.ndarray | None, figures: dict[str, Path]) -> Path:
        doc = MarkdownDoc(title=f"Input attribution: {run.backbone_name}")
        doc.paragraph(
            f"Gradient attribution on {self.config.n_azimuth_probes * self.config.n_range_probes} probe windows of the '{self.config.split}' split: "
            "the share of |∂ output / ∂ input| falling on each input channel, per output family. "
            "Occlusion, when computed, is the mean shift of the predicted profile when one channel is zeroed in normalized units, which puts it at that channel's normalization location: the median of the log1p values for the amplitude channels, zero phase for the interferogram phase channels, the mean only for z-scored channels."
        )

        header = ("Channel", "amp", "mu", "sigma") + (("occlusion MSE",) if occlusion is not None else ())
        table  = MarkdownTable(header)
        for c, label in enumerate(labels):
            row = [f"`{label}`"] + [f"{importance[family][c] * 100.0:.2f}%" for family in GradientAttribution.FAMILIES]
            if occlusion is not None:
                row.append(f"{occlusion[c]:.4g}")
            table.add_row(*row)
        doc.table(table)

        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run        = self._load_run()
        labels     = ChannelLabeler.build(run)
        importance = self._gradient_importance(run)
        occlusion  = self._occlusion_deltas(run)

        figures = self._render_figures(labels, importance, occlusion)

        payload = {
            "backbone"  : run.backbone_name,
            "split"     : self.config.split,
            "channels"  : labels,
            "gradient"  : {family: values.tolist() for family, values in importance.items()},
            "occlusion" : occlusion.tolist() if occlusion is not None else None,
        }
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, labels, importance, occlusion, figures)

        combined = np.nansum(np.stack([importance[family] for family in GradientAttribution.FAMILIES]), axis=0)
        top      = labels[int(np.argmax(combined))]
        self.logger.ok(f"{self.run_dir.name}: strongest attribution on '{top}' -> {report_path}")

        return payload


class InputAttributionBatch(RunBatch):

    SELECTOR_ACTION = "attribute"
    SECTION_TITLE   = "Input attribution"
    RUN_CLASS       = InputAttributionRun
