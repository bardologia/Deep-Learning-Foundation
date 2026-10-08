"""Attribution of a backbone's predictions to its input channels and tracks.

Combines a cheap gradient attribution on probe windows with a causal occlusion test
that zeroes single channels or whole tracks, compares the two rankings by Spearman
correlation and writes the figures and markdown report.
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisOutcome, AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import ModelDevice, PredictionCurves, ProbeWindows
from tools.data.io                                   import FileIO
from tools.metrics.scoring                           import RankCorrelation
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class ChannelLabeler:
    """Derives a human-readable label for every input channel of a run."""

    @staticmethod
    def build(run) -> list[str]:
        """Builds the per-channel labels in the order the input tensor stacks them.

        Args:
            run : Loaded run exposing the dataset input configuration, the secondary labels and the input channel count.

        Returns:
            One label per input channel, naming the primary, each secondary track, each
            derived channel and the aux field.

        Raises:
            ValueError : If the derived label count differs from the run's channel count.
        """
        input_config = run.dataset.input_config
        labels       = []

        if input_config.use_primary:
            per_source = input_config.primary_channels_per_source
            labels  += ["primary"] if per_source == 1 else [f"primary[{j}]" for j in range(per_source)]

        if input_config.use_secondaries:
            per_source = input_config.secondaries_channels_per_source
            for track in run.secondary_labels or [f"S{i}" for i in range(run.n_secondaries)]:
                labels += [f"sec {track}"] if per_source == 1 else [f"sec {track}[{j}]" for j in range(per_source)]

        if input_config.use_derived:
            per_source = input_config.derived_channels_per_source
            for track in run.secondary_labels or [f"S{i}" for i in range(run.n_secondaries)]:
                labels += [f"derived {track}"] if per_source == 1 else [f"derived {track}[{j}]" for j in range(per_source)]

        if input_config.use_aux:
            labels += ["aux"]

        if len(labels) != run.in_channels:
            raise ValueError(f"Derived {len(labels)} channel labels but the run has {run.in_channels} input channels; the input configuration does not match the label derivation")

        return labels


class ChannelGroups:
    """Maps the input channels onto the primary, secondaries, derived channels and aux field groups."""

    @staticmethod
    def build(run) -> dict[str, list[int]]:
        """Returns the channel indices of each enabled input group.

        Args:
            run : Loaded run exposing the dataset input configuration and the secondary count.

        Returns:
            Channel indices keyed by group name, in stacking order.

        Raises:
            ValueError : If the input configuration enables no group at all.
        """
        input_config = run.dataset.input_config
        offset       = 0
        groups       = {}

        if input_config.use_primary:
            per_pass          = input_config.primary_channels_per_source
            groups["primary"] = list(range(offset, offset + per_pass))
            offset           += per_pass

        if input_config.use_secondaries:
            count                 = run.n_secondaries * input_config.secondaries_channels_per_source
            groups["secondaries"] = list(range(offset, offset + count))
            offset               += count

        if input_config.use_derived:
            count                    = run.n_secondaries * input_config.derived_channels_per_source
            groups["derived"] = list(range(offset, offset + count))
            offset                  += count

        if input_config.use_aux:
            groups["aux"] = [offset]
            offset       += 1

        if not groups:
            raise ValueError("The input configuration enables no channel group; there is nothing to attribute")

        return groups

    @staticmethod
    def of_channels(groups: dict[str, list[int]], n_channels: int) -> list[str]:
        """Inverts the group map into a per-channel group name.

        Args:
            groups     : Channel indices keyed by group name.
            n_channels : Total number of input channels that must be covered.

        Returns:
            The group name of each channel, in channel order.

        Raises:
            ValueError : If any channel belongs to no group.
        """
        by_channel = {}
        for group, channels in groups.items():
            for channel in channels:
                by_channel[channel] = group

        missing = [c for c in range(n_channels) if c not in by_channel]
        if missing:
            raise ValueError(f"Channels {missing} belong to no input group; the group derivation does not match the channel count")

        return [by_channel[c] for c in range(n_channels)]


class TrackChannels:
    """Groups the input channels by the acquisition track they come from."""

    @staticmethod
    def build(run) -> list[list[int]]:
        """Returns the channel indices contributed by each secondary track.

        A track owns both its secondary channels and its derived channels, so
        occluding the returned indices removes that acquisition entirely.

        Args:
            run : Loaded run exposing the dataset input configuration and the secondary count.

        Returns:
            One list of channel indices per secondary track.

        Raises:
            ValueError : If neither secondaries nor derived channels are enabled, leaving no track-owned channel to drop.
        """
        input_config = run.dataset.input_config
        offset       = 0

        if input_config.use_primary:
            offset += input_config.primary_channels_per_source

        n_tracks  = run.n_secondaries
        per_track = [[] for _ in range(n_tracks)]

        if input_config.use_secondaries:
            per_source = input_config.secondaries_channels_per_source
            for track in range(n_tracks):
                per_track[track] += list(range(offset + track * per_source, offset + (track + 1) * per_source))
            offset += n_tracks * per_source

        if input_config.use_derived:
            per_source = input_config.derived_channels_per_source
            for track in range(n_tracks):
                per_track[track] += list(range(offset + track * per_source, offset + (track + 1) * per_source))
            offset += n_tracks * per_source

        if not any(per_track):
            raise ValueError("The input configuration holds neither secondaries nor derived; there is no track channel to drop")

        return per_track


class GradientAttribution:
    """Attributes the model output to its input channels through input gradients.

    Attributes:
        model  : Model to differentiate.
        window : Side of the probe windows, in pixels; the gradient targets the centre pixel.
    """

    FAMILIES = ("amp", "mu", "sigma")

    def __init__(self, model, window: int) -> None:
        """Stores the model and the probe-window size.

        Args:
            model  : Model whose input gradients are taken.
            window : Side of the square probe windows in pixels.
        """
        self.model  = model
        self.window = int(window)

    def channel_importance(self, windows: torch.Tensor) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, int]]]:
        """Measures the share of output sensitivity carried by each input channel.

        For every probe window and every output family, the absolute gradient of the
        summed centre-pixel prediction is accumulated per channel and normalized into a
        share, then averaged over the windows. A probe whose gradient mass is zero
        carries no share and is left out of that family's mean; the kept probes are
        counted.

        Args:
            windows : Probe windows of shape (probes, channels, window, window) in normalized input units.

        Returns:
            A pair of the attribution and the probe counts. Per output family ("amp",
            "mu", "sigma"), the attribution holds the mean gradient "share", its
            "share_std" across probes and the "grad_x_input" share, each an array of
            shape (channels,), and the counts hold how many probes entered the "share"
            and the "grad_x_input" mean.

        Raises:
            ValueError : If a gradient is non-finite, naming the probe and family, or if the gradient vanishes for every output family, meaning the output does not depend on the input.
        """
        half       = self.window // 2
        device     = ModelDevice.of(self.model)
        n_channels = windows.shape[1]

        grad_shares = {family: [] for family in self.FAMILIES}
        gxi_shares  = {family: [] for family in self.FAMILIES}

        for p in range(windows.shape[0]):
            x = windows[p:p + 1].clone().to(device).requires_grad_(True)
            y = self.model(x)

            for f, family in enumerate(self.FAMILIES):
                target = y[0, f::3, half, half].abs().sum()
                grad   = torch.autograd.grad(target, x, retain_graph=True)[0]

                grad_mass = grad.abs().sum(dim=(0, 2, 3)).cpu().numpy().astype(np.float64)
                gxi_mass  = (grad * x).abs().sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64)

                if not (np.isfinite(grad_mass).all() and np.isfinite(gxi_mass).all()):
                    raise ValueError(f"Input gradient of the predicted {family} at probe window {p} is non-finite; the checkpoint or the window is degenerate")

                if grad_mass.sum() > 0.0:
                    grad_shares[family].append(grad_mass / grad_mass.sum())
                if gxi_mass.sum() > 0.0:
                    gxi_shares[family].append(gxi_mass / gxi_mass.sum())

        if not any(grad_shares[family] for family in self.FAMILIES):
            raise ValueError("Gradient attribution is all zero for every output family; the model output does not depend on the input")

        importance = {}
        for family in self.FAMILIES:
            if grad_shares[family]:
                stacked = np.stack(grad_shares[family])
                share   = stacked.mean(axis=0)
                spread  = stacked.std(axis=0)
            else:
                share  = np.full(n_channels, np.nan)
                spread = np.full(n_channels, np.nan)

            grad_x_input = np.stack(gxi_shares[family]).mean(axis=0) if gxi_shares[family] else np.full(n_channels, np.nan)

            importance[family] = {"share": share, "share_std": spread, "grad_x_input": grad_x_input}

        kept = {family: {"share": len(grad_shares[family]), "grad_x_input": len(gxi_shares[family])} for family in self.FAMILIES}

        return importance, kept

    @classmethod
    def combined(cls, importance: dict[str, dict[str, np.ndarray]], key: str) -> np.ndarray:
        """Averages one attribution quantity across the output families.

        Args:
            importance : Per-family attribution as returned by channel_importance.
            key        : Quantity to combine, "share" or "grad_x_input".

        Returns:
            Array of shape (channels,) averaged over the families that carry finite values.

        Raises:
            ValueError : If no family carries a finite value for that quantity.
        """
        stacked = np.stack([importance[family][key] for family in cls.FAMILIES])
        live    = ~np.all(np.isnan(stacked), axis=1)

        if not live.any():
            raise ValueError(f"No output family carries a finite '{key}' attribution to combine")

        return np.nanmean(stacked[live], axis=0)


class ChannelOcclusion:
    """Measures how much the predicted profile shifts when input channels are zeroed.

    Attributes:
        model    : Model being evaluated.
        renderer : Renderer turning predicted Gaussian parameters into height profiles.
        on_pass  : Zero-argument callable invoked after every occluded forward pass, driving the progress bar.
    """

    def __init__(self, model, renderer: PredictionCurves, on_pass) -> None:
        """Stores the model, the curve renderer and the progress callback.

        Args:
            model    : Model to evaluate.
            renderer : Renderer producing the profiles the shift is measured on.
            on_pass  : Zero-argument callable invoked after every occluded forward pass.
        """
        self.model    = model
        self.renderer = renderer
        self.on_pass  = on_pass

    def deltas(self, loader, max_batches: int, channel_sets: list[list[int]]) -> np.ndarray:
        """Returns the mean curve MSE increase caused by zeroing each channel set.

        Args:
            loader       : Dataloader supplying the evaluation batches.
            max_batches  : Number of batches to average over.
            channel_sets : Channel index sets to zero, one occlusion per set.

        Returns:
            Array of shape (len(channel_sets),) holding the mean squared profile
            difference against the unoccluded prediction.

        Raises:
            ValueError : If the loader yields no batch within the budget.
        """
        totals = np.zeros(len(channel_sets), dtype=np.float64)
        count  = 0

        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break

                images      = batch[0]
                base_curves = self.renderer.render(self.model(images))

                for s, channels in enumerate(channel_sets):
                    occluded              = images.clone()
                    occluded[:, channels] = 0.0
                    curves                = self.renderer.render(self.model(occluded))
                    totals[s]            += float(((curves - base_curves) ** 2).mean())
                    self.on_pass()

                count += 1

        if count == 0:
            raise ValueError("Channel occlusion saw no batches; the loader is empty or max_batches is zero")

        return totals / count


class InputAttributionPlots(PlotBase):
    """Figures of the channel and track attribution, coloured by input group."""

    GROUP_COLORS = {
        "primary"     : PlotBase.OKABE_ITO[0],
        "secondaries" : PlotBase.OKABE_ITO[2],
        "derived"     : PlotBase.OKABE_ITO[1],
        "aux"         : PlotBase.OKABE_ITO[4],
    }

    def channel_bars(self, values: np.ndarray, stds: np.ndarray | None, labels: list[str], channel_groups: list[str], xlabel: str, title: str, path: Path, percent: bool = True) -> Path:
        """Draws a sorted horizontal bar chart of a per-channel attribution quantity.

        Args:
            values         : Attribution value of each channel, shape (channels,); non-finite entries are drawn as zero and annotated "n/a".
            stds           : Matching spread of shape (channels,) drawn as error bars, or None.
            labels         : Channel labels, in channel order.
            channel_groups : Group name of each channel, driving the bar colour.
            xlabel         : Axis label naming the quantity and its unit.
            title          : Figure title.
            path           : Output path of the figure.
            percent        : Whether the values are shares to be shown as percentages.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        finite = np.where(np.isfinite(values), values, 0.0)
        order  = np.argsort(finite)
        scale  = 100.0 if percent else 1.0

        fig, ax = plt.subplots(figsize=(self.FULL_WIDTH * 1.15, max(2.4, 0.30 * len(labels) + 1.2)))
        colors  = [self.GROUP_COLORS[channel_groups[i]] for i in order]
        errors  = np.where(np.isfinite(stds[order]), stds[order], 0.0) * scale if stds is not None else None

        ax.barh(range(len(order)), finite[order] * scale, xerr=errors, color=colors, error_kw={"elinewidth": 0.9, "ecolor": "0.3"})
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=8)

        span = float(finite.max() * scale) if finite.max() > 0 else 1.0
        for row, i in enumerate(order):
            text = "n/a" if not np.isfinite(values[i]) else (f"{values[i] * scale:.1f}%" if percent else f"{values[i]:.3g}")
            ax.text(finite[i] * scale + 0.01 * span, row, text, va="center", fontsize=7, color="0.25")

        present = [group for group in self.GROUP_COLORS if group in channel_groups]
        ax.legend(handles=[mpatches.Patch(color=self.GROUP_COLORS[g], label=g) for g in present], loc="lower right", fontsize=8, framealpha=0.9)

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(0.0, span * 1.14)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)

    def track_bars(self, values: np.ndarray, track_labels: list[str], path: Path) -> Path:
        """Draws the curve MSE increase caused by occluding each whole track.

        Args:
            values       : Occlusion delta of each track, shape (tracks,).
            track_labels : Track labels, in track order.
            path         : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        order = np.argsort(values)

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.55))
        ax.barh(range(len(order)), values[order], color=self.OKABE_ITO[1])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([track_labels[i] for i in order], fontsize=8)

        span = float(values.max()) if values.max() > 0 else 1.0
        for row, i in enumerate(order):
            ax.text(values[i] + 0.01 * span, row, f"{values[i]:.3g}", va="center", fontsize=7, color="0.25")

        ax.set_xlabel("Curve MSE increase when the whole track is zeroed")
        ax.set_title("Track occlusion impact")
        ax.set_xlim(0.0, span * 1.14)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)

    def agreement_scatter(self, gradient: np.ndarray, occlusion: np.ndarray, labels: list[str], rho: float, path: Path) -> Path:
        """Scatters the gradient channel ranking against the occlusion ranking.

        Args:
            gradient  : Combined gradient attribution of each channel, shape (channels,).
            occlusion : Occlusion delta of each channel, shape (channels,).
            labels    : Channel labels, in channel order.
            rho       : Spearman correlation between the two rankings, shown in the title.
            path      : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
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
        ax.set_title(f"Gradient vs occlusion rank agreement (Spearman ρ = {rho:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)


class InputAttributionRun(AnalysisRun):
    """Runs the gradient and occlusion attribution on a single training run."""

    SUMMARY_FILENAME = "input_attribution.json"
    REPORT_FILENAME  = "input_attribution.md"

    def _gradient_importance(self, run) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
        """Assembles the probe windows of the split and runs the gradient attribution on them.

        Families that left probes out of their share for a zero gradient are warned
        about once each.

        Args:
            run : Loaded run bundling the model and the dataset.

        Returns:
            A pair of the per-family attribution, as returned by
            GradientAttribution.channel_importance, and the probe record holding the
            total "n_probes" and the per-family "kept" counts.
        """
        probes  = ProbeWindows(run, self.config.window)
        centers = probes.centers(self.config.n_row_probes, self.config.n_col_probes)
        windows = probes.assemble(centers)

        with self.logger.timer(f"Input attribution: gradient pass ({len(centers)} probe windows)"):
            importance, kept = GradientAttribution(run.model.module, self.config.window).channel_importance(windows)

        self._warn_dropped_probes(kept, len(centers))

        return importance, {"n_probes": len(centers), "kept": kept}

    def _warn_dropped_probes(self, kept: dict[str, dict[str, int]], n_probes: int) -> None:
        """Warns once per output family whose share or gradient-times-input mean left probes out for a zero gradient.

        Args:
            kept     : Per family, the probes kept in the "share" and the "grad_x_input" mean.
            n_probes : Number of probe windows attributed.
        """
        for family, counts in kept.items():
            if counts["share"] < n_probes or counts["grad_x_input"] < n_probes:
                self.logger.warning(f"{self.run_dir.name}: gradient attribution of {family}: {n_probes - counts['share']}/{n_probes} probes had a zero gradient (share over {counts['share']}, gradient x input over {counts['grad_x_input']} probes)")

    def _occlusion(self, run, per_track: list[list[int]] | None) -> tuple[np.ndarray, np.ndarray | None] | tuple[None, None]:
        """Runs the single-channel and whole-track occlusion tests when enabled.

        Args:
            run       : Loaded run bundling the model and the dataloader.
            per_track : Channel indices of each track, or None when no track owns channels.

        Returns:
            A pair of the per-channel deltas of shape (channels,) and the per-track deltas
            of shape (tracks,), the latter None when no track set was given. Both are None
            when occlusion is disabled.
        """
        if not self.config.occlude:
            return None, None

        renderer     = PredictionCurves(run.n_gaussians, run.z_axis, self.config.render_amp_floor)
        channel_sets = [[c] for c in range(run.in_channels)]
        track_sets   = per_track if per_track is not None else []
        n_batches    = min(self.config.max_batches, len(run.loader))
        n_passes     = len(channel_sets + track_sets) * n_batches

        self.logger.info(f"{self.run_dir.name}: occluding {len(channel_sets)} channels and {len(track_sets)} secondaries over {n_batches} batches ({n_passes} forward passes)")

        with self.logger.timer(f"Input attribution: occlusion ({n_passes} passes)"), self.logger.track() as progress:
            task      = progress.add_task("[section]Channel occlusion[/section]", total=n_passes)
            occlusion = ChannelOcclusion(run.model, renderer, functools.partial(progress.advance, task))
            deltas    = occlusion.deltas(run.loader, self.config.max_batches, channel_sets + track_sets)

        channel_deltas = deltas[:run.in_channels]
        track_deltas   = deltas[run.in_channels:] if track_sets else None

        return channel_deltas, track_deltas

    def _render_figures(self, labels, groups_by_channel, importance, occlusion, track_deltas, track_labels, spearman) -> dict[str, Path]:
        """Renders the per-family, gradient-times-input, occlusion, agreement and track figures.

        Args:
            labels            : Channel labels, in channel order.
            groups_by_channel : Group name of each channel.
            importance        : Per-family gradient attribution.
            occlusion         : Per-channel occlusion deltas, or None when occlusion is disabled.
            track_deltas      : Per-track occlusion deltas, or None.
            track_labels      : Track labels, in track order.
            spearman          : Rank correlation between gradient and occlusion, or None.

        Returns:
            Figure paths keyed by figure name.
        """
        plots   = InputAttributionPlots()
        figures = {}

        for family in GradientAttribution.FAMILIES:
            entry = importance[family]
            figures[f"grad_{family}"] = plots.channel_bars(
                entry["share"], entry["share_std"], labels, groups_by_channel,
                xlabel = "Gradient attribution share [%] (± std across probes)",
                title  = f"Input-channel attribution of predicted {family}",
                path   = self.output_dir / "plots" / f"grad_{family}.png",
            )

        figures["grad_x_input"] = plots.channel_bars(
            GradientAttribution.combined(importance, "grad_x_input"), None, labels, groups_by_channel,
            xlabel = "Gradient × input attribution share [%] (mean over output families)",
            title  = "Input-channel attribution weighted by the input signal",
            path   = self.output_dir / "plots" / "grad_x_input.png",
        )

        if occlusion is not None:
            figures["occlusion"] = plots.channel_bars(
                occlusion, None, labels, groups_by_channel,
                xlabel  = "Curve MSE increase when the channel is zeroed",
                title   = "Prediction shift under single-channel occlusion",
                path    = self.output_dir / "plots" / "occlusion.png",
                percent = False,
            )
            figures["agreement"] = plots.agreement_scatter(GradientAttribution.combined(importance, "share"), occlusion, labels, spearman, self.output_dir / "plots" / "agreement.png")

        if track_deltas is not None:
            figures["occlusion_tracks"] = plots.track_bars(track_deltas, track_labels, self.output_dir / "plots" / "occlusion_tracks.png")

        return figures

    def _write_report(self, run, labels, groups_by_channel, importance, probes, occlusion, track_deltas, track_labels, spearman, figures) -> Path:
        """Writes the markdown report of the attribution.

        Args:
            run               : Loaded run, used for the backbone name in the title.
            labels            : Channel labels, in channel order.
            groups_by_channel : Group name of each channel.
            importance        : Per-family gradient attribution.
            probes            : Probe record with the total probe count and the probes each family kept.
            occlusion         : Per-channel occlusion deltas, or None.
            track_deltas      : Per-track occlusion deltas, or None.
            track_labels      : Track labels, in track order.
            spearman          : Rank correlation between gradient and occlusion, or None.
            figures           : Figure paths keyed by figure name.

        Returns:
            Path of the written report.
        """
        kept = ", ".join(f"{family} {probes['kept'][family]['share']}" for family in GradientAttribution.FAMILIES)

        doc = MarkdownDoc(title=f"Input attribution: {run.backbone_name}")
        doc.paragraph(
            f"Gradient attribution on {probes['n_probes']} probe windows of the '{self.config.split}' split "
            f"(share over {kept} probes; a probe with a zero gradient for a family carries no share): "
            "the share of |∂ output / ∂ input| falling on each input channel, per output family, with the spread across probe windows. "
            "Gradient × input weights the same sensitivities by the actual input signal, so channels that are sensitive but quiet rank lower. "
            "Occlusion, when computed, is the mean shift of the predicted profile when one channel (or one whole channel) is zeroed in normalized units, "
            "which puts it at that channel's normalization location: the median of the log1p values for the amplitude channels, zero phase for the "
            "derived phase channels, the mean only for z-scored channels. Spearman ρ measures how well the cheap gradient ranking predicts the "
            "causal occlusion ranking."
        )

        header = ("Channel", "Group", "amp %", "mu %", "sigma %", "g×i %")
        if occlusion is not None:
            header += ("occlusion MSE",)

        table = MarkdownTable(header)
        gxi   = GradientAttribution.combined(importance, "grad_x_input")
        for c, label in enumerate(labels):
            row = [f"`{label}`", groups_by_channel[c]]
            for family in GradientAttribution.FAMILIES:
                value = importance[family]["share"][c]
                row.append(f"{value * 100.0:.2f}" if np.isfinite(value) else "n/a")
            row.append(f"{gxi[c] * 100.0:.2f}" if np.isfinite(gxi[c]) else "n/a")
            if occlusion is not None:
                row.append(f"{occlusion[c]:.4g}")
            table.add_row(*row)
        doc.table(table)

        if spearman is not None:
            doc.paragraph(f"Gradient vs occlusion rank agreement: Spearman ρ = {spearman:.3f}.")

        if track_deltas is not None:
            doc.heading("Track occlusion", level=2)
            track_table = MarkdownTable(("Track", "Curve MSE increase"))
            for label, value in zip(track_labels, track_deltas):
                track_table.add_row(f"`{label}`", f"{value:.4g}")
            doc.table(track_table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def _analyse(self, run) -> AnalysisOutcome:
        """Attributes this run's predictions to its input channels and tracks.

        Args:
            run : Loaded run bundling the model, the dataloader and the dataset.

        Returns:
            The outcome whose payload holds the backbone name, the split, the channel
            labels and groups, the per-family gradient attribution, the occlusion deltas
            per channel and per secondary, and the gradient-occlusion Spearman correlation.
        """
        labels             = ChannelLabeler.build(run)
        groups             = ChannelGroups.build(run)
        groups_by_channel  = ChannelGroups.of_channels(groups, run.in_channels)
        importance, probes = self._gradient_importance(run)

        input_config = run.dataset.input_config
        has_tracks   = input_config.use_secondaries or input_config.use_derived
        per_track    = TrackChannels.build(run) if has_tracks else None
        track_labels = list(run.secondary_labels) if run.secondary_labels else [f"S{i}" for i in range(run.n_secondaries)]

        occlusion, track_deltas = self._occlusion(run, per_track)
        spearman                = RankCorrelation.spearman(GradientAttribution.combined(importance, "share"), occlusion) if occlusion is not None else None

        figures = self._render_figures(labels, groups_by_channel, importance, occlusion, track_deltas, track_labels, spearman)

        payload = {
            "backbone"        : run.backbone_name,
            "split"           : self.config.split,
            "channels"        : labels,
            "channel_groups"  : groups_by_channel,
            "gradient"        : {family: {key: values.tolist() for key, values in entry.items()} for family, entry in importance.items()},
            "gradient_probes" : probes,
            "occlusion"       : occlusion.tolist() if occlusion is not None else None,
            "occlusion_tracks": {label: float(value) for label, value in zip(track_labels, track_deltas)} if track_deltas is not None else None,
            "spearman"        : spearman,
        }
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, labels, groups_by_channel, importance, probes, occlusion, track_deltas, track_labels, spearman, figures)

        combined = GradientAttribution.combined(importance, "share")
        top      = labels[int(np.nanargmax(combined))]
        headline = f"strongest attribution on '{top}'" + (f", grad-occlusion ρ {spearman:.2f}" if spearman is not None else "")

        return AnalysisOutcome(payload, headline, report_path)


class InputAttributionBatch(RunBatch):
    """Runs the input attribution over every run matching the configured filter."""

    SELECTOR_ACTION = "attribute"
    SECTION_TITLE   = "Input attribution"
    RUN_CLASS       = InputAttributionRun
