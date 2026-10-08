"""Robustness sweep of a trained backbone under controlled input degradations.

Curve-MSE degradation is measured under additive input noise (globally or per
channel family), global gain error, whole-track dropout, and circular row
shifts that probe positional dependence of the dense per-pixel prediction.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.input_attribution import ChannelGroups, TrackChannels
from pipelines.backbone.inference.analysis.run_batch         import AnalysisOutcome, AnalysisRun, RunBatch
from pipelines.backbone.inference.probes                     import PredictionCurves
from tools.data.io                                           import FileIO
from tools.reporting.markdown                                import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                                import PlotBase


class RobustnessCore:
    """Runs the degradation sweeps and reduces them to curve-MSE rows.

    Attributes:
        model    : Backbone model wrapper called on perturbed inputs.
        renderer : Renderer turning predicted Gaussian parameters into profiles.
        on_sweep : Zero-argument callable invoked after every sweep over the batches, driving the progress bar.
        rng      : Seeded generator for noise draws and track-dropout subsets.
    """

    TOLERANCE_FACTOR = 2.0

    def __init__(self, model, renderer: PredictionCurves, on_sweep, seed: int = 0) -> None:
        """Binds the model and renderer and seeds the perturbation generator.

        Args:
            model    : Backbone model wrapper mapping an input batch to parameters.
            renderer : Renderer producing z profiles from parameters.
            on_sweep : Zero-argument callable invoked after every sweep over the batches.
            seed     : Seed for the noise and dropout random generator.
        """
        self.model    = model
        self.renderer = renderer
        self.on_sweep = on_sweep
        self.rng      = np.random.default_rng(seed)

    @staticmethod
    def noise_sweeps(sigmas: list[float], draws: int) -> int:
        """Returns the number of sweeps one noise curve runs: one clean anchor plus every draw of every nonzero sigma."""
        return sum(1 if sigma == 0.0 else draws for sigma in {float(sigma) for sigma in sigmas})

    @staticmethod
    def gain_sweeps(factors: list[float]) -> int:
        """Returns the number of sweeps the gain curve runs, the unit factor included."""
        return len(set(list(factors) + [1.0]))

    @staticmethod
    def drop_sweeps(n_secondaries: int, draws: int) -> int:
        """Returns the number of sweeps the dropout curve runs: the intact stack plus the capped draws per dropped count."""
        return 1 + sum(min(draws, math.comb(n_secondaries, n_dropped)) for n_dropped in range(1, n_secondaries + 1))

    def _mse(self, batches: list[tuple[torch.Tensor, np.ndarray]], transform) -> float:
        """Returns the mean curve MSE against ground truth after applying a perturbation.

        Args:
            batches   : Pairs of an input batch of shape (batch, channels, height, width) and the ground-truth curves of shape (batch, n_z, height, width).
            transform : Callable mapping an input batch to its perturbed version.

        Returns:
            The curve MSE averaged over batches.
        """
        total = 0.0
        for images, gt_curves in batches:
            perturbed = transform(images)
            with torch.no_grad():
                curves = self.renderer.render(self.model(perturbed))
            total += float(((curves - gt_curves) ** 2).mean())

        self.on_sweep()

        return total / len(batches)

    def noise_curve(self, batches: list, sigmas: list[float], channels: list[int] | None = None, draws: int = 1) -> list[dict]:
        """Measures curve MSE under additive Gaussian noise on the normalized inputs.

        Args:
            batches  : Input/ground-truth batch pairs as taken by the MSE helper.
            sigmas   : Noise standard deviations in normalized input units; must hold the 0.0 clean baseline.
            channels : Input channel indices to perturb, or None for all channels.
            draws    : Number of independent noise draws averaged per sigma.

        Returns:
            One row per sigma holding "sigma", "mse" and "mse_std" (None for a
            single draw), sorted by increasing sigma.

        Raises:
            ValueError : If the clean 0.0 baseline is absent from the sigmas.
        """
        if 0.0 not in sigmas:
            raise ValueError(f"noise_sigmas {sigmas} hold no 0.0 entry; the clean baseline anchors every degradation measure")

        rows = []
        for sigma in sorted({float(sigma) for sigma in sigmas}):
            if sigma == 0.0:
                rows.append({"sigma": 0.0, "mse": self._mse(batches, lambda images: images), "mse_std": None})
                continue

            values = []
            for _draw in range(draws):
                def transform(images, sigma=sigma):
                    jittered = images.clone()
                    selected = list(range(images.shape[1])) if channels is None else channels
                    jittered[:, selected] += torch.from_numpy(self.rng.normal(0.0, sigma, size=(images.shape[0], len(selected), images.shape[2], images.shape[3])).astype(np.float32))
                    return jittered

                values.append(self._mse(batches, transform))

            rows.append({"sigma": float(sigma), "mse": float(np.mean(values)), "mse_std": float(np.std(values)) if draws > 1 else None})

        return rows

    def gain_curve(self, batches: list, factors: list[float]) -> list[dict]:
        """Measures curve MSE under a global multiplicative gain on the inputs.

        Args:
            batches : Input/ground-truth batch pairs as taken by the MSE helper.
            factors : Gain factors to sweep; the unit factor is always included.

        Returns:
            One row per factor holding "gain" and "mse", sorted by factor.
        """
        rows = []
        for factor in sorted(set(list(factors) + [1.0])):
            def transform(images, factor=factor):
                return images * factor

            rows.append({"gain": float(factor), "mse": self._mse(batches, transform)})

        return rows

    def drop_curve(self, batches: list, per_track: list[list[int]], draws: int) -> list[dict]:
        """Measures curve MSE as whole tracks are zeroed out of the input stack.

        Args:
            batches   : Input/ground-truth batch pairs as taken by the MSE helper.
            per_track : Input channel indices belonging to each track.
            draws     : Random track subsets to average per dropped count, capped at the number of distinct subsets.

        Returns:
            One row per dropped count holding "dropped", "mse" and "mse_std"
            (None when only one subset exists).
        """
        n_tracks = len(per_track)
        rows     = []

        for n_dropped in range(n_tracks + 1):
            if n_dropped == 0:
                rows.append({"dropped": 0, "mse": self._mse(batches, lambda images: images), "mse_std": None})
                continue

            n_draws = min(draws, math.comb(n_tracks, n_dropped))
            values  = []

            for _draw in range(n_draws):
                chosen   = self.rng.choice(n_tracks, size=n_dropped, replace=False)
                channels = [index for track in chosen for index in per_track[track]]

                def transform(images, channels=tuple(channels)):
                    occluded = images.clone()
                    occluded[:, list(channels)] = 0.0
                    return occluded

                values.append(self._mse(batches, transform))

            rows.append({"dropped": n_dropped, "mse": float(np.mean(values)), "mse_std": float(np.std(values)) if n_draws > 1 else None})

        return rows

    def shift_consistency(self, batches: list, shifts: list[int]) -> list[dict]:
        """Measures how far predictions deviate from exact row-shift equivariance.

        The input is circularly shifted along row and the equally shifted
        baseline prediction is compared on the rows unaffected by the wrap.

        Args:
            batches : Input/ground-truth batch pairs as taken by the MSE helper.
            shifts  : Row shifts in pixels, each strictly inside the patch height.

        Returns:
            One row per shift holding "shift" in pixels and "mse".

        Raises:
            ValueError : If any shift falls outside 1..height-1.
        """
        height = batches[0][0].shape[2]
        bad    = [s for s in shifts if int(s) <= 0 or int(s) >= height]
        if bad:
            raise ValueError(f"shift_pixels {bad} fall outside the valid range 1..{height - 1} for {height}px patches")

        rows = []
        for shift in shifts:
            s     = int(shift)
            total = 0.0

            for images, _gt_curves in batches:
                with torch.no_grad():
                    base    = self.renderer.render(self.model(images))
                    shifted = self.renderer.render(self.model(torch.roll(images, s, dims=2)))

                expected = np.roll(base, s, axis=-2)
                total   += float(((expected[..., s:, :] - shifted[..., s:, :]) ** 2).mean())

            rows.append({"shift": s, "mse": total / len(batches)})
            self.on_sweep()

        return rows

    @classmethod
    def tolerance_sigma(cls, noise_rows: list[dict]) -> dict:
        """Interpolates the noise sigma at which the curve MSE doubles.

        Args:
            noise_rows : Noise rows starting at the clean 0.0 baseline, sorted by sigma.

        Returns:
            A dict holding the tolerance "sigma" in normalized input units and
            "censored", true when the loss never doubled within the swept range.

        Raises:
            ValueError : If the rows are unsorted or do not start at sigma 0.0.
        """
        sigmas = [row["sigma"] for row in noise_rows]

        if sigmas[0] != 0.0 or sigmas != sorted(sigmas):
            raise ValueError(f"Noise rows {sigmas} must start at the 0.0 clean baseline and rise; the tolerance walk anchors on the first row and interpolates forward")

        clean     = noise_rows[0]["mse"]
        threshold = cls.TOLERANCE_FACTOR * max(clean, 1e-12)

        previous = noise_rows[0]
        for row in noise_rows[1:]:
            if row["mse"] > threshold:
                fraction = (threshold - previous["mse"]) / max(row["mse"] - previous["mse"], 1e-300)
                return {"sigma": float(previous["sigma"] + fraction * (row["sigma"] - previous["sigma"])), "censored": False}
            previous = row

        return {"sigma": float(noise_rows[-1]["sigma"]), "censored": True}


class RobustnessPlots(PlotBase):
    """Publication figures for the four robustness degradation sweeps."""

    def noise_curves(self, curves: dict[str, list[dict]], clean: float, tolerance: dict, path: Path) -> Path:
        """Saves the noise degradation curves for every channel scope.

        Args:
            curves    : Noise rows keyed by channel-scope label.
            clean     : Curve MSE on unperturbed inputs.
            tolerance : Tolerance dict holding "sigma" and "censored".
            path      : Destination image path.

        Returns:
            The written figure path.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.55))

        for index, (label, rows) in enumerate(curves.items()):
            xs   = [row["sigma"] for row in rows]
            ys   = [row["mse"] for row in rows]
            errs = [row["mse_std"] if row["mse_std"] is not None else 0.0 for row in rows]
            ax.errorbar(xs, ys, yerr=errs, marker="o", markersize=4, linewidth=1.3, capsize=2.0, color=self.OKABE_ITO[index % len(self.OKABE_ITO)], label=label)

        ax.axhline(clean, color="0.4", linestyle="--", linewidth=1.0, label="clean loss")
        ax.axhline(RobustnessCore.TOLERANCE_FACTOR * max(clean, 1e-12), color="0.4", linestyle=":", linewidth=1.0, label=f"{RobustnessCore.TOLERANCE_FACTOR:.0f}× clean")

        suffix = "≥" if tolerance["censored"] else "="
        ax.axvline(tolerance["sigma"], color=self.OKABE_ITO[3], linestyle=":", linewidth=1.1)
        ax.annotate(f"tolerance σ {suffix} {tolerance['sigma']:.2f}", xy=(tolerance["sigma"], 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=7, color="0.25", va="top")

        ax.set_xlabel("Added noise sigma [normalized input units]")
        ax.set_ylabel("Curve MSE")
        ax.set_yscale("log")
        ax.set_title("Degradation under input noise by channel scope")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)

    def gain_curve(self, rows: list[dict], clean: float, path: Path) -> Path:
        """Saves the curve MSE against global input gain on log-log axes.

        Args:
            rows  : Gain rows holding "gain" and "mse".
            clean : Curve MSE on unperturbed inputs.
            path  : Destination image path.

        Returns:
            The written figure path.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))

        ax.plot([row["gain"] for row in rows], [row["mse"] for row in rows], marker="o", color=self.OKABE_ITO[1], linewidth=1.4)
        ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1.0, label="no gain error")
        ax.axhline(clean, color="0.4", linestyle=":", linewidth=1.0, label="clean loss")

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Global input gain factor")
        ax.set_ylabel("Curve MSE")
        ax.set_yscale("log")
        ax.set_title("Degradation under calibration (gain) error")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)

    def drop_curve(self, rows: list[dict], path: Path) -> Path:
        """Saves the curve MSE against the number of dropped tracks, with subset spread.

        Args:
            rows : Dropout rows holding "dropped", "mse" and "mse_std".
            path : Destination image path.

        Returns:
            The written figure path.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))

        xs   = [row["dropped"] for row in rows]
        ys   = [row["mse"] for row in rows]
        errs = [row["mse_std"] if row["mse_std"] is not None else 0.0 for row in rows]

        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=1.4, capsize=2.5, color=self.OKABE_ITO[1])
        ax.axhline(ys[0], color="0.4", linestyle="--", linewidth=1.0, label="clean loss")

        ax.set_xticks(xs)
        ax.set_xlabel("Tracks dropped (secondary + derived channel zeroed)")
        ax.set_ylabel("Curve MSE")
        ax.set_yscale("log")
        ax.set_title("Degradation under track dropout (± spread across subsets)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)

    def shift_curve(self, rows: list[dict], clean: float, path: Path) -> Path:
        """Saves the shift-equivariance error against row shift.

        Args:
            rows  : Shift rows holding "shift" in pixels and "mse".
            clean : Curve MSE against ground truth on unperturbed inputs.
            path  : Destination image path.

        Returns:
            The written figure path.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))

        ax.plot([row["shift"] for row in rows], [row["mse"] for row in rows], marker="o", color=self.OKABE_ITO[2], linewidth=1.4, label="shift inconsistency")
        ax.axhline(clean, color="0.4", linestyle="--", linewidth=1.0, label="clean loss vs GT")

        ax.set_xticks([row["shift"] for row in rows])
        ax.set_xlabel("Row shift [px]")
        ax.set_ylabel("Curve MSE between shifted predictions")
        ax.set_yscale("log")
        ax.set_title("Shift-equivariance error")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)


class RobustnessRun(AnalysisRun):
    """Runs the four robustness sweeps on one trained run and reports them."""

    SUMMARY_FILENAME = "robustness.json"
    REPORT_FILENAME  = "robustness.md"

    def _batches(self, run, renderer: PredictionCurves) -> list[tuple[torch.Tensor, np.ndarray]]:
        """Materialises the evaluation batches with denormalized ground-truth curves.

        Args:
            run      : Loaded run providing the loader and normalizer.
            renderer : Renderer producing z profiles from parameters.

        Returns:
            Pairs of an input batch of shape (batch, channels, height, width) and
            ground-truth curves of shape (batch, n_z, height, width).

        Raises:
            ValueError : If the loader yielded no batches.
        """
        batches = []
        for index, batch in enumerate(run.loader):
            if index >= self.config.max_batches:
                break
            gt_phys = run.dataset.normalizer.denormalize_output(batch[1].float()).numpy()
            batches.append((batch[0], renderer.render(gt_phys)))

        if not batches:
            raise ValueError("The loader yielded no batches for the robustness sweep")

        return batches

    def _noise_scopes(self, run) -> dict[str, list[int] | None]:
        """Returns the channel scopes to sweep, mapping label to channel indices or None for all."""
        groups = ChannelGroups.build(run)
        scopes = {"all channels": None}

        amplitude = groups.get("primary", []) + groups.get("secondaries", [])
        if amplitude:
            scopes["amplitude channels"] = amplitude
        if "derived" in groups:
            scopes["derived"] = groups["derived"]

        return scopes

    def _write_report(self, run, results: dict, figures: dict[str, Path]) -> Path:
        """Writes the markdown robustness report with one table per stress.

        Args:
            run     : Loaded run providing the backbone name.
            results : Sweep results holding "noise", "noise_tolerance", "gain", "drop" (possibly None) and "shift".
            figures : Mapping from figure name to its written path.

        Returns:
            The written report path.
        """
        clean     = results["noise"]["all channels"][0]["mse"]
        tolerance = results["noise_tolerance"]

        doc = MarkdownDoc(title=f"Robustness: {run.backbone_name}")
        doc.paragraph(
            f"Curve-MSE degradation on {self.config.max_batches} '{self.config.split}' batches under four controlled stresses. "
            f"Gaussian noise is added to the normalized inputs (averaged over {self.config.noise_draws} draws), either on all channels or restricted to one channel "
            "family, which separates amplitude fragility from derived-phase fragility. The gain sweep scales every input by a global factor, probing calibration "
            "error. Track dropout zeroes whole tracks (secondary and derived channels), averaged over up to "
            f"{self.config.draws_per_count} random track subsets per count, capped at the number of distinct subsets; counts with a single possible subset report no spread. "
            "Shift equivariance circularly shifts the input along row and compares the equally shifted prediction on the unwrapped rows: a dense per-pixel model "
            "should predict the same profile for the same pixel, so this error isolates positional dependence. "
            f"The noise tolerance is the interpolated sigma at which the loss doubles: σ {'≥' if tolerance['censored'] else '='} {tolerance['sigma']:.2f}."
        )

        doc.kv_table({
            "Clean curve MSE"     : f"{clean:.4g}",
            "Noise tolerance σ"   : f"{'≥ ' if tolerance['censored'] else ''}{tolerance['sigma']:.2f}",
            "Gain 0.5 / 2.0 MSE"  : " / ".join(f"{row['mse']:.4g}" for row in results["gain"] if row["gain"] in (0.5, 2.0)),
        })

        doc.heading("Input noise", level=2)
        noise_table = MarkdownTable(("Noise sigma",) + tuple(results["noise"]) )
        for index, row in enumerate(results["noise"]["all channels"]):
            cells = [f"{row['sigma']:.2f}"]
            for scope in results["noise"]:
                entry = results["noise"][scope][index]
                cells.append(f"{entry['mse']:.4g}" + (f" ± {entry['mse_std']:.2g}" if entry["mse_std"] is not None else ""))
            noise_table.add_row(*cells)
        doc.table(noise_table)

        doc.heading("Gain error", level=2)
        gain_table = MarkdownTable(("Gain factor", "Curve MSE"))
        for row in results["gain"]:
            gain_table.add_row(f"{row['gain']:.2f}", f"{row['mse']:.4g}")
        doc.table(gain_table)

        if results["drop"] is not None:
            doc.heading("Track dropout", level=2)
            drop_table = MarkdownTable(("Tracks dropped", "Curve MSE", "± across draws"))
            for row in results["drop"]:
                spread = row.get("mse_std")
                drop_table.add_row(str(row["dropped"]), f"{row['mse']:.4g}", f"{spread:.2g}" if spread is not None else "n/a")
            doc.table(drop_table)

        doc.heading("Shift equivariance", level=2)
        shift_table = MarkdownTable(("Row shift [px]", "Consistency MSE"))
        for row in results["shift"]:
            shift_table.add_row(str(row["shift"]), f"{row['mse']:.4g}")
        doc.table(shift_table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, path.name)

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def _analyse(self, run) -> AnalysisOutcome:
        """Runs the noise, gain, dropout and shift sweeps and writes their artifacts.

        Args:
            run : Loaded run bundling the model, the dataloader and the normalizer.

        Returns:
            The outcome whose payload holds the backbone name, split and all sweep results.
        """
        renderer = PredictionCurves(run.n_gaussians, run.z_axis, self.config.render_amp_floor)
        batches  = self._batches(run, renderer)

        sigmas        = list(self.config.noise_sigmas)
        scopes        = self._noise_scopes(run)
        shifts        = [int(s) for s in self.config.shift_pixels]
        per_secondary = TrackChannels.build(run) if run.dataset.input_config.use_secondaries or run.dataset.input_config.use_derived else None

        planned = {
            "noise"   : len(scopes) * RobustnessCore.noise_sweeps(sigmas, self.config.noise_draws),
            "gain"    : RobustnessCore.gain_sweeps(list(self.config.gain_factors)),
            "dropout" : RobustnessCore.drop_sweeps(len(per_secondary), self.config.draws_per_count) if per_secondary is not None else 0,
            "shift"   : len(shifts),
        }
        self.logger.info(f"{self.run_dir.name}: {sum(planned.values())} sweeps over {len(batches)} batches planned (" + ", ".join(f"{stress} {count}" for stress, count in planned.items()) + ")")

        with self.logger.track() as progress:
            task = progress.add_task("[section]Robustness sweeps[/section]", total=sum(planned.values()))
            core = RobustnessCore(run.model, renderer, functools.partial(progress.advance, task), seed=self.config.seed)

            with self.logger.timer(f"Robustness: noise ({planned['noise']} sweeps)"):
                noise = {scope: core.noise_curve(batches, sigmas, channels=channels, draws=self.config.noise_draws) for scope, channels in scopes.items()}

            with self.logger.timer(f"Robustness: gain ({planned['gain']} sweeps)"):
                gain = core.gain_curve(batches, list(self.config.gain_factors))

            drop = None
            if per_secondary is not None:
                with self.logger.timer(f"Robustness: dropout ({planned['dropout']} sweeps)"):
                    drop = core.drop_curve(batches, per_secondary, self.config.draws_per_count)

            with self.logger.timer(f"Robustness: shift ({planned['shift']} sweeps)"):
                shift = core.shift_consistency(batches, shifts)

        clean     = noise["all channels"][0]["mse"]
        tolerance = RobustnessCore.tolerance_sigma(noise["all channels"])

        plots   = RobustnessPlots()
        figures = {
            "noise" : plots.noise_curves(noise, clean, tolerance, self.output_dir / "noise.png"),
            "gain"  : plots.gain_curve(gain, clean, self.output_dir / "gain.png"),
            "shift" : plots.shift_curve(shift, clean, self.output_dir / "shift.png"),
        }
        if drop is not None:
            figures["drop"] = plots.drop_curve(drop, self.output_dir / "drop.png")

        results = {"noise": noise, "noise_tolerance": tolerance, "gain": gain, "drop": drop, "shift": shift}

        payload = {"backbone": run.backbone_name, "split": self.config.split, **results}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, results, figures)

        suffix = "≥" if tolerance["censored"] else "="

        return AnalysisOutcome(payload, f"clean MSE {clean:.3g}, noise tolerance σ {suffix} {tolerance['sigma']:.2f}", report_path)


class RobustnessBatch(RunBatch):
    """Runs the robustness sweep on every selected run."""

    SELECTOR_ACTION = "stress"
    SECTION_TITLE   = "Robustness sweep"
    RUN_CLASS       = RobustnessRun
