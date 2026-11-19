
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

    TOLERANCE_FACTOR = 2.0

    def __init__(self, model, renderer: PredictionCurves, on_sweep, seed: int = 0) -> None:
        self.model    = model
        self.renderer = renderer
        self.on_sweep = on_sweep
        self.rng      = np.random.default_rng(seed)

    @staticmethod
    def noise_sweeps(sigmas: list[float], draws: int) -> int:
        return sum(1 if sigma == 0.0 else draws for sigma in {float(sigma) for sigma in sigmas})

    @staticmethod
    def gain_sweeps(factors: list[float]) -> int:
        return len(set(list(factors) + [1.0]))

    @staticmethod
    def drop_sweeps(n_secondaries: int, draws: int) -> int:
        return 1 + sum(min(draws, math.comb(n_secondaries, n_dropped)) for n_dropped in range(1, n_secondaries + 1))

    def _mse(self, batches: list[tuple[torch.Tensor, np.ndarray]], transform) -> float:
        total = 0.0
        for images, gt_curves in batches:
            perturbed = transform(images)
            with torch.no_grad():
                curves = self.renderer.render(self.model(perturbed))
            total += float(((curves - gt_curves) ** 2).mean())

        self.on_sweep()

        return total / len(batches)

    def noise_curve(self, batches: list, sigmas: list[float], channels: list[int] | None = None, draws: int = 1) -> list[dict]:
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
        rows = []
        for factor in sorted(set(list(factors) + [1.0])):
            def transform(images, factor=factor):
                return images * factor

            rows.append({"gain": float(factor), "mse": self._mse(batches, transform)})

        return rows

    def drop_curve(self, batches: list, per_track: list[list[int]], draws: int) -> list[dict]:
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

    def noise_curves(self, curves: dict[str, list[dict]], clean: float, tolerance: dict, path: Path) -> Path:
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

    SUMMARY_FILENAME = "robustness.json"
    REPORT_FILENAME  = "robustness.md"

    def _batches(self, run, renderer: PredictionCurves) -> list[tuple[torch.Tensor, np.ndarray]]:
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
        groups = ChannelGroups.build(run)
        scopes = {"all channels": None}

        amplitude = groups.get("primary", []) + groups.get("secondaries", [])
        if amplitude:
            scopes["amplitude channels"] = amplitude
        if "derived" in groups:
            scopes["derived"] = groups["derived"]

        return scopes

    def _write_report(self, run, results: dict, figures: dict[str, Path], n_batches: int) -> Path:
        clean     = results["noise"]["all channels"][0]["mse"]
        tolerance = results["noise_tolerance"]

        doc = MarkdownDoc(title=f"Robustness: {run.backbone_name}")
        doc.paragraph(
            f"Curve-MSE degradation on {n_batches} '{self.config.split}' batches under four controlled stresses. "
            f"Gaussian noise is added to the normalized inputs (averaged over {self.config.noise_draws} draws), either on all channels or restricted to one channel "
            "family, which separates amplitude fragility from derived-phase fragility. The gain sweep scales every input by a global factor, probing calibration "
            "error. Track dropout zeroes whole tracks (secondary and derived channels), averaged over up to "
            f"{self.config.draws_per_count} random track subsets per count, capped at the number of distinct subsets; counts with a single possible subset report no spread. "
            "Shift equivariance circularly shifts the input along row and compares the equally shifted prediction on the unwrapped rows: a dense per-pixel model "
            "should predict the same profile for the same pixel, so this error isolates positional dependence. "
            f"The noise tolerance is the interpolated sigma at which the loss doubles: σ {'≥' if tolerance['censored'] else '='} {tolerance['sigma']:.2f}."
        )

        extremes   = [results["gain"][0], results["gain"][-1]] if len(results["gain"]) > 1 else results["gain"]
        gain_label = "Gain " + " / ".join(f"{row['gain']:g}" for row in extremes) + " MSE"

        doc.kv_table({
            "Clean curve MSE"   : f"{clean:.4g}",
            "Noise tolerance σ" : f"{'≥ ' if tolerance['censored'] else ''}{tolerance['sigma']:.2f}",
            gain_label          : " / ".join(f"{row['mse']:.4g}" for row in extremes),
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

        payload = {"backbone": run.backbone_name, "split": self.config.split, "n_batches": len(batches), **results}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, results, figures, len(batches))

        suffix = "≥" if tolerance["censored"] else "="

        return AnalysisOutcome(payload, f"clean MSE {clean:.3g}, noise tolerance σ {suffix} {tolerance['sigma']:.2f}", report_path)


class RobustnessBatch(RunBatch):

    SELECTOR_ACTION = "stress"
    SECTION_TITLE   = "Robustness sweep"
    RUN_CLASS       = RobustnessRun
