"""Per-pixel across-seed disagreement maps and their uncertainty diagnostics.

Seed replicas inferred on the same region are compared pixelwise: the profile
and GT-matched Gaussian parameters give across-seed standard-deviation maps,
which are then used as a confidence signal for risk-coverage analysis and for
flagging label suspects, pixels where the seeds agree with each other but not
with the label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.data.io                   import FileIO
from tools.metrics.gaussian_matching import GaussianMatcher
from tools.metrics.scoring           import RankCorrelation
from tools.reporting.plotting        import PlotBase


class SeedDisagreementPlots(PlotBase):
    """Publication figures for disagreement maps and the risk-coverage curve."""

    CMAP = "inferno"

    def map_figure(self, data: np.ndarray, title: str, colorbar_label: str, row_offset: int, col_offset: int, path: Path) -> Path:
        """Saves a spatial map in absolute scene pixel coordinates.

        Args:
            data           : Map of shape (row, col).
            title          : Figure title.
            colorbar_label : Colorbar label.
            row_offset      : Row index of the map's first row in the full scene.
            col_offset      : Range index of the map's first column in the full scene.
            path           : Destination image path.

        Returns:
            The written figure path.
        """
        H, W   = data.shape
        extent = [col_offset, col_offset + W, row_offset + H, row_offset]

        return self._imshow_figure(
            np.asarray(data, dtype=np.float64),
            x_label        = "Col [px]",
            y_label        = "Row [px]",
            title          = title,
            cmap           = self._cmap_with_bad(self.CMAP),
            extent         = extent,
            colorbar_label = colorbar_label,
            path           = path,
        )

    def risk_coverage_curve(self, rows: list[dict], full_risk: float, path: Path) -> Path:
        """Saves the risk-coverage curve under seed-disagreement confidence.

        Args:
            rows      : Coverage rows holding "coverage" and "risk".
            full_risk : Mean pixel curve MSE at full coverage.
            path      : Destination image path.

        Returns:
            The written figure path.
        """
        self._apply_style()

        coverages = [row["coverage"] * 100.0 for row in rows]
        risks     = [row["risk"] for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(coverages, risks, marker="o", color="#0072B2", linewidth=1.4)
        ax.axhline(full_risk, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Coverage [% of pixels kept, most confident first]")
        ax.set_ylabel("Mean pixel curve MSE among kept pixels")
        ax.set_title("Risk-coverage under seed-disagreement confidence")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class LabelSuspects:
    """Flags pixels where the seeds agree with each other but not with the label.

    Attributes:
        disagreement : Across-seed disagreement map of shape (rows, cols).
        risk         : Per-pixel error map of the same shape.
        label_r2     : Label fit-quality map of the same shape, or None.
    """

    STD_QUANTILE = 0.5
    ERR_QUANTILE = 0.9
    MIN_PIXELS   = 4

    def __init__(self, disagreement_map: np.ndarray, risk_map: np.ndarray, label_r2_map: np.ndarray | None = None) -> None:
        """Binds the maps and checks that disagreement and risk share a shape.

        Args:
            disagreement_map : Across-seed std map of shape (rows, cols).
            risk_map         : Per-pixel curve MSE map of the same shape.
            label_r2_map     : Per-pixel label fit R2 of the same shape, or None.

        Raises:
            ValueError : If the disagreement and risk maps differ in shape.
        """
        self.disagreement = np.asarray(disagreement_map, dtype=np.float64)
        self.risk         = np.asarray(risk_map, dtype=np.float64)
        self.label_r2     = np.asarray(label_r2_map, dtype=np.float64) if label_r2_map is not None else None

        if self.disagreement.shape != self.risk.shape:
            raise ValueError(f"Disagreement map {self.disagreement.shape} and risk map {self.risk.shape} disagree in shape")

    def run(self) -> tuple[np.ndarray, dict]:
        """Flags low-disagreement, high-error pixels as label suspects.

        Returns:
            A tuple of the suspect mask of shape (rows, cols) as uint8 and
            the scalar summary holding the suspect count, fraction, both applied
            quantile thresholds and, when a label R2 map exists, its mean over
            suspects and overall.

        Raises:
            ValueError : If too few pixels have finite disagreement and error.
        """
        both = np.isfinite(self.disagreement) & np.isfinite(self.risk)
        if both.sum() < self.MIN_PIXELS:
            raise ValueError(f"Label-suspect detection needs at least {self.MIN_PIXELS} pixels with finite disagreement and error")

        std_thr = float(np.quantile(self.disagreement[both], self.STD_QUANTILE))
        err_thr = float(np.quantile(self.risk[both], self.ERR_QUANTILE))

        mask = both & (self.disagreement <= std_thr) & (self.risk > err_thr)

        scalars = {
            "label_suspect_n"       : float(mask.sum()),
            "label_suspect_frac"    : float(mask.mean()),
            "label_suspect_std_thr" : std_thr,
            "label_suspect_err_thr" : err_thr,
        }

        if self.label_r2 is not None:
            finite_r2 = np.isfinite(self.label_r2)

            if (mask & finite_r2).any():
                scalars["label_suspect_label_r2_mean"] = float(self.label_r2[mask & finite_r2].mean())
            if finite_r2.any():
                scalars["label_suspect_label_r2_overall"] = float(self.label_r2[finite_r2].mean())

        return mask.astype(np.uint8), scalars


class RiskCoverage:
    """Selective-prediction analysis using seed disagreement as the confidence score.

    Attributes:
        sorted_risk         : Per-pixel error of the finite pixels, most confident first.
        sorted_disagreement : The matching disagreement values in the same order.
    """

    COVERAGES  = tuple(np.round(np.linspace(0.05, 1.0, 20), 3))
    MIN_PIXELS = 4

    def __init__(self, disagreement_map: np.ndarray, risk_map: np.ndarray) -> None:
        """Flattens both maps and orders the finite pixels by ascending disagreement.

        Args:
            disagreement_map : Across-seed std map of shape (rows, cols).
            risk_map         : Per-pixel curve MSE map of the same shape.

        Raises:
            ValueError : If the maps differ in size or too few pixels are finite.
        """
        disagreement = np.asarray(disagreement_map, dtype=np.float64).reshape(-1)
        risk         = np.asarray(risk_map, dtype=np.float64).reshape(-1)

        if disagreement.shape != risk.shape:
            raise ValueError(f"Disagreement map ({disagreement.size}) and risk map ({risk.size}) disagree in size")

        both = np.isfinite(disagreement) & np.isfinite(risk)
        if both.sum() < self.MIN_PIXELS:
            raise ValueError(f"Risk-coverage needs at least {self.MIN_PIXELS} pixels with finite disagreement and error")

        order = np.argsort(disagreement[both], kind="stable")

        self.sorted_risk         = risk[both][order]
        self.sorted_disagreement = disagreement[both][order]

    def curve(self) -> list[dict]:
        """Builds the risk-coverage curve over the fixed coverage grid.

        Returns:
            One row per coverage level holding "coverage", the kept pixel count
            "n", the mean "risk" among kept pixels and the disagreement
            "threshold" at that coverage.
        """
        n = self.sorted_risk.size

        rows = []
        for coverage in self.COVERAGES:
            keep = max(1, int(round(coverage * n)))
            rows.append({
                "coverage"  : float(coverage),
                "n"         : keep,
                "risk"      : float(self.sorted_risk[:keep].mean()),
                "threshold" : float(self.sorted_disagreement[keep - 1]),
            })

        return rows

    def run(self) -> tuple[list[dict], dict]:
        """Builds the risk-coverage curve and its summary scalars.

        Returns:
            A tuple of the coverage rows and the scalars holding the area under
            the risk-coverage curve, the full-coverage risk, the risk at 50%
            coverage and the disagreement-error Spearman correlation.
        """
        rows      = self.curve()
        full_risk = rows[-1]["risk"]

        spearman = RankCorrelation.spearman(self.sorted_disagreement, self.sorted_risk)

        scalars = {
            "aurc"                  : float(np.mean([row["risk"] for row in rows])),
            "full_coverage_risk"    : full_risk,
            "risk_at_half_coverage" : next(row["risk"] for row in rows if row["coverage"] >= 0.5),
            "disagreement_error_spearman": spearman,
        }

        return rows, scalars


class SeedDisagreementMaps:
    """Builds, writes and summarises the across-seed disagreement maps of a group.

    Attributes:
        group_dir        : Directory holding the seed runs.
        run_dirs         : Seed run directories in seed order.
        inference_dirs   : Inference directory of each seed run.
        cube_dirs        : Cube directory inside each seed's inference directory.
        metrics_filename : Metrics filename carrying the split region.
        output_dir       : Group-level directory receiving the artifacts.
        active_amp_thr   : Physical amplitude above which a Gaussian slot counts as active in the GT-matched slot alignment, the threshold every seed's inference was scored with.
        figures_dir      : Subdirectory receiving the rendered maps.
        logger           : Logger for the written-artifact line.
        seeds            : Seed labels, relative to the group directory.
    """

    PROFILE_CUBE = "pred_curves.npy"
    PARAMS_CUBE  = "params_pred.npy"
    GT_CUBE      = "params_gt.npy"
    RISK_CUBE    = "pixel_mse.npy"
    MIN_ACTIVE   = 2

    MAP_TITLES = {
        "seed_std_profile" : "Across-seed std of the reconstructed profile (mean over z)",
        "seed_std_amp"     : "Across-seed std of predicted amplitude (GT-matched slots)",
        "seed_std_mu"      : "Across-seed std of predicted mu (GT-matched slots, z units)",
        "seed_std_sigma"   : "Across-seed std of predicted sigma (GT-matched slots, z units)",
    }

    def __init__(self, group_dir: Path, run_dirs: list[Path], inference_dirs: list[Path], cubes_subdir: str, metrics_filename: str, output_dir: Path, active_amp_thr: float, logger) -> None:
        """Binds the seed group and resolves its cube, output and figure directories.

        Args:
            group_dir        : Directory holding the seed runs.
            run_dirs         : Seed run directories.
            inference_dirs   : Inference directory of each seed run.
            cubes_subdir     : Cube subdirectory name inside an inference directory.
            metrics_filename : Metrics filename carrying the split region.
            output_dir       : Group-level directory receiving the artifacts.
            active_amp_thr   : Activity threshold every seed's inference was scored with, applied to the GT-matched slot alignment.
            logger           : Logger for the written-artifact line.
        """
        self.group_dir        = Path(group_dir)
        self.run_dirs         = [Path(d) for d in run_dirs]
        self.inference_dirs   = [Path(d) for d in inference_dirs]
        self.cube_dirs        = [d / cubes_subdir for d in self.inference_dirs]
        self.metrics_filename = metrics_filename
        self.output_dir       = Path(output_dir)
        self.active_amp_thr   = float(active_amp_thr)
        self.figures_dir      = self.output_dir / "figures" / "seed_disagreement"
        self.logger           = logger

        self.seeds = [str(d.relative_to(self.group_dir)) for d in self.run_dirs]

    def _validate_cubes(self) -> None:
        """Checks that every seed saved the profile, parameter, GT and risk cubes.

        Raises:
            FileNotFoundError : If any required cube is missing for any seed.
        """
        for cube_dir in self.cube_dirs:
            for name in (self.PROFILE_CUBE, self.PARAMS_CUBE, self.GT_CUBE, self.RISK_CUBE):
                if not (cube_dir / name).is_file():
                    raise FileNotFoundError(f"{cube_dir / name} is missing; re-run inference with save_cubes enabled for every seed, or disable compute_disagreement")

    def _open_cubes(self, name: str) -> list[np.ndarray]:
        """Memory-maps one cube per seed and checks they share a shape.

        Args:
            name : Cube filename inside each seed's cube directory.

        Returns:
            The memory-mapped cubes, in seed order.

        Raises:
            ValueError : If the seeds' cubes disagree on shape.
        """
        cubes  = [np.load(cube_dir / name, mmap_mode="r") for cube_dir in self.cube_dirs]
        shapes = {cube.shape for cube in cubes}
        if len(shapes) != 1:
            raise ValueError(f"Seed cubes {name} disagree on shape across {self.seeds}: {sorted(shapes)}; the seeds were inferred on different regions and cannot be compared pixelwise")

        return cubes

    def _offsets(self) -> tuple[int, int]:
        """Returns the (rows, cols) scene offsets of the split region every seed shares.

        Returns:
            The row and column start indices of the inferred region.

        Raises:
            KeyError   : If a seed's metrics file carries no split region.
            ValueError : If the seeds' inferences cover different regions, whose pixels cannot be compared.
        """
        regions = {}
        for seed, inference_dir in zip(self.seeds, self.inference_dirs):
            metrics = FileIO.load_json(inference_dir / self.metrics_filename)
            if "split_region" not in metrics:
                raise KeyError(f"{inference_dir / self.metrics_filename} has no split_region; re-run inference for seed {seed} to regenerate it")

            regions[seed] = tuple(int(v) for v in metrics["split_region"])

        if len(set(regions.values())) != 1:
            raise ValueError(f"The seeds were inferred on different regions {regions}; their disagreement maps would compare unrelated pixels, so re-infer them on one region")

        row_start, row_end, col_start, col_end = next(iter(regions.values()))
        self.logger.info(f"Split region shared by all {len(self.seeds)} seeds: rows {row_start}..{row_end}, cols {col_start}..{col_end}")

        return row_start, col_start

    def _profile_map(self) -> np.ndarray:
        """Builds the across-seed std of the reconstructed profile, averaged over z.

        Returns:
            A float32 map of shape (rows, cols).
        """
        curves             = self._open_cubes(self.PROFILE_CUBE)
        n_z, n_rows, n_cols = curves[0].shape

        acc = np.zeros((n_rows, n_cols), dtype=np.float64)
        for e in range(n_z):
            stack = np.stack([np.asarray(cube[e], dtype=np.float64) for cube in curves])
            acc  += stack.std(axis=0, ddof=1)

        return (acc / n_z).astype(np.float32)

    def _masked_std(self, values: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Returns the across-seed sample std over the active seeds only.

        Args:
            values : Per-seed values of shape (seeds, rows, cols).
            active : Boolean mask of the same shape marking contributing seeds.

        Returns:
            A map of shape (rows, cols), NaN where fewer than the minimum
            number of seeds are active.
        """
        n    = active.sum(axis=0)
        fill = np.where(active, values, 0.0)
        mean = np.divide(fill.sum(axis=0), n, out=np.zeros(n.shape, dtype=np.float64), where=n > 0)
        sq   = np.where(active, (values - mean) ** 2, 0.0)
        var  = np.divide(sq.sum(axis=0), n - 1, out=np.full(n.shape, np.nan), where=n >= self.MIN_ACTIVE)

        return np.sqrt(var)

    def _aligned_params(self) -> list[np.ndarray]:
        """Matches each seed's predicted slots to the shared ground-truth slots.

        Returns:
            One array per seed of shape (3 * n_gaussians, rows, cols), with
            slots reordered so the same index means the same physical component.
        """
        preds   = self._open_cubes(self.PARAMS_CUBE)
        gt      = np.asarray(np.load(self.cube_dirs[0] / self.GT_CUBE), dtype=np.float32)
        n_k     = preds[0].shape[0] // 3
        matcher = GaussianMatcher(amp_threshold=self.active_amp_thr)

        return [matcher.aligned_prediction(np.asarray(pred, dtype=np.float32), gt, n_k) for pred in preds]

    def _param_maps(self) -> dict[str, np.ndarray]:
        """Builds the across-seed std maps of amplitude, mu and sigma over matched slots.

        Returns:
            Mapping from map name to a float32 map of shape (rows, cols),
            each the mean over contributing slots of the per-slot across-seed
            std; mu and sigma are in metres.
        """
        aligned          = self._aligned_params()
        n_ch, n_rows, n_cols = aligned[0].shape
        n_k              = n_ch // 3

        sums   = {field: np.zeros((n_rows, n_cols), dtype=np.float64) for field in ("amp", "mu", "sigma")}
        counts = {field: np.zeros((n_rows, n_cols), dtype=np.int32)   for field in ("amp", "mu", "sigma")}

        for k in range(n_k):
            stacks = {
                "amp"   : np.stack([np.asarray(seed[3 * k],     dtype=np.float64) for seed in aligned]),
                "mu"    : np.stack([np.asarray(seed[3 * k + 1], dtype=np.float64) for seed in aligned]),
                "sigma" : np.stack([np.asarray(seed[3 * k + 2], dtype=np.float64) for seed in aligned]),
            }
            present = np.isfinite(stacks["amp"])

            for field, values in stacks.items():
                std = self._masked_std(np.nan_to_num(values), present & np.isfinite(values))
                ok  = np.isfinite(std)

                sums[field][ok]  += std[ok]
                counts[field]    += ok

        maps = {}
        for field in ("amp", "mu", "sigma"):
            with np.errstate(invalid="ignore"):
                slot_mean = np.divide(sums[field], counts[field], out=np.full(sums[field].shape, np.nan), where=counts[field] > 0)
            maps[f"seed_std_{field}"] = slot_mean.astype(np.float32)

        return maps

    def _write_maps(self, maps: dict[str, np.ndarray]) -> dict[str, list[str]]:
        """Writes every disagreement map into every seed's cube directory.

        Args:
            maps : Mapping from map name to a map of shape (rows, cols).

        Returns:
            Mapping from map name to the written cube paths, one per seed.
        """
        written = {}
        for name, data in maps.items():
            paths = []
            for cube_dir in self.cube_dirs:
                path = cube_dir / f"{name}.npy"
                np.save(path, data)
                paths.append(str(path))
            written[name] = paths

        return written

    def _summarize(self, maps: dict[str, np.ndarray]) -> dict:
        """Reduces each map to mean, median and 95th percentile over its finite pixels.

        Args:
            maps : Mapping from map name to a map of shape (rows, cols).

        Returns:
            The seed count plus per-map statistics, None where no pixel is finite.
        """
        summary = {"n_seeds": len(self.seeds)}
        for name, data in maps.items():
            finite = data[np.isfinite(data)]
            if finite.size:
                summary[f"{name}_mean"]   = float(finite.mean())
                summary[f"{name}_median"] = float(np.median(finite))
                summary[f"{name}_p95"]    = float(np.percentile(finite, 95.0))
            else:
                summary[f"{name}_mean"]   = None
                summary[f"{name}_median"] = None
                summary[f"{name}_p95"]    = None

        return summary

    def _render_figures(self, maps: dict[str, np.ndarray], offsets: tuple[int, int]) -> dict[str, Path]:
        """Renders one figure per disagreement map.

        Args:
            maps    : Mapping from map name to a map of shape (row, col).
            offsets : Row and col scene offsets of the split region.

        Returns:
            Mapping from map name to the written figure path.
        """
        row_offset, col_offset = offsets
        plots                = SeedDisagreementPlots()

        figures = {}
        for name, data in maps.items():
            figures[name] = plots.map_figure(
                data,
                title          = self.MAP_TITLES[name],
                colorbar_label = "Across-seed std",
                row_offset     = row_offset,
                col_offset     = col_offset,
                path           = self.figures_dir / f"{name}.png",
            )

        return figures

    def _mean_risk_map(self) -> np.ndarray:
        """Returns the seed-averaged per-pixel curve MSE map of shape (rows, cols)."""
        risks = self._open_cubes(self.RISK_CUBE)
        return np.mean([np.asarray(risk, dtype=np.float64) for risk in risks], axis=0)

    def _risk_coverage(self, maps: dict[str, np.ndarray], risk_map: np.ndarray, summary: dict, figures: dict[str, Path]) -> list[dict]:
        """Runs the risk-coverage analysis, extending the summary and figures in place.

        Args:
            maps     : Disagreement maps; the profile map is used as the confidence score.
            risk_map : Seed-averaged per-pixel error map of shape (rows, cols).
            summary  : Scalar summary extended with the risk-coverage scalars.
            figures  : Figure mapping extended with the risk-coverage curve.

        Returns:
            The risk-coverage rows.
        """
        rows, scalars = RiskCoverage(maps["seed_std_profile"], risk_map).run()

        for key, value in scalars.items():
            summary[f"risk_coverage_{key}"] = value

        figures["risk_coverage"] = SeedDisagreementPlots().risk_coverage_curve(rows, scalars["full_coverage_risk"], self.figures_dir / "risk_coverage.png")

        return rows

    def _label_r2_map(self) -> np.ndarray | None:
        """Returns the label fit R2 map of shape (rows, cols), or None when absent."""
        path = self.cube_dirs[0] / "label_r2.npy"
        return np.asarray(np.load(path), dtype=np.float64) if path.is_file() else None

    def _flag_label_suspects(self, maps: dict[str, np.ndarray], risk_map: np.ndarray, offsets: tuple[int, int], summary: dict, figures: dict[str, Path]) -> None:
        """Flags label suspects, writes the mask into every seed cube dir and renders it.

        Args:
            maps     : Disagreement maps; the profile map is used as the agreement signal.
            risk_map : Seed-averaged per-pixel error map of shape (rows, cols).
            offsets  : Row and range scene offsets of the split region.
            summary  : Scalar summary extended with the label-suspect scalars.
            figures  : Figure mapping extended with the suspect map.
        """
        mask, scalars = LabelSuspects(maps["seed_std_profile"], risk_map, self._label_r2_map()).run()

        summary.update(scalars)

        for cube_dir in self.cube_dirs:
            np.save(cube_dir / "label_suspect.npy", mask)

        row_offset, col_offset     = offsets
        figures["label_suspect"] = SeedDisagreementPlots().map_figure(
            mask.astype(np.float64),
            title          = "Label suspects: seeds agree with each other, not with the label",
            colorbar_label = "Suspect",
            row_offset     = row_offset,
            col_offset     = col_offset,
            path           = self.figures_dir / "label_suspect.png",
        )

    def run(self) -> dict:
        """Builds every disagreement map and its diagnostics for the seed group.

        Returns:
            A payload holding the scalar "summary", the rendered "figures", the
            "maps_written" cube paths per map and the "risk_coverage" rows.
        """
        self._validate_cubes()
        FileIO.ensure_dirs(self.figures_dir)

        maps = {"seed_std_profile": self._profile_map()}
        maps.update(self._param_maps())

        offsets  = self._offsets()
        risk_map = self._mean_risk_map()

        written  = self._write_maps(maps)
        summary  = self._summarize(maps)
        figures  = self._render_figures(maps, offsets)
        coverage = self._risk_coverage(maps, risk_map, summary, figures)

        self._flag_label_suspects(maps, risk_map, offsets, summary, figures)

        self.logger.info(f"Seed-disagreement maps written into {len(self.cube_dirs)} seed cube dirs: {sorted(maps)}")

        return {"summary": summary, "figures": figures, "maps_written": written, "risk_coverage": coverage}
