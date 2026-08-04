from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.data.io                   import FileIO
from tools.metrics.gaussian_matching import GaussianMatcher
from tools.reporting.plotting        import PlotBase


class SeedDisagreementPlots(PlotBase):

    CMAP = "inferno"

    def map_figure(self, data: np.ndarray, title: str, colorbar_label: str, az_offset: int, rg_offset: int, path: Path) -> Path:
        H, W   = data.shape
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]

        return self._imshow_figure(
            np.asarray(data, dtype=np.float64),
            x_label        = "Range [px]",
            y_label        = "Azimuth [px]",
            title          = title,
            cmap           = self._cmap_with_bad(self.CMAP),
            extent         = extent,
            colorbar_label = colorbar_label,
            path           = path,
        )

    def risk_coverage_curve(self, rows: list[dict], full_risk: float, path: Path) -> Path:
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

    STD_QUANTILE = 0.5
    ERR_QUANTILE = 0.9
    MIN_PIXELS   = 4

    def __init__(self, disagreement_map: np.ndarray, risk_map: np.ndarray, label_r2_map: np.ndarray | None = None) -> None:
        self.disagreement = np.asarray(disagreement_map, dtype=np.float64)
        self.risk         = np.asarray(risk_map, dtype=np.float64)
        self.label_r2     = np.asarray(label_r2_map, dtype=np.float64) if label_r2_map is not None else None

        if self.disagreement.shape != self.risk.shape:
            raise ValueError(f"Disagreement map {self.disagreement.shape} and risk map {self.risk.shape} disagree in shape")

    def run(self) -> tuple[np.ndarray, dict]:
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

    COVERAGES  = tuple(np.round(np.linspace(0.05, 1.0, 20), 3))
    MIN_PIXELS = 4

    def __init__(self, disagreement_map: np.ndarray, risk_map: np.ndarray) -> None:
        confidence = np.asarray(disagreement_map, dtype=np.float64).reshape(-1)
        risk       = np.asarray(risk_map, dtype=np.float64).reshape(-1)

        if confidence.shape != risk.shape:
            raise ValueError(f"Disagreement map ({confidence.size}) and risk map ({risk.size}) disagree in size")

        both = np.isfinite(confidence) & np.isfinite(risk)
        if both.sum() < self.MIN_PIXELS:
            raise ValueError(f"Risk-coverage needs at least {self.MIN_PIXELS} pixels with finite disagreement and error")

        order = np.argsort(confidence[both], kind="stable")

        self.sorted_risk       = risk[both][order]
        self.sorted_confidence = confidence[both][order]

    def curve(self) -> list[dict]:
        n = self.sorted_risk.size

        rows = []
        for coverage in self.COVERAGES:
            keep = max(1, int(round(coverage * n)))
            rows.append({
                "coverage"  : float(coverage),
                "n"         : keep,
                "risk"      : float(self.sorted_risk[:keep].mean()),
                "threshold" : float(self.sorted_confidence[keep - 1]),
            })

        return rows

    def run(self) -> tuple[list[dict], dict]:
        rows      = self.curve()
        full_risk = rows[-1]["risk"]

        rank_conf = np.argsort(np.argsort(self.sorted_confidence)).astype(np.float64)
        rank_risk = np.argsort(np.argsort(self.sorted_risk)).astype(np.float64)
        spearman  = float(np.corrcoef(rank_conf, rank_risk)[0, 1])

        scalars = {
            "aurc"                  : float(np.mean([row["risk"] for row in rows])),
            "full_coverage_risk"    : full_risk,
            "risk_at_half_coverage" : next(row["risk"] for row in rows if row["coverage"] >= 0.5),
            "disagreement_error_spearman": spearman,
        }

        return rows, scalars


class SeedDisagreementMaps:

    PROFILE_CUBE = "pred_curves.npy"
    PARAMS_CUBE  = "params_pred.npy"
    GT_CUBE      = "params_gt.npy"
    RISK_CUBE    = "pixel_mse.npy"
    MIN_ACTIVE   = 2

    MAP_TITLES = {
        "seed_std_profile" : "Across-seed std of the reconstructed profile (mean over elevation)",
        "seed_std_amp"     : "Across-seed std of predicted amplitude (GT-matched slots)",
        "seed_std_mu"      : "Across-seed std of predicted mu (GT-matched slots, metres)",
        "seed_std_sigma"   : "Across-seed std of predicted sigma (GT-matched slots, metres)",
    }

    def __init__(self, group_dir: Path, run_dirs: list[Path], inference_dirs: list[Path], cubes_subdir: str, metrics_filename: str, output_dir: Path, logger) -> None:
        self.group_dir        = Path(group_dir)
        self.run_dirs         = [Path(d) for d in run_dirs]
        self.inference_dirs   = [Path(d) for d in inference_dirs]
        self.cube_dirs        = [d / cubes_subdir for d in self.inference_dirs]
        self.metrics_filename = metrics_filename
        self.output_dir       = Path(output_dir)
        self.figures_dir      = self.output_dir / "figures" / "seed_disagreement"
        self.logger           = logger

        self.seeds = [str(d.relative_to(self.group_dir)) for d in self.run_dirs]

    def _validate_cubes(self) -> None:
        for cube_dir in self.cube_dirs:
            for name in (self.PROFILE_CUBE, self.PARAMS_CUBE, self.GT_CUBE, self.RISK_CUBE):
                if not (cube_dir / name).is_file():
                    raise FileNotFoundError(f"{cube_dir / name} is missing; re-run inference with save_cubes enabled for every seed, or disable compute_disagreement")

    def _open_cubes(self, name: str) -> list[np.ndarray]:
        cubes  = [np.load(cube_dir / name, mmap_mode="r") for cube_dir in self.cube_dirs]
        shapes = {cube.shape for cube in cubes}
        if len(shapes) != 1:
            raise ValueError(f"Seed cubes {name} disagree on shape across {self.seeds}: {sorted(shapes)}; the seeds were inferred on different regions and cannot be compared pixelwise")

        return cubes

    def _offsets(self) -> tuple[int, int]:
        metrics = FileIO.load_json(self.inference_dirs[0] / self.metrics_filename)
        if "split_region" not in metrics:
            raise KeyError(f"{self.inference_dirs[0] / self.metrics_filename} has no split_region; re-run inference to regenerate it")

        az_start, _az_end, rg_start, _rg_end = (int(v) for v in metrics["split_region"])
        return az_start, rg_start

    def _profile_map(self) -> np.ndarray:
        curves             = self._open_cubes(self.PROFILE_CUBE)
        n_elev, n_az, n_rg = curves[0].shape

        acc = np.zeros((n_az, n_rg), dtype=np.float64)
        for e in range(n_elev):
            stack = np.stack([np.asarray(cube[e], dtype=np.float64) for cube in curves])
            acc  += stack.std(axis=0, ddof=1)

        return (acc / n_elev).astype(np.float32)

    def _masked_std(self, values: np.ndarray, active: np.ndarray) -> np.ndarray:
        n    = active.sum(axis=0)
        fill = np.where(active, values, 0.0)
        mean = np.divide(fill.sum(axis=0), n, out=np.zeros(n.shape, dtype=np.float64), where=n > 0)
        sq   = np.where(active, (values - mean) ** 2, 0.0)
        var  = np.divide(sq.sum(axis=0), n - 1, out=np.full(n.shape, np.nan), where=n >= self.MIN_ACTIVE)

        return np.sqrt(var)

    def _aligned_params(self) -> list[np.ndarray]:
        preds   = self._open_cubes(self.PARAMS_CUBE)
        gt      = np.asarray(np.load(self.cube_dirs[0] / self.GT_CUBE), dtype=np.float32)
        n_k     = preds[0].shape[0] // 3
        matcher = GaussianMatcher()

        return [matcher.aligned_prediction(np.asarray(pred, dtype=np.float32), gt, n_k) for pred in preds]

    def _param_maps(self) -> dict[str, np.ndarray]:
        aligned          = self._aligned_params()
        n_ch, n_az, n_rg = aligned[0].shape
        n_k              = n_ch // 3

        sums   = {field: np.zeros((n_az, n_rg), dtype=np.float64) for field in ("amp", "mu", "sigma")}
        counts = {field: np.zeros((n_az, n_rg), dtype=np.int32)   for field in ("amp", "mu", "sigma")}

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

    def _render_figures(self, maps: dict[str, np.ndarray]) -> dict[str, Path]:
        az_offset, rg_offset = self._offsets()
        plots                = SeedDisagreementPlots()

        figures = {}
        for name, data in maps.items():
            figures[name] = plots.map_figure(
                data,
                title          = self.MAP_TITLES[name],
                colorbar_label = "Across-seed std",
                az_offset      = az_offset,
                rg_offset      = rg_offset,
                path           = self.figures_dir / f"{name}.png",
            )

        return figures

    def _mean_risk_map(self) -> np.ndarray:
        risks = self._open_cubes(self.RISK_CUBE)
        return np.mean([np.asarray(risk, dtype=np.float64) for risk in risks], axis=0)

    def _risk_coverage(self, maps: dict[str, np.ndarray], summary: dict, figures: dict[str, Path]) -> list[dict]:
        rows, scalars = RiskCoverage(maps["seed_std_profile"], self._mean_risk_map()).run()

        for key, value in scalars.items():
            summary[f"risk_coverage_{key}"] = value

        figures["risk_coverage"] = SeedDisagreementPlots().risk_coverage_curve(rows, scalars["full_coverage_risk"], self.figures_dir / "risk_coverage.png")

        return rows

    def _label_r2_map(self) -> np.ndarray | None:
        path = self.cube_dirs[0] / "label_r2.npy"
        return np.asarray(np.load(path), dtype=np.float64) if path.is_file() else None

    def _flag_label_suspects(self, maps: dict[str, np.ndarray], summary: dict, figures: dict[str, Path]) -> None:
        mask, scalars = LabelSuspects(maps["seed_std_profile"], self._mean_risk_map(), self._label_r2_map()).run()

        summary.update(scalars)

        for cube_dir in self.cube_dirs:
            np.save(cube_dir / "label_suspect.npy", mask)

        az_offset, rg_offset    = self._offsets()
        figures["label_suspect"] = SeedDisagreementPlots().map_figure(
            mask.astype(np.float64),
            title          = "Label suspects: seeds agree with each other, not with the label",
            colorbar_label = "Suspect",
            az_offset      = az_offset,
            rg_offset      = rg_offset,
            path           = self.figures_dir / "label_suspect.png",
        )

    def run(self) -> dict:
        self._validate_cubes()
        FileIO.ensure_dirs(self.figures_dir)

        maps = {"seed_std_profile": self._profile_map()}
        maps.update(self._param_maps())

        written  = self._write_maps(maps)
        summary  = self._summarize(maps)
        figures  = self._render_figures(maps)
        coverage = self._risk_coverage(maps, summary, figures)

        self._flag_label_suspects(maps, summary, figures)

        self.logger.info(f"Seed-disagreement maps written into {len(self.cube_dirs)} seed cube dirs: {sorted(maps)}")

        return {"summary": summary, "figures": figures, "maps_written": written, "risk_coverage": coverage}
