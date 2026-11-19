
from __future__ import annotations

import numpy as np

from tools.data.gaussians  import GaussianActivity
from tools.metrics.scoring import RankCorrelation


class StratifiedErrors:

    N_BINS        = 12
    MIN_BIN_COUNT = 30

    def __init__(self, run, result) -> None:
        self.loaded         = run
        self.result         = result
        self.active_amp_thr = float(run.activity_threshold)

    def _gt_active_count(self) -> np.ndarray:
        params = np.asarray(self.result.params_gt, dtype=np.float64)
        n_k    = self.loaded.n_gaussians
        amps   = np.stack([params[3 * k] for k in range(n_k)])

        return GaussianActivity.is_active(amps, self.active_amp_thr).sum(axis=0).astype(np.float64)

    def covariates(self) -> dict[str, tuple[np.ndarray, bool]]:
        out = {}

        if self.result.params_gt is not None:
            out["gt_active_count"] = (self._gt_active_count(), True)

        if self.loaded.complex_inputs is not None:
            out["primary_amplitude"] = (np.abs(self.loaded.complex_inputs[0]).astype(np.float64), False)

        aux = self.loaded.dataset.aux
        if aux is not None:
            grad_row, grad_col = np.gradient(np.asarray(aux, dtype=np.float64))
            out["aux_gradient"]  = (np.hypot(grad_row, grad_col), False)

        if self.result.label_r2 is not None:
            out["label_r2"] = (np.asarray(self.result.label_r2, dtype=np.float64), False)

        if not out:
            raise ValueError("No covariate is available for error stratification; the run holds neither GT params, inputs, Aux nor a label-quality map")

        return out

    def _bin_rows(self, cov: np.ndarray, err: np.ndarray, discrete: bool) -> tuple[list[dict], list[dict], int]:
        flat_cov    = cov.reshape(-1)
        flat_err    = err.reshape(-1)
        both        = np.isfinite(flat_cov) & np.isfinite(flat_err)
        n_nonfinite = int((~both).sum())
        flat_cov    = flat_cov[both]
        flat_err    = flat_err[both]

        if not flat_cov.size:
            raise ValueError("Stratification received no finite (covariate, error) pairs")

        if discrete:
            levels  = np.unique(flat_cov)
            members = [(float(level), flat_err[flat_cov == level]) for level in levels]
        else:
            edges = np.unique(np.quantile(flat_cov, np.linspace(0.0, 1.0, self.N_BINS + 1)))
            if edges.size < 3:
                raise ValueError(f"Covariate is effectively constant ({edges.size - 1} distinct bins); stratification is meaningless")

            index   = np.clip(np.searchsorted(edges, flat_cov, side="right") - 1, 0, edges.size - 2)
            members = [((float(edges[b]) + float(edges[b + 1])) / 2.0, flat_err[index == b]) for b in range(edges.size - 1)]

        rows    = []
        dropped = []
        for center, values in members:
            if values.size < self.MIN_BIN_COUNT:
                dropped.append({"center": center, "n": int(values.size)})
                continue

            rows.append({
                "center" : center,
                "n"      : int(values.size),
                "median" : float(np.median(values)),
                "q25"    : float(np.percentile(values, 25.0)),
                "q75"    : float(np.percentile(values, 75.0)),
            })

        if not rows:
            raise ValueError(f"Every stratification bin holds fewer than {self.MIN_BIN_COUNT} pixels; the region is too small to stratify")

        return rows, dropped, n_nonfinite

    def run(self) -> tuple[dict[str, list[dict]], dict[str, float]]:
        err     = np.asarray(self.result.pixel_mse, dtype=np.float64)
        tables  = {}
        scalars = {}

        for name, (cov, discrete) in self.covariates().items():
            rows, dropped, n_nonfinite = self._bin_rows(cov, err, discrete)
            tables[name]               = {"rows": rows, "discrete": discrete, "dropped": dropped}

            scalars[f"strat_{name}_spearman"]         = RankCorrelation.spearman(cov, err)
            scalars[f"strat_{name}_n_bins"]           = float(len(rows))
            scalars[f"strat_{name}_n_dropped_bins"]   = float(len(dropped))
            scalars[f"strat_{name}_pixels_dropped"]   = float(sum(entry["n"] for entry in dropped))
            scalars[f"strat_{name}_pixels_nonfinite"] = float(n_nonfinite)

        return tables, scalars
