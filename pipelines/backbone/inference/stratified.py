from __future__ import annotations

import numpy as np

from tools.loss.param_loss import ParamMatcher
from tools.sar.coherence   import CoherenceEstimator


class StratifiedErrors:

    N_BINS        = 12
    MIN_BIN_COUNT = 30

    def __init__(self, run, result, coherence_window: tuple[int, int] = (7, 7)) -> None:
        self.loaded           = run
        self.result           = result
        self.coherence_window = coherence_window

    def _gt_active_count(self) -> np.ndarray:
        params = np.asarray(self.result.params_gt, dtype=np.float64)
        n_k    = self.loaded.n_gaussians
        amps   = np.stack([params[3 * k] for k in range(n_k)])

        return ParamMatcher.is_active(amps).sum(axis=0).astype(np.float64)

    def _mean_coherence(self) -> np.ndarray | None:
        inputs = self.loaded.complex_inputs
        n_sec  = self.loaded.n_secondaries

        if inputs is None or n_sec == 0:
            return None

        estimator = CoherenceEstimator(self.coherence_window)
        primary   = inputs[0]

        total = np.zeros(primary.shape, dtype=np.float64)
        for s in range(n_sec):
            magnitude, _phase = estimator.estimate(primary, inputs[1 + s])
            total += magnitude

        return total / n_sec

    def covariates(self) -> dict[str, tuple[np.ndarray, bool]]:
        out = {}

        if self.result.params_gt is not None:
            out["gt_active_count"] = (self._gt_active_count(), True)

        if self.loaded.complex_inputs is not None:
            out["primary_amplitude"] = (np.abs(self.loaded.complex_inputs[0]).astype(np.float64), False)

        coherence = self._mean_coherence()
        if coherence is not None:
            out["coherence"] = (coherence, False)

        dem = self.loaded.dataset.dem
        if dem is not None:
            grad_az, grad_rg = np.gradient(np.asarray(dem, dtype=np.float64))
            out["dem_slope"]  = (np.hypot(grad_az, grad_rg), False)

        if self.result.label_r2 is not None:
            out["label_r2"] = (np.asarray(self.result.label_r2, dtype=np.float64), False)

        if not out:
            raise ValueError("No covariate is available for error stratification; the run holds neither GT params, inputs, DEM nor a label-quality map")

        return out

    def _bin_rows(self, cov: np.ndarray, err: np.ndarray, discrete: bool) -> list[dict]:
        flat_cov = cov.reshape(-1)
        flat_err = err.reshape(-1)
        both     = np.isfinite(flat_cov) & np.isfinite(flat_err)
        flat_cov = flat_cov[both]
        flat_err = flat_err[both]

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

        rows = []
        for center, values in members:
            if values.size < self.MIN_BIN_COUNT:
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

        return rows

    @staticmethod
    def _tie_ranks(values: np.ndarray) -> np.ndarray:
        _uniques, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)

        starts   = np.cumsum(counts) - counts
        averaged = starts + (counts - 1) / 2.0

        return averaged[inverse]

    @staticmethod
    def _spearman(cov: np.ndarray, err: np.ndarray) -> float:
        flat_cov = cov.reshape(-1)
        flat_err = err.reshape(-1)
        both     = np.isfinite(flat_cov) & np.isfinite(flat_err)

        if both.sum() < 3:
            return float("nan")

        rank_cov = StratifiedErrors._tie_ranks(flat_cov[both])
        rank_err = StratifiedErrors._tie_ranks(flat_err[both])

        if rank_cov.std() == 0.0 or rank_err.std() == 0.0:
            return float("nan")

        return float(np.corrcoef(rank_cov, rank_err)[0, 1])

    def run(self) -> tuple[dict[str, list[dict]], dict[str, float]]:
        err     = np.asarray(self.result.pixel_mse, dtype=np.float64)
        tables  = {}
        scalars = {}

        for name, (cov, discrete) in self.covariates().items():
            rows          = self._bin_rows(cov, err, discrete)
            tables[name]  = {"rows": rows, "discrete": discrete}

            scalars[f"strat_{name}_spearman"] = self._spearman(cov, err)
            scalars[f"strat_{name}_n_bins"]   = float(len(rows))

        return tables, scalars
