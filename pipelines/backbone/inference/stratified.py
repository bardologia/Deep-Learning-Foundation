"""Stratification of per-pixel curve error by scene covariates.

Bins the per-pixel curve MSE against covariates the scene itself provides — ground-truth
component count, primary amplitude, aux gradient and label fit quality — so error can be
attributed to scene difficulty rather than to the model alone.
"""

from __future__ import annotations

import numpy as np

from tools.loss.param_loss import ParamMatcher
from tools.metrics.scoring import RankCorrelation


class StratifiedErrors:
    """Bins per-pixel curve MSE against scene covariates and scores their rank correlation.

    Attributes:
        N_BINS         : Number of quantile bins used for continuous covariates.
        MIN_BIN_COUNT  : Minimum pixels a bin needs before it is reported.
        loaded         : Loaded run supplying the complex stack, the aux channel and the slot count.
        result         : Inference result supplying the ground-truth parameters and the per-pixel MSE map.
        active_amp_thr : Physical amplitude at or above which a ground-truth Gaussian slot counts as active, read from the loaded run.
    """

    N_BINS        = 12
    MIN_BIN_COUNT = 30

    def __init__(self, run, result) -> None:
        """Stores the run and the inference result, taking the active-amplitude threshold from the run."""
        self.loaded         = run
        self.result         = result
        self.active_amp_thr = float(run.activity_threshold)

    def _gt_active_count(self) -> np.ndarray:
        """Returns the number of active ground-truth Gaussians per pixel, of shape (rows, cols)."""
        params = np.asarray(self.result.params_gt, dtype=np.float64)
        n_k    = self.loaded.n_gaussians
        amps   = np.stack([params[3 * k] for k in range(n_k)])

        return ParamMatcher.is_active(amps, self.active_amp_thr).sum(axis=0).astype(np.float64)

    def covariates(self) -> dict[str, tuple[np.ndarray, bool]]:
        """Collects every covariate map available for this run.

        Returns:
            Mapping from covariate name to its map of shape (rows, cols) and a flag
            marking it as discrete, covering the ground-truth active count, primary
            amplitude, aux gradient magnitude and label fit R².

        Raises:
            ValueError : If the run offers no covariate at all.
        """
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

    def _bin_rows(self, cov: np.ndarray, err: np.ndarray, discrete: bool) -> list[dict]:
        """Bins the error by covariate and summarises each bin by median and quartiles.

        Args:
            cov      : Covariate map of shape (rows, cols).
            err      : Per-pixel error map of the same shape.
            discrete : Whether to bin on distinct covariate values instead of quantile edges.

        Returns:
            Per-bin records carrying "center", "n", "median", "q25" and "q75", skipping bins
            below MIN_BIN_COUNT.

        Raises:
            ValueError : If no finite pairs exist, the covariate is effectively constant, or every bin falls below MIN_BIN_COUNT.
        """
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

    def run(self) -> tuple[dict[str, list[dict]], dict[str, float]]:
        """Stratifies the per-pixel MSE by every available covariate.

        Returns:
            Tuple of a per-covariate table, each holding its binned "rows" and its "discrete"
            flag, and the scalars giving the Spearman correlation of covariate against error
            and the number of reported bins.
        """
        err     = np.asarray(self.result.pixel_mse, dtype=np.float64)
        tables  = {}
        scalars = {}

        for name, (cov, discrete) in self.covariates().items():
            rows          = self._bin_rows(cov, err, discrete)
            tables[name]  = {"rows": rows, "discrete": discrete}

            scalars[f"strat_{name}_spearman"] = RankCorrelation.spearman(cov, err)
            scalars[f"strat_{name}_n_bins"]   = float(len(rows))

        return tables, scalars
