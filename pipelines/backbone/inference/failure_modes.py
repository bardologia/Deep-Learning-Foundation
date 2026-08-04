from __future__ import annotations

import numpy as np

from tools.loss.param_loss           import ParamMatcher
from tools.metrics.gaussian_matching import GaussianMatcher


class FailureModes:

    MODES = ("ok", "missed", "hallucinated", "position", "width", "amplitude")

    MATCH_TOL     = 5.0
    POSITION_TOL  = 2.0
    WIDTH_REL_TOL = 0.5
    AMP_REL_TOL   = 0.5
    N_SEP_BINS    = 8
    MIN_BIN_COUNT = 30

    def __init__(self, params_pred: np.ndarray, params_gt: np.ndarray, n_gaussians: int, assignment: np.ndarray | None = None) -> None:
        self.params_pred = np.asarray(params_pred, dtype=np.float64)
        self.params_gt   = np.asarray(params_gt, dtype=np.float64)
        self.n_gaussians = int(n_gaussians)
        self.assignment  = assignment

        self.shape = self.params_pred.shape[1:]

    def _channels(self, params: np.ndarray, offset: int) -> np.ndarray:
        return np.stack([params[3 * k + offset] for k in range(self.n_gaussians)], axis=0).reshape(self.n_gaussians, -1)

    def pair_predictions(self) -> dict:
        n_K = self.n_gaussians
        sel = GaussianMatcher().assignment(self.params_pred, self.params_gt, n_K) if self.assignment is None else self.assignment

        amp_pred = self._channels(self.params_pred, 0)
        mu_pred  = self._channels(self.params_pred, 1)
        sig_pred = self._channels(self.params_pred, 2)
        amp_gt   = self._channels(self.params_gt, 0)
        mu_gt    = self._channels(self.params_gt, 1)
        sig_gt   = self._channels(self.params_gt, 2)

        act_pred = ParamMatcher.is_active(amp_pred)
        act_gt   = ParamMatcher.is_active(amp_gt)

        n_pixels     = mu_pred.shape[1]
        rows         = np.arange(n_pixels)
        gt_matched   = np.zeros((n_K, n_pixels), dtype=bool)
        pred_matched = np.zeros((n_K, n_pixels), dtype=bool)
        any_pos      = np.zeros(n_pixels, dtype=bool)
        any_width    = np.zeros(n_pixels, dtype=bool)
        any_amp      = np.zeros(n_pixels, dtype=bool)

        for i in range(n_K):
            j     = sel[:, i]
            both  = act_pred[i] & act_gt[j, rows]
            d_mu  = np.abs(mu_pred[i] - mu_gt[j, rows])
            valid = both & (d_mu <= self.MATCH_TOL)

            gt_matched[j[valid], rows[valid]] = True

            rel_sig = np.abs(sig_pred[i] - sig_gt[j, rows]) / np.maximum(np.abs(sig_gt[j, rows]), 1e-6)
            rel_amp = np.abs(amp_pred[i] - amp_gt[j, rows]) / np.maximum(np.abs(amp_gt[j, rows]), 1e-6)

            any_pos   |= valid & (d_mu > self.POSITION_TOL)
            any_width |= valid & (rel_sig > self.WIDTH_REL_TOL)
            any_amp   |= valid & (rel_amp > self.AMP_REL_TOL)

            pred_matched[i] = act_pred[i] & valid

        n_missed = (act_gt & ~gt_matched).sum(axis=0)
        n_halluc = (act_pred & ~pred_matched).sum(axis=0)

        return {
            "act_gt"       : act_gt,
            "act_pred"     : act_pred,
            "amp_pred"     : amp_pred,
            "mu_gt"        : mu_gt,
            "pred_matched" : pred_matched,
            "n_missed"     : n_missed,
            "n_halluc"     : n_halluc,
            "any_pos"      : any_pos,
            "any_width"    : any_width,
            "any_amp"      : any_amp,
        }

    def _mode_map(self, pairs: dict) -> np.ndarray:
        mode = np.zeros(pairs["n_missed"].size, dtype=np.int8)

        mode[pairs["any_amp"]]        = self.MODES.index("amplitude")
        mode[pairs["any_width"]]      = self.MODES.index("width")
        mode[pairs["any_pos"]]        = self.MODES.index("position")
        mode[pairs["n_halluc"] > 0]   = self.MODES.index("hallucinated")
        mode[pairs["n_missed"] > 0]   = self.MODES.index("missed")

        return mode.reshape(self.shape)

    def _min_gt_separation(self, pairs: dict) -> np.ndarray:
        mu_gt  = pairs["mu_gt"]
        act_gt = pairs["act_gt"]
        n_K    = self.n_gaussians

        sep = np.full(mu_gt.shape[1], np.inf)
        for a in range(n_K):
            for b in range(a + 1, n_K):
                both     = act_gt[a] & act_gt[b]
                distance = np.abs(mu_gt[a] - mu_gt[b])
                sep      = np.where(both, np.minimum(sep, distance), sep)

        return sep

    def _miss_by_separation(self, pairs: dict) -> list[dict]:
        sep       = self._min_gt_separation(pairs)
        missed    = pairs["n_missed"] > 0
        has_pairs = np.isfinite(sep)

        if has_pairs.sum() < self.MIN_BIN_COUNT:
            return []

        values = sep[has_pairs]
        misses = missed[has_pairs]
        edges  = np.unique(np.quantile(values, np.linspace(0.0, 1.0, self.N_SEP_BINS + 1)))

        if edges.size < 3:
            return []

        index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)

        rows = []
        for b in range(edges.size - 1):
            members = index == b
            if members.sum() < self.MIN_BIN_COUNT:
                continue
            rows.append({
                "center"    : (float(edges[b]) + float(edges[b + 1])) / 2.0,
                "n"         : int(members.sum()),
                "miss_rate" : float(misses[members].mean()),
            })

        return rows

    def _scalars(self, pairs: dict, mode_map: np.ndarray) -> dict:
        n_pixels   = mode_map.size
        n_gt_act   = int(pairs["act_gt"].sum())
        n_pred_act = int(pairs["act_pred"].sum())

        out = {
            "failure_miss_rate"   : float(pairs["n_missed"].sum()) / n_gt_act if n_gt_act else float("nan"),
            "failure_halluc_rate" : float(pairs["n_halluc"].sum()) / n_pred_act if n_pred_act else float("nan"),
        }

        for m, name in enumerate(self.MODES):
            out[f"failure_frac_{name}"] = float((mode_map == m).sum()) / n_pixels

        return out

    def run(self) -> tuple[np.ndarray, list[dict], dict]:
        pairs    = self.pair_predictions()
        mode_map = self._mode_map(pairs)
        by_sep   = self._miss_by_separation(pairs)
        scalars  = self._scalars(pairs, mode_map)

        return mode_map, by_sep, scalars
