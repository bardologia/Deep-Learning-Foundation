from __future__ import annotations

import numpy as np

from pipelines.backbone.inference.failure_modes import FailureModes
from pipelines.backbone.inference.stratified    import StratifiedErrors


class PresenceCalibration:

    N_BINS        = 10
    MIN_BIN_COUNT = 30

    def __init__(self, params_pred: np.ndarray, params_gt: np.ndarray, n_gaussians: int, assignment: np.ndarray | None = None) -> None:
        self.params_pred = params_pred
        self.params_gt   = params_gt
        self.n_gaussians = int(n_gaussians)
        self.assignment  = assignment

    def run(self) -> tuple[list[dict], dict]:
        pairs = FailureModes(self.params_pred, self.params_gt, self.n_gaussians, assignment=self.assignment).pair_predictions()

        active = pairs["act_pred"].reshape(-1)
        amps   = pairs["amp_pred"].reshape(-1)[active]
        hits   = pairs["pred_matched"].reshape(-1)[active].astype(np.float64)

        if amps.size < self.MIN_BIN_COUNT:
            raise ValueError(f"Only {amps.size} active predicted Gaussians; presence calibration needs at least {self.MIN_BIN_COUNT}")

        edges = np.unique(np.quantile(amps, np.linspace(0.0, 1.0, self.N_BINS + 1)))
        if edges.size < 3:
            raise ValueError("Predicted amplitudes are effectively constant; a reliability curve over amplitude is meaningless")

        index = np.clip(np.searchsorted(edges, amps, side="right") - 1, 0, edges.size - 2)

        rows = []
        for b in range(edges.size - 1):
            members = index == b
            if members.sum() < self.MIN_BIN_COUNT:
                continue
            rows.append({
                "amp_mean"  : float(amps[members].mean()),
                "n"         : int(members.sum()),
                "precision" : float(hits[members].mean()),
            })

        if not rows:
            raise ValueError(f"Every amplitude bin holds fewer than {self.MIN_BIN_COUNT} predictions; the region is too small to calibrate")

        scalars = {
            "presence_overall_precision" : float(hits.mean()),
            "presence_rank_corr"         : StratifiedErrors._spearman(amps, hits),
            "presence_precision_low"     : rows[0]["precision"],
            "presence_precision_high"    : rows[-1]["precision"],
            "presence_n_scored"          : float(amps.size),
        }

        return rows, scalars
