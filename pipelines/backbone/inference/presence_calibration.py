
from __future__ import annotations

import numpy as np

from tools.metrics.scoring import RankCorrelation


class PresenceCalibration:

    N_BINS        = 10
    MIN_BIN_COUNT = 30

    def __init__(self, pairs: dict) -> None:
        self.pairs = pairs

    def run(self) -> tuple[list[dict], dict]:
        active = self.pairs["act_pred"].reshape(-1)
        amps   = self.pairs["amp_pred"].reshape(-1)[active]
        hits   = self.pairs["pred_matched"].reshape(-1)[active].astype(np.float64)

        edges = np.unique(np.quantile(amps, np.linspace(0.0, 1.0, self.N_BINS + 1)))

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

        scalars = {
            "presence_overall_precision" : float(hits.mean()),
            "presence_rank_corr"         : RankCorrelation.spearman(amps, hits),
            "presence_precision_low"     : rows[0]["precision"],
            "presence_precision_high"    : rows[-1]["precision"],
            "presence_n_scored"          : float(amps.size),
        }

        return rows, scalars
