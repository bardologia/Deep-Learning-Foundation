"""Reliability of predicted scatterer amplitude as a presence score.

Bins the active predicted Gaussians by amplitude and measures, per bin, the fraction that
Hungarian-matched a ground-truth scatterer, so amplitude can be read as a detection
confidence rather than only as a magnitude.
"""

from __future__ import annotations

import numpy as np

from tools.metrics.scoring import RankCorrelation


class PresenceCalibration:
    """Scores predicted-amplitude bins by the precision of the detections they contain.

    Attributes:
        N_BINS        : Number of quantile bins over predicted amplitude.
        MIN_BIN_COUNT : Minimum predictions a bin needs before it is reported.
        pairs         : Matched-pair arrays from FailureModes.pair_predictions.
    """

    N_BINS        = 10
    MIN_BIN_COUNT = 30

    def __init__(self, pairs: dict) -> None:
        """Stores the per-slot matched-pair arrays the calibration is computed from."""
        self.pairs = pairs

    def run(self) -> tuple[list[dict], dict]:
        """Bins active predictions by amplitude and measures the match precision per bin.

        Returns:
            Tuple of the per-bin rows, each carrying "amp_mean", "n" and "precision", and the
            scalar summary with overall precision, the Spearman correlation between amplitude
            and match success, the precision of the lowest and highest bins, and the number of
            scored predictions.

        Raises:
            ValueError : If too few active predictions exist, if the predicted amplitudes are effectively constant, or if no bin reaches MIN_BIN_COUNT.
        """
        active = self.pairs["act_pred"].reshape(-1)
        amps   = self.pairs["amp_pred"].reshape(-1)[active]
        hits   = self.pairs["pred_matched"].reshape(-1)[active].astype(np.float64)

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
            "presence_rank_corr"         : RankCorrelation.spearman(amps, hits),
            "presence_precision_low"     : rows[0]["precision"],
            "presence_precision_high"    : rows[-1]["precision"],
            "presence_n_scored"          : float(amps.size),
        }

        return rows, scalars
