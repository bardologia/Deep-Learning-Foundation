from __future__ import annotations

import numpy as np

from pipelines.comparison.preprocessing_comparison import WindowMetrics


HEIGHT = 32


def _profile(peaks: list[tuple[float, float]]) -> np.ndarray:
    bins  = np.arange(HEIGHT, dtype=np.float64)
    curve = np.zeros(HEIGHT, dtype=np.float64)

    for centre, amplitude in peaks:
        curve += amplitude * np.exp(-0.5 * ((bins - centre) / 2.0) ** 2)

    return curve.astype(np.float32).reshape(HEIGHT, 1, 1)


def _spurious(profile: np.ndarray) -> float:
    metrics = WindowMetrics.__new__(WindowMetrics)
    return float(metrics._chunk_maps(profile)[2].ravel()[0])


def test_single_scatterer_scores_no_spurious_peak():
    assert _spurious(_profile([(16.0, 1.0)])) == 0.0


def test_competing_scatterer_is_counted_once():
    assert _spurious(_profile([(16.0, 1.0), (26.0, 0.5)])) == 1.0


def test_peak_below_the_significance_fraction_is_ignored():
    assert _spurious(_profile([(16.0, 1.0), (26.0, 0.1)])) == 0.0


def test_empty_profile_scores_no_spurious_peak():
    assert _spurious(np.zeros((HEIGHT, 1, 1), dtype=np.float32)) == 0.0
