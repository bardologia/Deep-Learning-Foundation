"""Behavioral tests of the slot-activity threshold the inference analyses receive.

Places a ground-truth amplitude inside the open band between a 1e-4 and a 1e-3
label-run threshold, then checks that the higher threshold flips the Hungarian slot
assignment, the matched detection metrics and the failure-mode classification
relative to the lower one.
"""

from __future__ import annotations

import numpy as np

from pipelines.backbone.inference.failure_modes import FailureModes
from pipelines.backbone.inference.metrics       import Metrics, Result
from tools.metrics.gaussian_matching            import GaussianMatcher

from tests.conftest import SilentLogger


N_GAUSSIANS = 2
N_Z         = 16
SUB_AMP     = 5e-4
LOW_THR     = 1e-4
RAISED_THR  = 1e-3


def _params(slots: list[tuple[float, float, float]], height: int = 1, width: int = 1) -> np.ndarray:
    """Builds a constant parameter cube from per-slot (amplitude, mu, sigma) triples.

    Args:
        slots  : One (amplitude, mu, sigma) triple per Gaussian slot, mu and sigma in metres.
        height : Row extent of the cube.
        width  : Range extent of the cube.

    Returns:
        Parameter cube of shape (3 * len(slots), height, width).
    """
    cube = np.zeros((3 * len(slots), height, width), dtype=np.float32)

    for k, (amp, mu, sig) in enumerate(slots):
        cube[3 * k]     = amp
        cube[3 * k + 1] = mu
        cube[3 * k + 2] = sig

    return cube


def _result(params_pred: np.ndarray, params_gt: np.ndarray) -> Result:
    """Builds a Result around the parameter cubes with trivial curve and pixel maps.

    Args:
        params_pred : Predicted parameter cube of shape (3 * K, height, width).
        params_gt   : Ground-truth parameter cube of the same shape.

    Returns:
        A Result whose curves are identical flat cubes, so only the parameters matter.
    """
    H, W   = params_gt.shape[-2:]
    curves = np.ones((N_Z, H, W), dtype=np.float32)
    zeros  = np.zeros((H, W), dtype=np.float32)

    return Result(
        pred_curves        = curves,
        gt_curves          = curves,
        pixel_mse          = zeros,
        pixel_mae          = zeros,
        pixel_r2           = zeros,
        pixel_cosine       = zeros,
        pixel_peak_err_idx = zeros.astype(np.int32),
        cube_directory     = None,
        row_offset     = 0,
        col_offset       = 0,
        params_pred        = params_pred,
        params_gt          = params_gt,
    )


def _metrics(params_pred: np.ndarray, params_gt: np.ndarray, thr: float) -> dict:
    """Returns the matched-Gaussian metrics of the cube pair at the given activity threshold."""
    z_axis = np.linspace(-20.0, 80.0, N_Z).astype(np.float32)

    return Metrics(_result(params_pred, params_gt), z_axis, N_GAUSSIANS, 1, active_amp_thr=thr, logger=SilentLogger()).matched_gaussian_metrics()


def test_raised_threshold_flips_hungarian_assignment():
    """A sub-threshold GT slot claims the closest prediction at 1e-4 but is excluded at 1e-3."""
    gt   = _params([(SUB_AMP, 10.0, 5.0), (1.0, 12.0, 5.0)])
    pred = _params([(1.0, 10.0, 5.0), (0.0, 0.0, 5.0)])

    sel_default = GaussianMatcher(amp_threshold=LOW_THR).assignment(pred, gt, N_GAUSSIANS)
    sel_raised  = GaussianMatcher(amp_threshold=RAISED_THR).assignment(pred, gt, N_GAUSSIANS)

    assert sel_default[0, 0] == 0
    assert sel_raised [0, 0] == 1


def test_raised_threshold_changes_matched_metrics():
    """Deactivating the sub-threshold GT slot halves precision and empties one GT slot."""
    gt   = _params([(1.0, 10.0, 5.0), (SUB_AMP, 40.0, 5.0)])
    pred = _params([(1.0, 10.0, 5.0), (1.0, 40.0, 5.0)])

    default = _metrics(pred, gt, LOW_THR)
    raised  = _metrics(pred, gt, RAISED_THR)

    assert default["matched_recall"]    == 1.0
    assert default["matched_precision"] == 1.0
    assert default["matched_n_pairs"]   == 2.0

    assert raised["matched_recall"]    == 1.0
    assert raised["matched_precision"] == 0.5
    assert raised["matched_n_pairs"]   == 1.0


def test_raised_threshold_flips_failure_mode():
    """The pixel reads amplitude error at 1e-4 but hallucination at 1e-3."""
    gt   = _params([(1.0, 10.0, 5.0), (SUB_AMP, 40.0, 5.0)])
    pred = _params([(1.0, 10.0, 5.0), (1.0, 40.0, 5.0)])

    mode_default, _, scalars_default = FailureModes(pred, gt, N_GAUSSIANS, active_amp_thr=LOW_THR).run()
    mode_raised,  _, scalars_raised  = FailureModes(pred, gt, N_GAUSSIANS, active_amp_thr=RAISED_THR).run()

    assert FailureModes.MODES[int(mode_default[0, 0])] == "amplitude"
    assert FailureModes.MODES[int(mode_raised [0, 0])] == "hallucinated"

    assert scalars_default["failure_halluc_rate"] == 0.0
    assert scalars_raised ["failure_halluc_rate"] == 0.5


def test_result_assignment_seeds_matcher_with_threshold():
    """Result.assignment matches with the supplied threshold and caches that assignment."""
    gt   = _params([(SUB_AMP, 10.0, 5.0), (1.0, 12.0, 5.0)])
    pred = _params([(1.0, 10.0, 5.0), (0.0, 0.0, 5.0)])

    sel = _result(pred, gt).assignment(N_GAUSSIANS, RAISED_THR)

    assert sel[0, 0] == 1
