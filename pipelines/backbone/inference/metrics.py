"""Scores a stitched backbone prediction against its ground truth.

Holds the `Result` container that carries every cube and per-pixel map produced by an
inference pass, the `Metrics` calculator that turns them into the flat metric
dictionary written to metrics.json (curve errors, per-slice SSIM, per-z
profiles, Gaussian-parameter statistics, the classical-baseline comparison and the
label-quality split) and the `BoundedProcessPool` scheduler both
SSIM sweeps run their per-slice workers on.
"""

from __future__ import annotations

import os
from collections        import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses        import dataclass
from pathlib            import Path
from typing             import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm            import tqdm

from tools.data.io                   import FileIO
from tools.data.preprocessing        import ProfileNormalizer
from tools.loss.param_loss           import ParamMatcher
from tools.metrics.gaussian_matching import GaussianMatcher
from tools.metrics.slot_organization import SlotOrganization
from tools.metrics.scoring           import R2


@dataclass
class Result:
    """Stitched output of one inference pass together with every optional evaluation.

    Attributes:
        pred_curves        : Predicted cube of shape (n_z, rows, cols), denormalised intensity.
        gt_curves          : Ground-truth Gaussian cube of the same shape.
        pixel_mse          : Per-pixel curve MSE of shape (rows, cols).
        pixel_mae          : Per-pixel curve MAE of shape (rows, cols).
        pixel_r2           : Per-pixel curve R² of shape (rows, cols).
        pixel_cosine       : Per-pixel curve cosine similarity of shape (rows, cols).
        pixel_peak_err_idx : Per-pixel absolute peak-bin displacement of shape (rows, cols).
        cube_directory     : Directory the stitched cubes were written to.
        row_offset         : Row index of the crop origin in scene coordinates.
        col_offset         : Column index of the crop origin in scene coordinates.
        params_pred        : Predicted Gaussian parameters of shape (3 * K, rows, cols), or None.
        params_gt          : Ground-truth Gaussian parameters of the same shape, or None.
        params_source      : Whether the parameters came from the model ("model") or a curve refit ("extracted").
        extract_r2         : Per-pixel refit R² of shape (rows, cols) when parameters were extracted.
        extract_floor      : Matched-Gaussian metrics of the extractor run on GT curves, its own error floor.
        embedding_maps     : Per-pixel JEPA embedding maps, or None.
        label_r2           : Per-pixel R² of the GT curves against the raw cube, of shape (rows, cols).
        flip_consistency   : Per-pixel flip-equivariance disagreement of shape (rows, cols).
        stratified         : Covariate-stratified error tables keyed by covariate name.
        failure_mode_map   : Per-pixel failure-mode class index of shape (rows, cols).
        failure_pairs      : Matched prediction/GT pair records feeding the failure and presence analyses.
        miss_by_sep        : Miss rate binned by GT peak separation.
        presence_rows      : Presence-reliability bins relating predicted amplitude to precision.
        param_assignment   : Cached Hungarian assignment of predicted to GT slots.
        pred_norm_cube     : Cached unit-area normalisation of `pred_curves`.
        gt_norm_cube       : Cached unit-area normalisation of `gt_curves`.
    """

    pred_curves        : np.ndarray
    gt_curves          : np.ndarray

    pixel_mse          : np.ndarray
    pixel_mae          : np.ndarray
    pixel_r2           : np.ndarray
    pixel_cosine       : np.ndarray
    pixel_peak_err_idx : np.ndarray

    cube_directory     : Path
    row_offset     : int
    col_offset       : int

    params_pred      : Optional[np.ndarray]        = None
    params_gt        : Optional[np.ndarray]        = None
    params_source    : str                         = "model"
    extract_r2       : Optional[np.ndarray]        = None
    extract_floor    : Optional[Dict[str, float]]  = None
    embedding_maps   : Optional[object]            = None
    label_r2         : Optional[np.ndarray]        = None
    flip_consistency : Optional[np.ndarray]        = None
    stratified       : Optional[dict]              = None
    failure_mode_map : Optional[np.ndarray]        = None
    failure_pairs    : Optional[dict]              = None
    miss_by_sep      : Optional[list]              = None
    presence_rows    : Optional[list]              = None
    param_assignment : Optional[np.ndarray]        = None
    pred_norm_cube   : Optional[np.ndarray]        = None
    gt_norm_cube     : Optional[np.ndarray]        = None

    def assignment(self, n_gaussians: int, amp_threshold: float) -> np.ndarray:
        """Returns the cached predicted-to-GT slot assignment of shape (n_pixels, K), matching on first call.

        The assignment is matched once and cached, so every caller within one inference
        chan must supply the same activity threshold.

        Args:
            n_gaussians   : Gaussian count K.
            amp_threshold : Physical amplitude at or above which a Gaussian slot counts as active.

        Returns:
            Per-pixel GT slot index chosen for each predicted slot, of shape (n_pixels, K).
        """

        if self.param_assignment is None:
            self.param_assignment = GaussianMatcher(amp_threshold=amp_threshold).assignment(self.params_pred, self.params_gt, n_gaussians)

        return self.param_assignment

    def pred_unit_area(self) -> np.ndarray:
        """Returns the predicted cube normalised to unit area per profile, cached after the first call."""

        if self.pred_norm_cube is None:
            self.pred_norm_cube = ProfileNormalizer.unit_area(self.pred_curves)

        return self.pred_norm_cube

    def gt_unit_area(self) -> np.ndarray:
        """Returns the ground-truth cube normalised to unit area per profile, cached after the first call."""

        if self.gt_norm_cube is None:
            self.gt_norm_cube = ProfileNormalizer.unit_area(self.gt_curves)

        return self.gt_norm_cube


class BoundedProcessPool:
    """Runs a picklable worker over a payload stream with a bounded submission window.

    Payloads are pulled from their iterator only as slots free up, so the parent process
    never materialises more than `2 * num_workers` of them at once. Both SSIM sweeps
    depend on that bound, since one payload carries a full 2-D cube slice.

    Attributes:
        num_workers : Process count, capped at the machine's CPU count.
    """

    def __init__(self, num_workers: int) -> None:
        """Stores the worker count, capped at the machine's CPU count and at least one.

        Args:
            num_workers : Requested worker count.
        """

        self.num_workers = max(1, min(int(num_workers), os.cpu_count() or 1))

    def results(self, worker: Callable, payloads: Iterable, total: int, description: str) -> List:
        """Maps a worker over the payloads in submission order, showing a slice progress bar.

        Args:
            worker      : Picklable callable taking one payload and returning one result.
            payloads    : Iterable yielding exactly `total` payloads, consumed lazily.
            total       : Number of payloads to submit and collect.
            description : Progress-bar label.

        Returns:
            Worker return values in payload order, of length `total`.
        """

        n_workers = min(self.num_workers, max(1, total))
        in_flight = 2 * n_workers
        stream    = iter(payloads)
        submitted = 0
        out       = []

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            pending: deque = deque()

            with tqdm(total=total, desc=description, unit="slice", leave=False) as pbar:
                for _ in range(total):
                    while submitted < total and len(pending) < in_flight:
                        pending.append(pool.submit(worker, next(stream)))
                        submitted += 1

                    out.append(pending.popleft().result())
                    pbar.update(1)

        return out


class Metrics:
    """Computes every scalar and per-pixel metric describing one inference result.

    Attributes:
        result         : The inference result being scored.
        x_axis         : Z axis in metres, of shape (n_z,).
        n_gaussians    : Gaussian count K.
        x_step         : Z bin spacing in metres.
        num_workers    : Process count used for the parallel SSIM sweep.
        active_amp_thr : Physical amplitude at or above which a Gaussian slot counts as active.
    """

    def __init__(
        self,
        result         : Result,
        z_axis         : np.ndarray,
        n_gaussians    : int,
        cpu_workers    : int,
        active_amp_thr : float = ParamMatcher.ACTIVE_AMP_THR,
    ) -> None:
        """Stores the result and derives the z step and SSIM worker count.

        Args:
            result         : Inference result carrying the cubes and per-pixel maps.
            z_axis         : Z axis in metres, of shape (n_z,).
            n_gaussians    : Gaussian count K.
            cpu_workers    : Requested worker count, capped at the machine's CPU count.
            active_amp_thr : Physical amplitude at or above which a Gaussian slot counts as active.
        """

        self.result         = result
        self.z_axis         = z_axis
        self.n_gaussians    = n_gaussians
        self.x_step         = float(z_axis[1] - z_axis[0])
        self.num_workers    = max(1, min(int(cpu_workers), os.cpu_count() or 1))
        self.active_amp_thr = float(active_amp_thr)
        self._aligned       = None

    @staticmethod
    def curve_pixel_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, np.ndarray]:
        """Computes the per-pixel curve error maps between two cubes.

        Args:
            pred : Predicted cube of shape (n_z, rows, cols).
            ref  : Reference cube of the same shape.

        Returns:
            Mapping with keys "mse", "mae", "r2", "cos" and "peak", each of shape (rows, cols);
            "peak" is the absolute displacement of the argmax z bin.
        """

        diff   = pred - ref
        mse    = (diff * diff).mean(axis=0).astype(np.float32)
        mae    = np.abs(diff).mean(axis=0).astype(np.float32)
        r2     = R2.pixel_map(pred, ref, axis=0)
        dot    = (pred * ref).sum(axis=0)
        norm_p = np.sqrt((pred * pred).sum(axis=0)) + 1e-8
        norm_r = np.sqrt((ref  * ref ).sum(axis=0)) + 1e-8
        cos    = (dot / (norm_p * norm_r)).astype(np.float32)
        peak   = np.abs(pred.argmax(axis=0) - ref.argmax(axis=0)).astype(np.int32)

        return {"mse": mse, "mae": mae, "r2": r2, "cos": cos, "peak": peak}

    @staticmethod
    def write_json(metrics: Dict[str, object], path: Path) -> Path:
        """Writes the metric dictionary to `path` as indented JSON and returns that path."""

        return FileIO.save_json(metrics, path, indent=4)

    @staticmethod
    def select_pixels(
        metric_map : np.ndarray,
        *,
        n_best     : int,
        n_worst    : int,
        n_random   : int,
        seed       : int = 0,
    ) -> Dict[str, np.ndarray]:
        """Selects best, worst and random pixel coordinates by a per-pixel metric.

        Lower metric values count as better, and the random pool excludes the best and
        worst picks so the three sets are disjoint.

        Args:
            metric_map : Per-pixel metric of shape (rows, cols); non-finite pixels are ignored.
            n_best     : Number of lowest-metric pixels to select.
            n_worst    : Number of highest-metric pixels to select, ordered worst first.
            n_random   : Number of random pixels to draw from the remaining pool.
            seed       : Seed of the random generator drawing the random set.

        Returns:
            Mapping with keys "best", "worst" and "random", each an array of (rows, cols)
            index pairs of shape (n_selected, 2).
        """

        flat      = metric_map.reshape(-1)
        H, W      = metric_map.shape
        rng       = np.random.default_rng(seed)
        valid_idx = np.where(np.isfinite(flat))[0]
        order     = valid_idx[np.argsort(flat[valid_idx])]

        best_flat  = order[:n_best]
        worst_flat = order[-n_worst:][::-1] if n_worst > 0 else order[:0]

        pool      = np.setdiff1d(valid_idx, np.concatenate([best_flat, worst_flat]), assume_unique=False)
        n_random  = min(n_random, pool.size)
        rand_flat = rng.choice(pool, size=n_random, replace=False) if n_random > 0 else np.array([], dtype=np.int64)

        return {
            "best"   : Metrics._flat_to_yx(best_flat,  W),
            "worst"  : Metrics._flat_to_yx(worst_flat, W),
            "random" : Metrics._flat_to_yx(rand_flat,  W),
        }

    @staticmethod
    def _flat_to_yx(flat_idx: np.ndarray, width: int) -> np.ndarray:
        """Returns flat indices unravelled into (rows, cols) pairs of shape (n, 2)."""

        return np.stack([(flat_idx // width).astype(np.int32), (flat_idx % width).astype(np.int32)], axis=1)

    def _curve_scalar_metrics(self, pred: np.ndarray, ref: np.ndarray, suffix: str) -> Dict[str, float]:
        """Computes cube-wide MSE, MAE, RMSE, R² and PSNR between two cubes.

        Args:
            pred   : Predicted cube of shape (n_z, rows, cols).
            ref    : Reference cube of the same shape.
            suffix : Tag appended to every metric key, naming the reference variant.

        Returns:
            Scalar metrics keyed "curve_mse_<suffix>", "curve_mae_<suffix>", "curve_rmse_<suffix>",
            "overall_r2_<suffix>" and "psnr_db_<suffix>".
        """

        diff       = pred - ref
        mse        = float((diff * diff).mean(dtype=np.float64))
        mae        = float(np.abs(diff).mean(dtype=np.float64))
        ref_mean   = float(ref.mean(dtype=np.float64))
        overall_r2 = 1.0 - float((diff * diff).sum(dtype=np.float64)) / (float(((ref - ref_mean) ** 2).sum(dtype=np.float64)) + 1e-12)

        return {
            f"curve_mse_{suffix}"  : mse,
            f"curve_mae_{suffix}"  : mae,
            f"curve_rmse_{suffix}" : float(np.sqrt(mse)),
            f"overall_r2_{suffix}" : float(overall_r2),
            f"psnr_db_{suffix}"    : self._psnr(pred, ref),
        }

    @staticmethod
    def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
        """Returns the peak signal-to-noise ratio in dB, using the reference range as peak.

        Args:
            pred : Predicted cube of shape (n_z, rows, cols).
            ref  : Reference cube of the same shape.

        Returns:
            PSNR in dB, infinity for an exact match and NaN for a constant reference.
        """

        diff = pred - ref
        mse  = float((diff * diff).mean(dtype=np.float64))

        if mse <= 0.0:
            return float("inf")

        data_range = float(ref.max() - ref.min())
        if data_range <= 0.0:
            return float("nan")

        return 10.0 * np.log10(data_range * data_range / mse)

    def _expand_pixel_stats(self, tagged_arrays: Tuple[Tuple[str, np.ndarray], ...]) -> Dict[str, float]:
        """Flattens moments and percentiles of several per-pixel maps into one tagged dictionary.

        Args:
            tagged_arrays : Pairs of metric tag and per-pixel map of shape (rows, cols).

        Returns:
            Scalar metrics keyed "<tag>_<statistic>" for each supplied map.
        """

        out: Dict[str, float] = {}

        for tag, arr in tagged_arrays:
            out.update({f"{tag}_{k}": v for k, v in self._basic_stats(arr).items()})
            out.update({f"{tag}_{k}": v for k, v in self._percentiles(arr).items()})

        return out

    @staticmethod
    def _basic_stats(x: np.ndarray) -> Dict[str, float]:
        """Returns mean, standard deviation, median, minimum and maximum of an array."""

        flat = x.reshape(-1)

        return {
            "mean"   : float(flat.mean(dtype=np.float64)),
            "std"    : float(flat.std(dtype=np.float64)),
            "median" : float(np.median(flat)),
            "min"    : float(flat.min()),
            "max"    : float(flat.max()),
        }

    @staticmethod
    def _percentiles(x: np.ndarray, qs: Tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
        """Returns the requested percentiles of an array, keyed "p<q>"."""

        flat = x.reshape(-1)
        out  = {}
        p    = np.percentile(flat, qs)

        for q, v in zip(qs, p):
            out[f"p{q}"] = float(v)

        return out

    def _slice_ssim(self, pred : np.ndarray, ref : np.ndarray, z_indices : Optional[np.ndarray], col_indices : Optional[np.ndarray], row_indices : Optional[np.ndarray], prefix : str) -> Dict[str, float]:
        """Computes SSIM for every requested slice of both cubes, in parallel over a process pool.

        Args:
            pred          : Predicted cube of shape (n_z, row, col).
            ref           : Reference cube of the same shape.
            z_indices  : Z indices to score, or None to skip that axis.
            col_indices : Range indices to score, or None to skip that axis.
            row_indices    : Row indices to score, or None to skip that axis.
            prefix        : Tag naming the reference variant, embedded in every metric key.

        Returns:
            Scalar metrics keyed "ssim_<prefix>_<axis>_<i>" per slice plus a
            "ssim_<prefix>_<axis>_mean" over the finite values of each axis.
        """

        out   : Dict[str, float]                = {}
        specs : list[tuple[str, str, int, int]] = []

        self._enqueue_specs(specs, prefix, z_indices,  axis=0, label="z")
        self._enqueue_specs(specs, prefix, col_indices, axis=2, label="col")
        self._enqueue_specs(specs, prefix, row_indices,    axis=1, label="row")

        if not specs:
            return out

        scored = BoundedProcessPool(self.num_workers).results(Metrics._ssim_worker, self._ssim_payloads(pred, ref, specs), len(specs), f"SSIM ({prefix})")

        for key, value in scored:
            out[key] = value

        for label in {spec[1] for spec in specs}:
            vals = [out[key] for key, lbl, _axis, _index in specs if lbl == label and np.isfinite(out[key])]
            out[f"ssim_{prefix}_{label}_mean"] = float(np.mean(vals)) if vals else float("nan")

        return out

    def _ssim_payloads(self, pred: np.ndarray, ref: np.ndarray, specs: list) -> Iterator[tuple]:
        """Yields one worker payload per SSIM spec, slicing both cubes on demand.

        Args:
            pred  : Predicted cube of shape (n_z, rows, cols).
            ref   : Reference cube of the same shape.
            specs : List of (metric key, axis label, cube axis, slice index) tuples.

        Yields:
            Tuple of metric key, predicted 2-D slice and reference 2-D slice.
        """

        for key, _label, axis, index in specs:
            yield key, self._axis_slice(pred, axis, index), self._axis_slice(ref, axis, index)

    @staticmethod
    def _enqueue_specs(
        specs  : list,
        prefix : str,
        indices: Optional[np.ndarray],
        axis   : int,
        label  : str,
    ) -> None:
        """Appends one SSIM work item per slice index to the spec list, in place.

        Args:
            specs   : List of (metric key, axis label, cube axis, slice index) tuples, extended in place.
            prefix  : Tag naming the reference variant.
            indices : Slice indices to enqueue, or None to enqueue nothing.
            axis    : Cube axis the indices address.
            label   : Axis name used in the metric key.
        """

        if indices is None:
            return

        for i, idx in enumerate(indices):
            specs.append((f"ssim_{prefix}_{label}_{i}", label, axis, int(idx)))

    @staticmethod
    def _axis_slice(cube: np.ndarray, axis: int, index: int) -> np.ndarray:
        """Returns the 2-D slice of a cube taken at `index` along `axis`."""

        if axis == 0:
            return cube[index, :, :]
        if axis == 1:
            return cube[:, index, :]

        return cube[:, :, index]

    @staticmethod
    def _ssim_worker(args: tuple) -> tuple[str, float]:
        """Computes SSIM for one slice pair inside a worker process.

        The window size is shrunk to the largest odd value the slice can hold, capped at 7.

        Args:
            args : Tuple of metric key, predicted 2-D slice and reference 2-D slice.

        Returns:
            Tuple of the metric key and its SSIM value.
        """

        key, pred_slice, ref_slice = args

        pred_slice = pred_slice.astype(np.float64)
        ref_slice  = ref_slice.astype(np.float64)
        data_range = float(ref_slice.max() - ref_slice.min())
        min_side   = min(pred_slice.shape)
        win_size   = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        value      = float(ssim(ref_slice, pred_slice, data_range=data_range, win_size=win_size))

        return key, value

    def _expand_z(self, pred: np.ndarray, gt: np.ndarray, suffix: str) -> Dict[str, float]:
        """Flattens the per-z metric series into per-bin scalars plus their means.

        Args:
            pred   : Predicted cube of shape (n_z, rows, cols).
            gt     : Ground-truth cube of the same shape.
            suffix : Tag naming the reference variant.

        Returns:
            Scalar metrics keyed "<metric>_<suffix>_<bin>" and "<metric>_<suffix>_mean".
        """

        out: Dict[str, float] = {}

        for metric_name, arr in self._z_metrics(pred, gt, suffix=suffix).items():
            out.update({f"{metric_name}_{i}": float(v) for i, v in enumerate(arr)})
            out[f"{metric_name}_mean"] = float(np.nanmean(arr))

        return out

    def _z_metrics(self, pred: np.ndarray, gt: np.ndarray, suffix: str = "gt") -> Dict[str, np.ndarray]:
        """Computes MAE, RMSE, R² and cross-entropy per z bin over all pixels.

        Cross-entropy compares the two cubes after normalising each profile to a
        distribution along z.

        Args:
            pred   : Predicted cube of shape (n_z, rows, cols).
            gt     : Ground-truth cube of the same shape.
            suffix : Tag naming the reference variant.

        Returns:
            Mapping of "z_<metric>_<suffix>" to an array of shape (n_z,).
        """

        P = pred.reshape(pred.shape[0], -1)
        G = gt  .reshape(gt  .shape[0], -1)

        diff_g  = P - G
        mae_gt  = np.abs(diff_g).mean(axis=1, dtype=np.float64)
        rmse_gt = np.sqrt((diff_g ** 2).mean(axis=1, dtype=np.float64))
        g_var   = ((G - G.mean(axis=1, keepdims=True, dtype=np.float64)) ** 2).sum(axis=1, dtype=np.float64) + 1e-12
        r2_gt   = 1.0 - (diff_g ** 2).sum(axis=1, dtype=np.float64) / g_var

        gt_prob   = G / G.sum(axis=0, keepdims=True, dtype=np.float64).clip(1e-12, None)
        pred_prob = P / P.sum(axis=0, keepdims=True, dtype=np.float64).clip(1e-12, None)

        log_pp = np.log(pred_prob.clip(1e-12, None))
        ce_gt  = -(gt_prob * log_pp).mean(axis=1, dtype=np.float64)

        return {
            f"z_mae_{suffix}"  : mae_gt,
            f"z_rmse_{suffix}" : rmse_gt,
            f"z_r2_{suffix}"   : r2_gt,
            f"z_ce_{suffix}"   : ce_gt,
        }

    @staticmethod
    def roughness_map(curves: np.ndarray) -> np.ndarray:
        """Returns the mean squared second difference along z, of shape (rows, cols).

        Args:
            curves : Cube of shape (n_z, rows, cols).

        Returns:
            Per-pixel curve roughness.
        """

        second = np.diff(curves, n=2, axis=0)
        return (second * second).mean(axis=0, dtype=np.float64).astype(np.float32)

    def _roughness_stats(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Compares predicted against ground-truth curve roughness.

        Args:
            pred : Predicted cube of shape (n_z, rows, cols).
            gt   : Ground-truth cube of the same shape.

        Returns:
            Mean roughness of each cube, their ratio, and the pixel fraction where the
            prediction is more than twice as rough as the ground truth.
        """

        rough_pred = self.roughness_map(pred)
        rough_gt   = self.roughness_map(gt)

        pred_mean = float(rough_pred.mean(dtype=np.float64))
        gt_mean   = float(rough_gt.mean(dtype=np.float64))
        excess    = rough_pred > (2.0 * rough_gt)

        return {
            "roughness_pred_mean"   : pred_mean,
            "roughness_gt_mean"     : gt_mean,
            "roughness_ratio"       : pred_mean / gt_mean if gt_mean > 0.0 else float("nan"),
            "roughness_excess_frac" : float(excess.mean()),
        }

    def _param_assignment(self) -> np.ndarray:
        """Returns the cached predicted-to-GT slot assignment for this result."""

        return self.result.assignment(self.n_gaussians, self.active_amp_thr)

    def _aligned_prediction(self) -> np.ndarray:
        """Returns the predicted parameter cube reordered into GT slot order, cached after the first call."""

        if self._aligned is None:
            self._aligned = GaussianMatcher(amp_threshold=self.active_amp_thr).aligned_prediction(self.result.params_pred, self.result.params_gt, self.n_gaussians, assignment=self._param_assignment())

        return self._aligned

    def _active_count_stats(self) -> Dict[str, float]:
        """Compares how many Gaussians the model activates per pixel against the ground truth.

        Returns:
            Per-slot active fractions, overall active fractions and counts, the fraction of
            pixels where the predicted component count is exact, under or over, and the exact-count
            accuracy conditioned on the GT and on the predicted count.
        """

        pp      = self.result.params_pred
        pg      = self.result.params_gt
        n_K     = self.n_gaussians
        aligned = self._aligned_prediction()
        out: Dict[str, float] = {}

        H, W       = pg.shape[-2:]
        gt_count   = np.zeros((H, W), dtype=np.int32)
        pred_count = np.zeros((H, W), dtype=np.int32)

        for k in range(n_K):
            gt_active      = ParamMatcher.is_active(pg     [3 * k], self.active_amp_thr)
            pred_active    = ParamMatcher.is_active(pp     [3 * k], self.active_amp_thr)
            aligned_active = ParamMatcher.is_active(aligned[3 * k], self.active_amp_thr)

            out[f"slot_{k}_active_gt_frac"]   = float(gt_active.mean())
            out[f"slot_{k}_active_pred_frac"] = float(aligned_active.mean())

            gt_count   += gt_active.astype(np.int32)
            pred_count += pred_active.astype(np.int32)

        out["active_frac_gt"]   = float(gt_count.sum(dtype=np.float64)   / (n_K * H * W))
        out["active_frac_pred"] = float(pred_count.sum(dtype=np.float64) / (n_K * H * W))

        out["active_count_gt_mean"]   = float(gt_count.mean(dtype=np.float64))
        out["active_count_pred_mean"] = float(pred_count.mean(dtype=np.float64))

        exact = pred_count == gt_count
        out["count_exact_frac"] = float(exact.mean())
        out["count_under_frac"] = float((pred_count < gt_count).mean())
        out["count_over_frac"]  = float((pred_count > gt_count).mean())

        for k in range(1, n_K + 1):
            gt_mask_k = gt_count == k
            gt_denom  = int(gt_mask_k.sum())
            if gt_denom > 0:
                out[f"count_acc_gt{k}"] = float((exact & gt_mask_k).sum() / gt_denom)

            pred_mask_k = pred_count == k
            pred_denom  = int(pred_mask_k.sum())
            if pred_denom > 0:
                out[f"count_acc_pred{k}"] = float((exact & pred_mask_k).sum() / pred_denom)

        return out

    def _param_population_stats(self) -> Dict[str, float]:
        """Computes per-slot means and standard deviations of amplitude, μ and σ.

        Ground-truth moments are reported over the active pixels of each slot (amplitude
        additionally over inactive and all pixels), and matched-prediction moments over
        all three scopes, so predicted values on inactive slots stay visible.

        Returns:
            Scalar metrics keyed "slot_<k>_<param>_<scope>_<gt|pred>_<mean|std>".
        """

        pg      = self.result.params_gt
        n_K     = self.n_gaussians
        aligned = self._aligned_prediction()
        out: Dict[str, float] = {}

        for k in range(n_K):
            gt_amp   = pg[3 * k].reshape(-1)
            finite   = np.isfinite(gt_amp)
            active   = finite & ParamMatcher.is_active(gt_amp, self.active_amp_thr)
            inactive = finite & ~active
            scopes   = (("active", active), ("inactive", inactive), ("global", finite))

            for j, param in enumerate(("amp", "mu", "sig")):
                ch        = 3 * k + j
                gt_flat   = pg     [ch].reshape(-1)
                pred_flat = aligned[ch].reshape(-1)
                gt_scopes = scopes if param == "amp" else (("active", active),)

                for scope, mask in gt_scopes:
                    out.update(self._population_moments(f"slot_{k}_{param}_{scope}_gt", gt_flat, mask))

                for scope, mask in scopes:
                    out.update(self._population_moments(f"slot_{k}_{param}_{scope}_pred", pred_flat, mask))

        return out

    @staticmethod
    def _population_moments(prefix: str, values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Returns the mean and standard deviation of the finite masked values, NaN when empty.

        Args:
            prefix : Metric-key prefix.
            values : Flat value array.
            mask   : Boolean selection over `values`.

        Returns:
            Metrics keyed "<prefix>_mean" and "<prefix>_std".
        """

        vals = values[mask & np.isfinite(values)]

        return {
            f"{prefix}_mean" : float(vals.mean(dtype=np.float64)) if vals.size else float("nan"),
            f"{prefix}_std"  : float(vals.std(dtype=np.float64))  if vals.size else float("nan"),
        }

    def matched_gaussian_metrics(self, match_tol: float = 5.0) -> Dict[str, float]:
        """Scores the matched predicted and GT Gaussians on parameter error and detection quality.

        Only pairs where both the predicted and the assigned GT slot are active count; a
        pair is a true positive when the μ error stays within `match_tol`. Every quantity
        is also reported per GT-component-count bucket.

        Args:
            match_tol : Maximum absolute μ error, in metres, for a matched pair to count as detected.

        Returns:
            Matched μ, σ and amplitude MAE and RMSE, recall, precision, F1, the pair count,
            the tolerance used, and the per-bucket variants keyed with a "_gt<k>" suffix.
        """

        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians

        sel = self._param_assignment()

        amp_pred = np.stack([pp[3 * k]     for k in range(n_K)], axis=0).reshape(n_K, -1)
        mu_pred  = np.stack([pp[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)
        sig_pred = np.stack([pp[3 * k + 2] for k in range(n_K)], axis=0).reshape(n_K, -1)
        amp_gt   = np.stack([pg[3 * k]     for k in range(n_K)], axis=0).reshape(n_K, -1)
        mu_gt    = np.stack([pg[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)
        sig_gt   = np.stack([pg[3 * k + 2] for k in range(n_K)], axis=0).reshape(n_K, -1)

        act_pred = ParamMatcher.is_active(amp_pred, self.active_amp_thr)
        act_gt   = ParamMatcher.is_active(amp_gt, self.active_amp_thr)
        gt_count = act_gt.sum(axis=0).astype(np.int64)

        n_buckets        = n_K + 1
        rows             = np.arange(gt_count.size)
        sum_mu_ae        = np.zeros(n_buckets, dtype=np.float64)
        sum_mu_se        = np.zeros(n_buckets, dtype=np.float64)
        sum_sig_ae       = np.zeros(n_buckets, dtype=np.float64)
        sum_sig_se       = np.zeros(n_buckets, dtype=np.float64)
        sum_amp_ae       = np.zeros(n_buckets, dtype=np.float64)
        sum_amp_se       = np.zeros(n_buckets, dtype=np.float64)
        n_matched_bucket = np.zeros(n_buckets, dtype=np.float64)
        tp_bucket        = np.zeros(n_buckets, dtype=np.float64)

        for i in range(n_K):
            j       = sel[:, i]
            matched = act_pred[i] & act_gt[j, rows]
            if not matched.any():
                continue

            jm   = j[matched]
            rm   = rows[matched]
            dmu  = np.abs(mu_pred [i, matched] - mu_gt [jm, rm])
            dsig = np.abs(sig_pred[i, matched] - sig_gt[jm, rm])
            damp = np.abs(amp_pred[i, matched] - amp_gt[jm, rm])
            ck   = gt_count[matched]
            tp   = ck[dmu <= match_tol]

            sum_mu_ae        += np.bincount(ck, weights=dmu,         minlength=n_buckets)
            sum_mu_se        += np.bincount(ck, weights=dmu * dmu,   minlength=n_buckets)
            sum_sig_ae       += np.bincount(ck, weights=dsig,        minlength=n_buckets)
            sum_sig_se       += np.bincount(ck, weights=dsig * dsig, minlength=n_buckets)
            sum_amp_ae       += np.bincount(ck, weights=damp,        minlength=n_buckets)
            sum_amp_se       += np.bincount(ck, weights=damp * damp, minlength=n_buckets)
            n_matched_bucket += np.bincount(ck,                      minlength=n_buckets)
            tp_bucket        += np.bincount(tp,                      minlength=n_buckets)

        out: Dict[str, float] = {}

        total_matched = float(n_matched_bucket.sum())
        total_gt      = float(act_gt.sum())
        total_pred    = float(act_pred.sum())
        total_tp      = float(tp_bucket.sum())

        out["matched_mu_mae"]   = float(sum_mu_ae.sum()  / total_matched)          if total_matched > 0 else float("nan")
        out["matched_mu_rmse"]  = float(np.sqrt(sum_mu_se.sum()  / total_matched)) if total_matched > 0 else float("nan")
        out["matched_sig_mae"]  = float(sum_sig_ae.sum() / total_matched)          if total_matched > 0 else float("nan")
        out["matched_sig_rmse"] = float(np.sqrt(sum_sig_se.sum() / total_matched)) if total_matched > 0 else float("nan")
        out["matched_amp_mae"]  = float(sum_amp_ae.sum() / total_matched)          if total_matched > 0 else float("nan")
        out["matched_amp_rmse"] = float(np.sqrt(sum_amp_se.sum() / total_matched)) if total_matched > 0 else float("nan")

        recall    = total_tp / total_gt   if total_gt   > 0 else float("nan")
        precision = total_tp / total_pred if total_pred > 0 else float("nan")
        denom_f1  = recall + precision

        out["matched_recall"]    = float(recall)
        out["matched_precision"] = float(precision)
        out["matched_f1"]        = float(2.0 * recall * precision / denom_f1) if denom_f1 > 0 else float("nan")
        out["matched_n_pairs"]   = total_matched
        out["matched_tol"]       = float(match_tol)

        pixel_count_hist  = np.bincount(gt_count, minlength=n_buckets).astype(np.float64)
        pred_active       = act_pred.sum(axis=0).astype(np.float64)
        pred_active_bucket = np.bincount(gt_count, weights=pred_active, minlength=n_buckets)

        for k in range(1, n_buckets):
            n_k = n_matched_bucket[k]
            if n_k > 0:
                out[f"matched_mu_mae_gt{k}"]  = float(sum_mu_ae[k]  / n_k)
                out[f"matched_sig_mae_gt{k}"] = float(sum_sig_ae[k] / n_k)
                out[f"matched_amp_mae_gt{k}"] = float(sum_amp_ae[k] / n_k)

            gt_active_k = k * pixel_count_hist[k]
            if gt_active_k > 0:
                out[f"matched_recall_gt{k}"] = float(tp_bucket[k] / gt_active_k)

            if pred_active_bucket[k] > 0:
                out[f"matched_precision_gt{k}"] = float(tp_bucket[k] / pred_active_bucket[k])

        return out

    def _slot_organization_stats(self) -> Dict[str, float]:
        """Measures how the model organises its output slots.

        Returns:
            The entropy of per-slot usage, the diagonality of the slot-versus-μ-rank matrix,
            and, when GT parameters exist, the diagonality of the slot-versus-matched-GT matrix.
        """

        pp  = self.result.params_pred
        pg  = self.result.params_gt
        n_K = self.n_gaussians
        out: Dict[str, float] = {}

        usage = SlotOrganization.usage_fractions(pp, n_K)
        out["slot_usage_entropy"] = SlotOrganization.usage_entropy(usage)

        rank_counts = SlotOrganization.mu_rank_matrix(pp, n_K)
        out["slot_mu_rank_diag"] = SlotOrganization.diagonality(rank_counts)

        if pg is not None:
            assign_counts = SlotOrganization.assignment_matrix(pp, pg, n_K, assignment=self._param_assignment())
            out["slot_gt_alignment"] = SlotOrganization.diagonality(assign_counts)

        return out

    def compute(self, *, z_indices  : Optional[np.ndarray] = None, col_indices : Optional[np.ndarray] = None, row_indices : Optional[np.ndarray] = None, param_space : bool = True, slot_organization : bool = True, match_tol : float = 5.0) -> Dict[str, float]:
        """Computes the full metric dictionary for this inference result.

        Args:
            z_indices      : Z indices to score with SSIM, or None to skip that axis.
            col_indices     : Range indices to score with SSIM, or None to skip that axis.
            row_indices        : Row indices to score with SSIM, or None to skip that axis.
            param_space       : Whether Gaussian-parameter metrics should be computed.
            slot_organization : Whether slot-organization metrics should be computed; meaningful only when the parameters came from the model rather than a curve refit.
            match_tol         : μ tolerance in metres for the matched-Gaussian detection metrics.

        Returns:
            Flat mapping of metric name to value, covering cube statistics, curve errors,
            per-pixel distributions, peak displacement in metres, SSIM in denormalised and
            unit-area form, per-z series, roughness and the parameter-space metrics.
        """

        pred = self.result.pred_curves
        gt   = self.result.gt_curves

        metrics: Dict[str, float] = {
            "n_pixels"    : int(self.result.pixel_mse.size),
            "n_z"         : int(pred.shape[0]),
            "x_axis_min"  : float(self.z_axis.min()),
            "x_axis_max"  : float(self.z_axis.max()),
            "x_axis_step" : self.x_step,

            "gt_mean"     : float(gt.mean(dtype=np.float64)),
            "gt_std"      : float(gt.std(dtype=np.float64)),
            "gt_max"      : float(gt.max()),
            "pred_mean"   : float(pred.mean(dtype=np.float64)),
            "pred_std"    : float(pred.std(dtype=np.float64)),
            "pred_max"    : float(pred.max()),
        }

        metrics.update(self._curve_scalar_metrics(pred, gt, suffix="gt"))

        metrics.update(self._expand_pixel_stats((
            ("pixel_mse_gt",        self.result.pixel_mse),
            ("pixel_mae_gt",        self.result.pixel_mae),
            ("pixel_r2_gt",         self.result.pixel_r2),
            ("pixel_cosine_gt",     self.result.pixel_cosine),
            ("pixel_peak_idx_d_gt", self.result.pixel_peak_err_idx.astype(np.float32)),
        )))

        metrics["pixel_peak_err_units_mean_gt"]   = float(self.result.pixel_peak_err_idx.mean())              * self.x_step
        metrics["pixel_peak_err_units_median_gt"] = float(np.median(self.result.pixel_peak_err_idx))          * self.x_step
        metrics["pixel_peak_err_units_p95_gt"]    = float(np.percentile(self.result.pixel_peak_err_idx, 95))  * self.x_step

        metrics.update(self._slice_ssim(pred, gt, z_indices, col_indices, row_indices, prefix="gt"))

        pred_norm = self.result.pred_unit_area()
        gt_norm   = self.result.gt_unit_area()
        metrics.update(self._slice_ssim(pred_norm, gt_norm, z_indices, col_indices, row_indices, prefix="norm"))

        metrics.update(self._expand_z(pred, gt, suffix="gt"))
        metrics.update(self._roughness_stats(pred, gt))

        if param_space:
            metrics.update(self._active_count_stats())
            metrics.update(self._param_population_stats())
            metrics.update(self.matched_gaussian_metrics(match_tol=match_tol))

            if slot_organization:
                metrics.update(self._slot_organization_stats())

        return metrics

    def label_quality(self, full_curves: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        """Measures how well the Gaussian labels reproduce the raw cube they were fitted to.

        Args:
            full_curves : Raw cube of shape (n_z, rows, cols).

        Returns:
            Tuple of the per-pixel label fit R² map of shape (rows, cols) and the scalar
            statistics summarising it, including the split of prediction error by label quality.

        Raises:
            ValueError : If the GT cube and the raw cube disagree in shape, or if the R² map holds no finite value.
        """

        gt   = np.asarray(self.result.gt_curves, dtype=np.float64)
        full = np.asarray(full_curves,           dtype=np.float64)

        if gt.shape != full.shape:
            raise ValueError(f"GT curves {gt.shape} and raw cube {full.shape} disagree in shape; the label-quality map needs both on the same region and z axis")

        r2_map = R2.pixel_map(gt, full, axis=0)
        finite = r2_map[np.isfinite(r2_map)]

        if not finite.size:
            raise ValueError("Label-quality R² map holds no finite value; the GT curves or the raw cube are degenerate on this region")

        stats = {
            "label_r2_mean"            : float(finite.mean()),
            "label_r2_median"          : float(np.median(finite)),
            "label_r2_p05"             : float(np.percentile(finite, 5.0)),
            "label_r2_frac_below_half" : float((finite < 0.5).mean()),
        }

        stats.update(self._label_noise_split(r2_map))

        return r2_map.astype(np.float32), stats

    def _label_noise_split(self, r2_map: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
        """Splits the per-pixel prediction error by whether the label fit was good or poor.

        Args:
            r2_map    : Per-pixel label fit R² of shape (rows, cols).
            threshold : R² value separating poorly fitted labels from well fitted ones.

        Returns:
            Mean prediction error above the threshold, and, when any pixel falls below it,
            the mean error there plus the low-to-high ratio; a status string instead when
            no pixel reaches the threshold.
        """

        err  = np.asarray(self.result.pixel_mse, dtype=np.float64).reshape(-1)
        r2   = np.asarray(r2_map, dtype=np.float64).reshape(-1)
        both = np.isfinite(err) & np.isfinite(r2)

        low  = err[both & (r2 < threshold)]
        high = err[both & (r2 >= threshold)]

        if not high.size:
            return {"label_err_split_status": "skipped: no pixels with label fit R2 >= threshold"}

        out = {"label_err_mean_high_r2": float(high.mean())}

        if low.size:
            out["label_err_mean_low_r2"]    = float(low.mean())
            out["label_err_ratio_low_high"] = float(low.mean() / max(high.mean(), 1e-12))

        return out
