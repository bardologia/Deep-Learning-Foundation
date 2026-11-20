
from __future__ import annotations

import os
from collections        import deque
from concurrent.futures import ProcessPoolExecutor
from typing             import List, Optional, Tuple

import numpy as np
import torch

from pipelines.backbone.dataset.spatial              import GridInfo
from pipelines.backbone.inference.loader             import Run
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics            import Metrics, Result
from tools.data.gaussians                            import GaussianActivity, GaussianReconstructor
from tools.metrics.scoring                           import R2
from tools.monitoring.logger                         import Logger
from tools.runtime.cuda_mask                         import InheritedCudaMask


class CubeStitcher:

    def __init__(
        self,
        grid        : GridInfo,
        n_channels  : int,
        window_kind : str           = "hann",
        dtype       : str           = "float32",
        memmap_path : Optional[str] = None,
    ) -> None:
        self.grid       = grid
        self.n_channels = int(n_channels)
        self.dtype      = np.dtype(dtype)
        self.window     = CubeStitcher.make_patch_window(grid.patch_size, kind=window_kind)

        H_pad, W_pad = grid.padded_size
        shape_pad    = (self.n_channels, H_pad, W_pad)

        if memmap_path is not None:
            self._accum = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=self.dtype, shape=shape_pad)
        else:
            self._accum = np.zeros(shape_pad, dtype=self.dtype)

        self._weight = np.zeros((H_pad, W_pad), dtype=np.float32)

    @staticmethod
    def make_patch_window(patch_size: Tuple[int, int], kind: str = "hann") -> np.ndarray:
        ph, pw = patch_size
        if kind == "uniform":
            return np.ones((ph, pw), dtype=np.float32)

        if kind == "hann":
            wv = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(ph) + 0.5) / ph)
            wh = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(pw) + 0.5) / pw)

        else:
            wv, wh = {"triangular": lambda: (1.0 - np.abs((np.arange(ph) + 0.5) / ph * 2.0 - 1.0), 1.0 - np.abs((np.arange(pw) + 0.5) / pw * 2.0 - 1.0))}[kind]()

        wv = np.clip(wv, 1e-3, None).astype(np.float32)
        wh = np.clip(wh, 1e-3, None).astype(np.float32)

        return np.outer(wv, wh)

    @property
    def number_of_patches(self) -> int:
        return self.grid.number_of_patches

    def add_patch(self, idx: int, patch: np.ndarray) -> None:
        ph, pw = self.grid.patch_size
        sv, sh = self.grid.stride
        iv, ih = divmod(idx, self.grid.n_h)
        v0 = iv * sv
        h0 = ih * sh
        w  = self.window

        self._accum[:, v0:v0 + ph, h0:h0 + pw] += (patch * w[None, :, :]).astype(self.dtype, copy=False)
        self._weight[v0:v0 + ph, h0:h0 + pw]   += w

    def finalize_cube(self) -> np.ndarray:
        H, W         = self.grid.spatial_size
        pad_t, pad_l = self.grid.pad_top, self.grid.pad_left

        weight  = self._weight[pad_t:pad_t + H, pad_l:pad_l + W]

        cube = self._accum[:, pad_t:pad_t + H, pad_l:pad_l + W] / weight[None, :, :]

        return np.ascontiguousarray(cube.astype(self.dtype, copy=False))


class SelectStitcher:

    def __init__(
        self,
        grid        : GridInfo,
        n_channels  : int,
        dtype       : str           = "float32",
        memmap_path : Optional[str] = None,
    ) -> None:
        self.grid       = grid
        self.n_channels = int(n_channels)
        self.dtype      = np.dtype(dtype)
        self.centrality = CubeStitcher.make_patch_window(grid.patch_size, kind="hann")

        H_pad, W_pad = grid.padded_size
        shape_pad    = (self.n_channels, H_pad, W_pad)

        if memmap_path is not None:
            self._values = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=self.dtype, shape=shape_pad)
        else:
            self._values = np.zeros(shape_pad, dtype=self.dtype)

        self._best = np.zeros((H_pad, W_pad), dtype=np.float32)

    def add_patch(self, idx: int, patch: np.ndarray) -> None:
        ph, pw = self.grid.patch_size
        sv, sh = self.grid.stride
        iv, ih = divmod(idx, self.grid.n_h)
        v0 = iv * sv
        h0 = ih * sh

        best   = self._best[v0:v0 + ph, h0:h0 + pw]
        values = self._values[:, v0:v0 + ph, h0:h0 + pw]
        take   = self.centrality > best

        values[:, take] = patch[:, take].astype(self.dtype, copy=False)
        best[take]      = self.centrality[take]

    def finalize_cube(self) -> np.ndarray:
        H, W         = self.grid.spatial_size
        pad_t, pad_l = self.grid.pad_top, self.grid.pad_left

        cube = self._values[:, pad_t:pad_t + H, pad_l:pad_l + W]

        return np.ascontiguousarray(cube.astype(self.dtype, copy=False))


class Predictor:

    RESULT_BUDGET_BYTES = 2 * 1024 ** 3

    def __init__(
        self,
        run         : Run,
        logger      : Logger,
        *,
        window_kind      : str,
        cube_dtype       : str,
        save_cubes       : bool,
        meta             : InferenceMetadata,
        cpu_workers      : int | None = None,
    ) -> None:
        self.run              = run
        self.logger           = logger
        self.window_kind      = window_kind
        self.cube_dtype       = cube_dtype
        self.save_cubes       = save_cubes
        self.cube_dir         = meta.cube_dir
        self.cpu_workers      = cpu_workers if cpu_workers is not None else min(8, os.cpu_count() or 1)

    def _device_rows(self) -> dict:
        device = torch.device(self.run.model.device)
        rows   = {"Backend": f"PyTorch / {device}"}

        if device.type == "cuda":
            rows["GPU"]                  = torch.cuda.get_device_name(device)
            rows["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
            rows["Inherited CUDA mask"]  = os.environ.get(InheritedCudaMask.STASH_VAR, "unset")

        return rows

    def _forward_pass(self) -> Tuple[List[List[int]], List[np.ndarray], List[np.ndarray]]:
        run = self.run
        n_K = run.n_gaussians

        all_indices     : List[List[int]]  = []
        all_pred_params : List[np.ndarray] = []
        all_gt_params   : List[np.ndarray] = []
        sample_count = 0

        with self.logger.track(transient=True) as prog:
            task = prog.add_task("[section]GPU Forward Pass[/section]", total=len(run.loader))
            for batch in run.loader:
                images, gt_params_b = batch[0], batch[1]

                pred_params = run.model(images)

                gt_params_ready = gt_params_b[:, :n_K * 3]
                gt_params_ready = run.dataset.normalizer.denormalize_output(gt_params_ready)

                B = images.shape[0]
                all_indices.append(list(range(sample_count, sample_count + B)))
                all_pred_params.append(pred_params)
                all_gt_params.append(gt_params_ready.cpu().numpy().astype(np.float32))
                sample_count += B
                prog.advance(task)

        return all_indices, all_pred_params, all_gt_params

    @staticmethod
    def _sort_by_mu(gauss: np.ndarray, activity_threshold: float) -> np.ndarray:
        sort_key   = np.where(GaussianActivity.is_active(gauss[:, :, 0], activity_threshold), gauss[:, :, 1], np.inf)
        sort_idx   = np.argsort(sort_key, axis=1)
        sort_idx_e = sort_idx[:, :, None, :, :].repeat(3, axis=2)

        return np.take_along_axis(gauss, sort_idx_e, axis=1)

    @staticmethod
    def _cpu_worker(args: tuple) -> tuple:
        pred_params_chunk, gt_params_chunk, z_axis, n_gaussians, activity_threshold = args

        x          = z_axis.reshape(1, 1, -1, 1, 1).astype(np.float32)
        B, _, H, W = pred_params_chunk.shape

        n_K        = n_gaussians
        pred_gauss = pred_params_chunk[:, :n_K * 3].reshape(B, n_K, 3, H, W).astype(np.float32)
        gt_gauss   = gt_params_chunk[:,   :n_K * 3].reshape(B, n_K, 3, H, W).astype(np.float32)

        gt_gauss_matched = Predictor._sort_by_mu(gt_gauss, activity_threshold)

        pred_gauss_flat = pred_gauss.reshape(      B, n_K * 3, H, W)
        gt_gauss_flat   = gt_gauss_matched.reshape(B, n_K * 3, H, W)

        pred_curves = GaussianReconstructor.reconstruct_batch(pred_gauss,       x)
        gt_curves   = GaussianReconstructor.reconstruct_batch(gt_gauss_matched, x)

        return (
            pred_curves,
            gt_curves,
            pred_gauss_flat,
            gt_gauss_flat,
        )

    def _stitch_results(self, all_indices : List[List[int]], all_pred_params : List[np.ndarray], all_gt_params : List[np.ndarray]) -> Result:
        run    = self.run
        n_K    = run.n_gaussians
        out_ch = run.out_channels
        n_z    = run.x_axis_length

        pred_curve_stitcher = self._create_stitcher(n_z,  "pred_curves")
        gt_curve_stitcher   = self._create_stitcher(n_z,  "gt_curves")
        param_pred_stitcher = self._create_param_stitcher(out_ch,  "params_pred")
        gt_param_stitcher   = self._create_param_stitcher(n_K * 3, "params_gt")

        tasks     = [(pred, gt, run.z_axis, n_K, run.activity_threshold) for pred, gt in zip(all_pred_params, all_gt_params)]
        in_flight = self._result_window(all_pred_params)

        with self.logger.track(transient=True) as prog:
            task_id = prog.add_task("[section]CPU Metrics[/section]", total=len(tasks))
            with ProcessPoolExecutor(max_workers=self.cpu_workers) as pool:
                pending : deque = deque()
                submitted       = 0

                for batch_indices in all_indices:
                    while submitted < len(tasks) and len(pending) < in_flight:
                        pending.append(pool.submit(Predictor._cpu_worker, tasks[submitted]))
                        submitted += 1

                    pred_curves, gt_curves, pred_params, gt_params = pending.popleft().result()

                    for b, idx in enumerate(batch_indices):
                        pred_curve_stitcher.add_patch(idx, pred_curves[b].astype(self.cube_dtype))
                        gt_curve_stitcher.add_patch(  idx, gt_curves  [b].astype(self.cube_dtype))
                        param_pred_stitcher.add_patch(idx, pred_params[b].astype(self.cube_dtype))
                        gt_param_stitcher.add_patch(  idx, gt_params  [b].astype(self.cube_dtype))

                    prog.advance(task_id)

        return self._finalize_results(
            pred_curve_stitcher, gt_curve_stitcher,
            param_pred_stitcher, gt_param_stitcher,
        )

    def _create_stitcher(self, n_channels: int, name: str) -> CubeStitcher:
        memmap_path = str(self.cube_dir / f"_tmp_{name}.npy") if self.save_cubes else None
        return CubeStitcher(
            grid        = self.run.grid,
            n_channels  = n_channels,
            window_kind = self.window_kind,
            dtype       = self.cube_dtype,
            memmap_path = memmap_path,
        )

    def _create_param_stitcher(self, n_channels: int, name: str) -> SelectStitcher:
        memmap_path = str(self.cube_dir / f"_tmp_{name}.npy") if self.save_cubes else None
        return SelectStitcher(
            grid        = self.run.grid,
            n_channels  = n_channels,
            dtype       = self.cube_dtype,
            memmap_path = memmap_path,
        )

    def _result_window(self, all_pred_params: List[np.ndarray]) -> int:
        run    = self.run
        ph, pw = run.grid.patch_size
        n_elev = run.x_axis_length
        n_K    = run.n_gaussians

        batch          = max(int(params.shape[0]) for params in all_pred_params)
        channels       = 2 * n_elev + 2 * (3 * n_K)
        bytes_per_task = batch * channels * ph * pw * np.dtype(np.float32).itemsize

        window = min(2 * self.cpu_workers, max(1, self.RESULT_BUDGET_BYTES // bytes_per_task))

        self.logger.info(f"Stitching keeps up to {window} finished batches in flight ({bytes_per_task / 1024 ** 3:.2f} GB each, {self.RESULT_BUDGET_BYTES / 1024 ** 3:.2f} GB budget)")

        return int(window)

    def _finalize_results(
        self,
        pred_curve_stitcher : CubeStitcher,
        gt_curve_stitcher   : CubeStitcher,
        param_pred_stitcher : SelectStitcher,
        gt_param_stitcher   : SelectStitcher,
    ) -> Result:
        pred_curves_cube = pred_curve_stitcher.finalize_cube()
        gt_curves_cube   = gt_curve_stitcher.finalize_cube()
        params_pred_cube = param_pred_stitcher.finalize_cube()
        params_gt_cube   = gt_param_stitcher.finalize_cube()

        n_K = self.run.n_gaussians
        for k in range(n_K):
            a_gt    = params_gt_cube[3 * k]
            mask_gt = ~GaussianActivity.is_active(a_gt, self.run.activity_threshold)
            params_gt_cube[3 * k + 1][mask_gt] = np.nan
            params_gt_cube[3 * k + 2][mask_gt] = np.nan

        pixel_maps     = Metrics.curve_pixel_metrics(pred_curves_cube, gt_curves_cube)
        pixel_mse      = pixel_maps["mse"]
        pixel_mae      = pixel_maps["mae"]
        pixel_r2       = pixel_maps["r2"]
        pixel_cos      = pixel_maps["cos"]
        pixel_peak_idx = pixel_maps["peak"].astype(np.int32)

        self.cube_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cube_dir / "pixel_mse.npy",  pixel_mse)
        np.save(self.cube_dir / "pixel_mae.npy",  pixel_mae)
        np.save(self.cube_dir / "pixel_r2.npy",   pixel_r2)
        np.save(self.cube_dir / "pixel_cos.npy",  pixel_cos)
        np.save(self.cube_dir / "pixel_peak.npy", pixel_peak_idx)

        if self.save_cubes:
            np.save(self.cube_dir / "pred_curves.npy", pred_curves_cube)
            np.save(self.cube_dir / "gt_curves.npy",   gt_curves_cube)
            np.save(self.cube_dir / "params_pred.npy", params_pred_cube)
            np.save(self.cube_dir / "params_gt.npy",   params_gt_cube)
            for tmp in self.cube_dir.glob("_tmp_*.npy"):
                tmp.unlink(missing_ok=True)

        undefined_r2 = R2.undefined_count(pixel_r2)

        self.logger.section("[Inference: Results]")
        self.logger.kv_table({
            "Curves cube    (denorm)": pred_curves_cube.shape,
            "Params cube    (denorm)": params_pred_cube.shape,
            "GT params cube (denorm)": params_gt_cube.shape,
            "Mean pixel MSE (denorm)": f"{pixel_mse.mean():.4g}  (pred vs gt)",
            "Mean pixel R²  (denorm)": f"{np.nanmean(pixel_r2):.4g}  (pred vs gt, over {pixel_r2.size - undefined_r2} of {pixel_r2.size} pixels)",
        })

        if undefined_r2:
            self.logger.warning(f"{undefined_r2} of {pixel_r2.size} pixels have a constant GT profile; their R² is undefined and excluded from the pixel_r2_gt statistics (counted as pixel_r2_gt_undefined_count)")

        return Result(
            pred_curves        = pred_curves_cube,
            gt_curves          = gt_curves_cube,
            params_pred        = params_pred_cube,
            params_gt          = params_gt_cube,
            pixel_mse          = pixel_mse,
            pixel_mae          = pixel_mae,
            pixel_r2           = pixel_r2,
            pixel_cosine       = pixel_cos,
            pixel_peak_err_idx = pixel_peak_idx,
            cube_directory     = self.cube_dir,
            row_offset         = self.run.split_region.row_start,
            col_offset         = self.run.split_region.col_start,
        )

    def run_inference(self) -> Result:
        self.logger.section("[Inference: Predict]")
        self.logger.kv_table({
            **self._device_rows(),
            "Cube dir"    : self.cube_dir,
            "Window"      : self.window_kind,
            "Cube dtype"  : self.cube_dtype,
            "CPU workers" : self.cpu_workers,
        })

        all_indices, all_pred_params, all_gt_params = self._forward_pass()
        results = self._stitch_results(all_indices, all_pred_params, all_gt_params)

        return results
