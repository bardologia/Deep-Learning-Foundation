"""Curve-space predictor for JEPA runs, streaming patches into stitched cubes.

The JEPA backbone predicts a profile-autoencoder embedding rather than Gaussian
parameters, so inference decodes z curves directly and keeps only the
ground-truth Gaussian parameters alongside them.
"""

from __future__ import annotations

import numpy as np
import torch

from pipelines.backbone.inference.metrics   import Metrics, Result
from pipelines.backbone.inference.predictor import Predictor
from tools.data.gaussians                   import GaussianActivity, GaussianReconstructor


class JepaCurvePredictor(Predictor):
    """Predictor that stitches decoded z curves instead of Gaussian slots.

    Attributes:
        embedding_evaluator : Optional accumulator fed the predicted embedding and decoded curve of each batch to score embedding error transfer; when None the model is called through its plain forward path.
    """

    def __init__(self, run, logger, *, embedding_evaluator=None, **kwargs) -> None:
        """Stores the embedding evaluator on top of the base predictor setup.

        Args:
            run                 : Loaded inference run bundle (model, loader, dataset, axes, split region).
            logger              : Logger used for progress tracking and result tables.
            embedding_evaluator : Optional evaluator accumulating per-batch embedding and decoded-curve statistics.
            **kwargs            : Forwarded to the base Predictor (cube dtype, window kind, save flags).
        """
        super().__init__(run, logger, **kwargs)
        self.embedding_evaluator = embedding_evaluator

    def _create_curve_stitchers(self):
        """Returns the predicted-curve, ground-truth-curve and ground-truth-parameter stitchers."""
        n_elev    = self.run.x_axis_length
        pred_st   = self._create_stitcher(n_elev, "pred_curves")
        gt_st     = self._create_stitcher(n_elev, "gt_curves")
        gt_par_st = self._create_param_stitcher(self.run.n_gaussians * 3, "params_gt")

        return pred_st, gt_st, gt_par_st

    def _forward_with_embedding(self, images):
        """Runs the wrapped module's embedding-aware forward pass on one batch.

        Args:
            images : Input patch batch of shape (batch, channels, rows, cols).

        Returns:
            Tuple of decoded curves as a NumPy array of shape
            (batch, z, rows, cols), the normalized predicted embedding
            tensor and the decoded normalized curve tensor.
        """
        wrapper = self.run.model

        with torch.no_grad():
            curves, z_hat_n, curve_n_hat = wrapper.module.forward_with_embedding(torch.as_tensor(images).to(dtype=torch.float32, device=wrapper.device))

        return curves.cpu().numpy(), z_hat_n, curve_n_hat

    def _stream_patches(self, pred_st, gt_st, gt_par_st):
        """Runs the loader once, adding predicted and ground-truth patches to the stitchers.

        Ground-truth Gaussian parameters are denormalized, sorted by mean and
        rendered to curves on the z axis so predictions and targets live
        in the same curve space.

        Args:
            pred_st   : Stitcher collecting predicted curve patches.
            gt_st     : Stitcher collecting ground-truth curve patches.
            gt_par_st : Stitcher collecting flattened ground-truth Gaussian parameters.
        """
        run = self.run
        n_K = run.n_gaussians
        x   = np.asarray(run.z_axis, dtype=np.float32).reshape(1, 1, -1, 1, 1)

        sample_count = 0

        with self.logger.track(transient=True) as prog:
            task = prog.add_task("[section]GPU Forward Pass[/section]", total=len(run.loader))
            for batch in run.loader:
                images, gt_params_b = batch[0], batch[1]

                if self.embedding_evaluator is None:
                    pred_curves = run.model(images)
                else:
                    pred_curves, z_hat_n, curve_n_hat = self._forward_with_embedding(images)

                gt_params = gt_params_b[:, :n_K * 3]
                gt_params = run.dataset.normalizer.denormalize_output(gt_params).cpu().numpy().astype(np.float32)

                B         = pred_curves.shape[0]
                H, W      = gt_params.shape[-2:]
                gt_gauss  = Predictor._sort_by_mu(gt_params.reshape(B, n_K, 3, H, W), run.activity_threshold)
                gt_curves = GaussianReconstructor.reconstruct_batch(Predictor._render_masked(gt_gauss, self.render_amp_floor), x)
                gt_flat   = gt_gauss.reshape(B, n_K * 3, H, W)

                for b in range(B):
                    idx = sample_count + b
                    pred_st.add_patch(  idx, pred_curves[b].astype(self.cube_dtype))
                    gt_st.add_patch(    idx, gt_curves  [b].astype(self.cube_dtype))
                    gt_par_st.add_patch(idx, gt_flat    [b].astype(self.cube_dtype))

                if self.embedding_evaluator is not None:
                    self.embedding_evaluator.accumulate_batch(z_hat_n, curve_n_hat, gt_curves)

                sample_count += B
                prog.advance(task)

    def _mask_inactive_gt(self, params_gt: np.ndarray) -> np.ndarray:
        """Sets mean and sigma to NaN wherever a ground-truth Gaussian slot is inactive.

        Args:
            params_gt : Ground-truth parameter cube of shape (3 * n_gaussians, rows, cols), ordered amplitude, mean, sigma per slot, in physical units (metres for mean and sigma).

        Returns:
            The same array with inactive slots' mean and sigma channels masked.
        """
        for k in range(self.run.n_gaussians):
            inactive = ~GaussianActivity.is_active(params_gt[3 * k], self.run.activity_threshold)
            params_gt[3 * k + 1][inactive] = np.nan
            params_gt[3 * k + 2][inactive] = np.nan

        return params_gt

    def _finalize(self, pred_cube: np.ndarray, gt_cube: np.ndarray, params_gt: np.ndarray) -> Result:
        """Computes per-pixel curve metrics, optionally saves cubes, and packs the result.

        Args:
            pred_cube : Predicted curves of shape (z, rows, cols).
            gt_cube   : Ground-truth curves of shape (z, rows, cols).
            params_gt : Ground-truth Gaussian parameters of shape (3 * n_gaussians, rows, cols), inactive slots already masked.

        Returns:
            Result carrying both cubes, the parameter cube, the per-pixel MSE, MAE,
            R², cosine and peak-index maps of shape (rows, cols), the cube
            directory and the row/col offsets of the inference split region.
        """
        pixel_maps     = Metrics.curve_pixel_metrics(pred_cube, gt_cube)
        pixel_mse      = pixel_maps["mse"]
        pixel_mae      = pixel_maps["mae"]
        pixel_r2       = pixel_maps["r2"]
        pixel_cos      = pixel_maps["cos"]
        pixel_peak_idx = pixel_maps["peak"].astype(np.int32)

        if self.save_cubes:
            np.save(self.cube_dir / "pred_curves.npy", pred_cube)
            np.save(self.cube_dir / "gt_curves.npy",   gt_cube)
            np.save(self.cube_dir / "params_gt.npy",   params_gt)
            np.save(self.cube_dir / "pixel_mse.npy",   pixel_mse)
            np.save(self.cube_dir / "pixel_mae.npy",   pixel_mae)
            np.save(self.cube_dir / "pixel_r2.npy",    pixel_r2)
            np.save(self.cube_dir / "pixel_cos.npy",   pixel_cos)
            np.save(self.cube_dir / "pixel_peak.npy",  pixel_peak_idx)
            for tmp in self.cube_dir.glob("_tmp_*.npy"):
                tmp.unlink(missing_ok=True)

        self.logger.section("[Inference: Results]")
        self.logger.kv_table({
            "Curves cube    (denorm)": pred_cube.shape,
            "GT params cube (denorm)": params_gt.shape,
            "Mean pixel MSE (denorm)": f"{pixel_mse.mean():.4g}  (pred vs gt)",
            "Mean pixel R²  (denorm)": f"{pixel_r2.mean():.4g}  (pred vs gt)",
        })

        return Result(
            pred_curves        = pred_cube,
            gt_curves          = gt_cube,
            params_gt          = params_gt,
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
        """Runs the full curve-space inference pass and returns the finalized result.

        Returns:
            Result with stitched predicted and ground-truth curve cubes, the masked
            ground-truth parameter cube and the per-pixel metric maps.
        """
        backend = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.section("[Inference: Predict]")
        self.logger.kv_table({
            "Backend"    : f"PyTorch / {backend}",
            "Cube dir"   : self.cube_dir,
            "Window"     : self.window_kind,
            "Cube dtype" : self.cube_dtype,
        })

        pred_st, gt_st, gt_par_st = self._create_curve_stitchers()
        self._stream_patches(pred_st, gt_st, gt_par_st)

        pred_cube = pred_st.finalize_cube()
        gt_cube   = gt_st.finalize_cube()
        params_gt = gt_par_st.finalize_cube()

        return self._finalize(pred_cube, gt_cube, self._mask_inactive_gt(params_gt))
