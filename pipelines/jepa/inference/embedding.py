"""Latent-space diagnostics of a JEPA run against the profile-autoencoder target.

Accumulates, batch by batch, the discrepancy between the embedding the JEPA
predictor infers from the SAR stack and the embedding the profile autoencoder
produces from the ground-truth profile, then separates the curve error into the
decoder's own reconstruction floor and the excess the predictor adds. Every
per-pixel field is stitched back into scene maps on the inference patch grid.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from pipelines.backbone.inference.predictor import CubeStitcher
from tools.monitoring.logger                import Logger


class JepaEmbeddingMaps:
    """Stitched scene maps of the JEPA latent and curve diagnostics.

    Attributes:
        embedding_error  : Mean squared embedding error per pixel, of shape (azimuth, range).
        embedding_cosine : Cosine similarity between predicted and target embedding per pixel, of shape (azimuth, range).
        decode_mse       : Decoder-only curve error per pixel in normalized profile units, of shape (azimuth, range).
        chain_mse        : Full predictor-plus-decoder curve error per pixel in normalized profile units, of shape (azimuth, range).
    """

    def __init__(self, embedding_error: np.ndarray, embedding_cosine: np.ndarray, decode_mse: np.ndarray, chain_mse: np.ndarray) -> None:
        """Stores the four stitched diagnostic maps.

        Args:
            embedding_error  : Per-pixel embedding MSE map.
            embedding_cosine : Per-pixel embedding cosine map.
            decode_mse       : Per-pixel decoder-only curve MSE map.
            chain_mse        : Per-pixel full-chain curve MSE map.
        """
        self.embedding_error  = embedding_error
        self.embedding_cosine = embedding_cosine
        self.decode_mse       = decode_mse
        self.chain_mse        = chain_mse


class JepaEmbeddingEvaluator:
    """Accumulates and reports the latent-space diagnostics of a JEPA inference run.

    Attributes:
        logger      : Logger the diagnostics table is written to.
        adapter     : JEPA inference adapter holding the trained modules and the profile normalizer.
        device      : Torch device the accumulation runs on.
        window_kind : Blending window used when stitching patches into maps.
    """

    def __init__(self, run, logger: Logger, window_kind: str = "uniform") -> None:
        """Prepares the four map stitchers and the running sums of the diagnostics.

        Args:
            run         : Loaded JEPA run holding the model wrapper and the patch grid.
            logger      : Logger the diagnostics are reported through.
            window_kind : Blending window of the patch stitchers.
        """
        self._run        = run
        self.logger      = logger
        self.adapter     = run.model.module
        self.device      = torch.device(run.model.device)
        self.window_kind = window_kind

        self._error_st  = self._create_stitcher()
        self._cosine_st = self._create_stitcher()
        self._decode_st = self._create_stitcher()
        self._chain_st  = self._create_stitcher()

        self._sums = {
            "embedding_sq" : 0.0,
            "embedding_n"  : 0.0,
            "z_star_sq"    : 0.0,
            "cosine_sum"   : 0.0,
            "cosine_n"     : 0.0,
            "decode_sq"    : 0.0,
            "chain_sq"     : 0.0,
            "curve_n"      : 0.0,
        }

        self._per_dim_sum : np.ndarray | None = None
        self._sample_count                    = 0

    def _create_stitcher(self) -> CubeStitcher:
        """Returns a single-channel stitcher over the run's patch grid."""
        return CubeStitcher(grid=self._run.grid, n_channels=1, window_kind=self.window_kind, dtype="float32")

    def _batch_fields(self, z_hat_n: torch.Tensor, curve_n_hat: torch.Tensor, gt_curves: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns the per-element diagnostic fields of one batch.

        The target embedding is the profile autoencoder's own encoding of the
        normalized ground-truth profile, and the decoder-only error is what that
        autoencoder loses when it re-decodes its own encoding.

        Args:
            z_hat_n     : Predicted normalized embeddings of shape (batch, embedding_dim, azimuth, range).
            curve_n_hat : Decoded normalized profiles of shape (batch, height_bins, azimuth, range).
            gt_curves   : Ground-truth profiles of shape (batch, height_bins, azimuth, range), in physical units.

        Returns:
            Squared embedding error, embedding cosine similarity, decoder-only
            squared curve error, full-chain squared curve error and target
            embedding power, each keeping its element axes.
        """
        autoencoder = self.adapter.jepa.profile_autoencoder
        normalizer  = self.adapter.profile_normalizer

        gt_curve_n = normalizer.normalize(gt_curves)
        z_star_n   = autoencoder.encode(gt_curve_n)

        decode_star = autoencoder.decode(z_star_n)

        return {
            "embedding_sq"  : (z_hat_n - z_star_n) ** 2,
            "embedding_cos" : F.cosine_similarity(z_hat_n, z_star_n, dim=1).unsqueeze(1),
            "decode_sq"     : (decode_star - gt_curve_n) ** 2,
            "chain_sq"      : (curve_n_hat - gt_curve_n) ** 2,
            "z_star_sq"     : z_star_n ** 2,
        }

    def accumulate_batch(self, z_hat_n: torch.Tensor, curve_n_hat: torch.Tensor, gt_curves: np.ndarray) -> None:
        """Adds one batch to the running sums, the per-dimension sums and the stitched maps.

        Args:
            z_hat_n     : Predicted normalized embeddings of shape (batch, embedding_dim, azimuth, range).
            curve_n_hat : Decoded normalized profiles of shape (batch, height_bins, azimuth, range).
            gt_curves   : Ground-truth profiles of shape (batch, height_bins, azimuth, range), in physical units.
        """
        with torch.no_grad():
            curves = torch.from_numpy(np.asarray(gt_curves, dtype=np.float32)).to(self.device)
            fields = self._batch_fields(z_hat_n.to(self.device), curve_n_hat.to(self.device), curves)

            embedding_map = fields["embedding_sq"].mean(dim=1, keepdim=True).cpu().numpy().astype(np.float32)
            decode_map    = fields["decode_sq"].mean(dim=1, keepdim=True).cpu().numpy().astype(np.float32)
            chain_map     = fields["chain_sq"].mean(dim=1, keepdim=True).cpu().numpy().astype(np.float32)
            cosine_map    = fields["embedding_cos"].cpu().numpy().astype(np.float32)

            for b in range(embedding_map.shape[0]):
                self._error_st.add_patch( self._sample_count + b, embedding_map[b])
                self._cosine_st.add_patch(self._sample_count + b, cosine_map[b])
                self._decode_st.add_patch(self._sample_count + b, decode_map[b])
                self._chain_st.add_patch( self._sample_count + b, chain_map[b])

            dim_sum           = fields["embedding_sq"].sum(dim=(0, 2, 3)).cpu().numpy().astype(np.float64)
            self._per_dim_sum = dim_sum if self._per_dim_sum is None else self._per_dim_sum + dim_sum

            self._sums["embedding_sq"] += float(fields["embedding_sq"].sum())
            self._sums["embedding_n"]  += float(fields["embedding_sq"].numel())
            self._sums["z_star_sq"]    += float(fields["z_star_sq"].sum())
            self._sums["cosine_sum"]   += float(fields["embedding_cos"].sum())
            self._sums["cosine_n"]     += float(fields["embedding_cos"].numel())
            self._sums["decode_sq"]    += float(fields["decode_sq"].sum())
            self._sums["chain_sq"]     += float(fields["chain_sq"].sum())
            self._sums["curve_n"]      += float(fields["chain_sq"].numel())

            self._sample_count += embedding_map.shape[0]

    def _finalize(self) -> tuple[Dict[str, float], np.ndarray, JepaEmbeddingMaps]:
        """Closes the accumulation into sums, per-dimension errors and stitched maps.

        Returns:
            The running sums, the mean squared error of each embedding dimension
            of shape (embedding_dim,), and the stitched diagnostic maps.

        Raises:
            ValueError : If no batch was accumulated.
        """
        if self._sums["embedding_n"] == 0.0:
            raise ValueError("JEPA embedding evaluation saw no samples; the inference loader is empty.")

        maps = JepaEmbeddingMaps(
            embedding_error  = self._error_st.finalize_cube()[0],
            embedding_cosine = self._cosine_st.finalize_cube()[0],
            decode_mse       = self._decode_st.finalize_cube()[0],
            chain_mse        = self._chain_st.finalize_cube()[0],
        )

        samples_per_dim = self._sums["embedding_n"] / self._per_dim_sum.size

        return self._sums, self._per_dim_sum / samples_per_dim, maps

    def _scalars(self, sums: Dict[str, float], per_dim_mse: np.ndarray, maps: JepaEmbeddingMaps) -> Dict[str, float]:
        """Reduces the accumulated sums and maps to the scalar error-transfer metrics.

        The chain error is split into the decoder floor and the predictor excess,
        and the amplification states how much normalized curve MSE each unit of
        embedding MSE costs.

        Args:
            sums        : Running sums produced by the accumulation.
            per_dim_mse : Mean squared error of each embedding dimension, of shape (embedding_dim,).
            maps        : Stitched diagnostic maps, used for the tail percentiles.

        Returns:
            The named scalar diagnostics, curve errors in normalized profile units.
        """
        embedding_mse = sums["embedding_sq"] / sums["embedding_n"]
        decode_mse    = sums["decode_sq"]    / sums["curve_n"]
        chain_mse     = sums["chain_sq"]     / sums["curve_n"]
        target_power  = sums["z_star_sq"]    / sums["embedding_n"]

        per_dim_rmse = np.sqrt(per_dim_mse)

        return {
            "jepa_embedding_mse"          : embedding_mse,
            "jepa_embedding_rmse"         : float(np.sqrt(embedding_mse)),
            "jepa_embedding_cosine"       : sums["cosine_sum"] / sums["cosine_n"],
            "jepa_embedding_nmse"         : embedding_mse / target_power if target_power > 0.0 else float("nan"),
            "jepa_embedding_dim_rmse_min" : float(per_dim_rmse.min()),
            "jepa_embedding_dim_rmse_max" : float(per_dim_rmse.max()),
            "jepa_embedding_dim_rmse_std" : float(per_dim_rmse.std()),
            "jepa_decode_mse_norm"        : decode_mse,
            "jepa_chain_mse_norm"         : chain_mse,
            "jepa_predictor_excess_mse"   : chain_mse - decode_mse,
            "jepa_decoder_floor_frac"     : decode_mse / chain_mse if chain_mse > 0.0 else float("nan"),
            "jepa_decode_amplification"   : (chain_mse - decode_mse) / embedding_mse if embedding_mse > 0.0 else float("nan"),
            "jepa_embedding_err_p95"      : float(np.percentile(maps.embedding_error[np.isfinite(maps.embedding_error)], 95.0)),
            "jepa_embedding_cosine_p5"    : float(np.percentile(maps.embedding_cosine[np.isfinite(maps.embedding_cosine)], 5.0)),
        }

    def _correlate_with_error(self, maps: JepaEmbeddingMaps, pixel_mse: np.ndarray | None) -> Dict[str, float]:
        """Correlates the embedding-error map with the per-pixel curve error map.

        Args:
            maps      : Stitched diagnostic maps.
            pixel_mse : Per-pixel curve MSE map of shape (azimuth, range), or None when the pipeline did not produce one.

        Returns:
            The Pearson correlation under `jepa_embedding_error_corr`, empty when
            no curve-error map was given, NaN when fewer than three pixels are
            finite in both maps.

        Raises:
            ValueError : If the two maps hold different pixel counts, meaning they were stitched on different grids.
        """
        if pixel_mse is None:
            return {}

        embedding = maps.embedding_error.reshape(-1).astype(np.float64)
        curve     = np.asarray(pixel_mse, dtype=np.float64).reshape(-1)

        if embedding.size != curve.size:
            raise ValueError(f"Embedding-error map holds {embedding.size} pixels but the curve-error map holds {curve.size}; they were stitched on different grids.")

        both = np.isfinite(embedding) & np.isfinite(curve)
        if both.sum() <= 2:
            return {"jepa_embedding_error_corr": float("nan")}

        return {"jepa_embedding_error_corr": float(np.corrcoef(embedding[both], curve[both])[0, 1])}

    def _report(self, metrics: Dict[str, float]) -> None:
        """Logs the embedding, decoder-floor and amplification diagnostics as a table.

        Args:
            metrics : Scalar diagnostics produced by the reduction.
        """
        self.logger.section("[Inference: JEPA Embedding Diagnostics]")
        self.logger.kv_table({
            "Embedding MSE"           : f"{metrics['jepa_embedding_mse']:.6g}",
            "Embedding RMSE"          : f"{metrics['jepa_embedding_rmse']:.6g} (per-component spread; calibrates the profile-AE latent_noise_std)",
            "Embedding NMSE"          : f"{metrics['jepa_embedding_nmse']:.6g} (error power over target power)",
            "Embedding cosine"        : f"{metrics['jepa_embedding_cosine']:.4f}",
            "Per-dim RMSE spread"     : f"min {metrics['jepa_embedding_dim_rmse_min']:.4g}, max {metrics['jepa_embedding_dim_rmse_max']:.4g}",
            "Decoder-only MSE (norm)" : f"{metrics['jepa_decode_mse_norm']:.6g}",
            "Full-chain MSE (norm)"   : f"{metrics['jepa_chain_mse_norm']:.6g}",
            "Predictor excess MSE"    : f"{metrics['jepa_predictor_excess_mse']:.6g} ({metrics['jepa_decoder_floor_frac'] * 100.0:.1f}% of the chain error is the decoder floor)",
            "Decode amplification"    : f"{metrics['jepa_decode_amplification']:.4g} (curve MSE added per unit of embedding MSE)",
        })

        if "jepa_embedding_error_corr" in metrics:
            self.logger.subsection(f"Embedding error correlates with per-pixel curve MSE at r = {metrics['jepa_embedding_error_corr']:.3f}")

    def run(self, pixel_mse: np.ndarray | None = None) -> tuple[Dict[str, float], JepaEmbeddingMaps]:
        """Closes the accumulation, reduces it to scalars, reports it and returns the results.

        Args:
            pixel_mse : Per-pixel curve MSE map of shape (azimuth, range) the embedding error is correlated against, or None to skip the correlation.

        Returns:
            The scalar diagnostics and the stitched diagnostic maps.
        """
        sums, per_dim_mse, maps = self._finalize()

        metrics = self._scalars(sums, per_dim_mse, maps)
        metrics.update(self._correlate_with_error(maps, pixel_mse))

        self._report(metrics)

        return metrics, maps
