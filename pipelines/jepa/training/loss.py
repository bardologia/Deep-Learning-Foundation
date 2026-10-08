"""Embedding-matching loss for JEPA runs coupled to a profile autoencoder.

The backbone predicts an embedding of the z profile; the target branch
encodes the Gaussian-rendered ground-truth curve with the same autoencoder, and
the loss compares the two normalized embeddings, optionally anchoring them by
decoding the prediction back to curve space.
"""

from __future__ import annotations

import torch

from tools.data.gaussians_torch import GaussianCurve
from tools.loss.curve_loss      import CurveLoss
from tools.monitoring.logger    import Logger


class Loss:
    """Embedding-match criterion between a predicted and an encoded target embedding.

    Attributes:
        autoencoder        : Coupled profile autoencoder providing the encoder, decoder and embedding normalization.
        target_provider    : TargetProvider producing the target embedding branch.
        emb_cfg            : Embedding-loss configuration holding term toggles and weights.
        z_axis             : Z axis in metres used to render ground-truth curves.
        norm_stats         : Normalizer denormalizing the ground-truth Gaussian parameters.
        ppg                : Number of parameters per Gaussian in the target parameter vector.
        profile_normalizer : Normalizer mapping rendered curves into the profile autoencoder's training scale.
        logger             : Logger receiving the active-term table at build and the composition after every stage change.
        loss_generation    : Number of stage swaps applied so far; the checkpoint tracker restarts its best-loss baseline when it changes.
    """

    def __init__(self, autoencoder, target_provider, embedding_cfg, z_axis, norm_stats, params_per_gaussian: int, profile_normalizer, logger: Logger) -> None:
        """Stores the coupled autoencoder, target provider and normalization pieces, and logs the active terms.

        Args:
            autoencoder         : Coupled profile autoencoder.
            target_provider     : TargetProvider for the target embedding branch.
            embedding_cfg       : Embedding-loss configuration of the stage training starts on.
            z_axis              : Z axis in metres, of length equal to the profile length.
            norm_stats          : Output normalizer of the training dataset.
            params_per_gaussian : Number of parameters per Gaussian (amplitude, mean, sigma).
            profile_normalizer  : Curve normalizer matching the profile autoencoder run.
            logger              : Logger receiving the loss tables.
        """
        self.autoencoder        = autoencoder
        self.target_provider    = target_provider
        self.emb_cfg            = embedding_cfg
        self.z_axis             = z_axis
        self.norm_stats         = norm_stats
        self.ppg                = params_per_gaussian
        self.profile_normalizer = profile_normalizer
        self.logger             = logger
        self.loss_generation    = 0

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Regime"          : "embedding match (profile autoencoder coupled)",
            "Target provider" : target_provider.kind,
            "Active terms"    : self.describe_terms(embedding_cfg),
        })

    @staticmethod
    def describe_terms(emb) -> str:
        """Formats the enabled embedding-loss terms and weights of a stage as one banner string.

        Args:
            emb : Embedding-loss configuration of one stage.

        Returns:
            Pipe-separated term description, or "no terms enabled".
        """
        terms = []
        if emb.use_embedding_mse:
            terms.append(f"mse w={emb.weight_embedding_mse:g}")
        if emb.use_embedding_cosine:
            terms.append(f"cosine w={emb.weight_embedding_cosine:g}")
        if emb.use_embedding_smoothl1:
            terms.append(f"smoothl1 w={emb.weight_embedding_smoothl1:g} beta={emb.smoothl1_beta:g}")
        if emb.use_embedding_huber:
            terms.append(f"huber w={emb.weight_embedding_huber:g} delta={emb.embedding_huber_delta:g}")
        if emb.use_embedding_charbonnier:
            terms.append(f"charbonnier w={emb.weight_embedding_charbonnier:g} eps={emb.embedding_charbonnier_eps:g}")
        if emb.use_embedding_variance:
            terms.append(f"variance w={emb.weight_embedding_variance:g} gamma={emb.variance_gamma:g}")
        if emb.use_embedding_covariance:
            terms.append(f"covariance w={emb.weight_embedding_covariance:g}")
        if emb.use_curve_recon:
            terms.append(f"curve_recon({emb.curve_kind}) w={emb.weight_curve_recon:g}")
        if emb.use_curve_sobolev:
            terms.append(f"curve_sobolev w={emb.weight_curve_sobolev:g}")

        return "  |  ".join(terms) if terms else "no terms enabled"

    def set_curriculum(self, embedding_cfg, context: str = "curriculum swap") -> None:
        """Replaces the embedding-loss stage configuration, advances the loss generation and logs the new composition.

        Warmup and main-stage embedding losses live on different scales, so
        the generation step makes the checkpoint tracker restart its baseline
        at the swap instead of keeping a warmup epoch as the best checkpoint.

        Args:
            embedding_cfg : Embedding-loss configuration the criterion switches to.
            context       : Label describing why the stage changed, quoted in the log line.
        """
        self.emb_cfg          = embedding_cfg
        self.loss_generation += 1

        self.logger.subsection(f"Active terms ({context}) : {self.describe_terms(embedding_cfg)}")

    def _embedding_terms(self, z_hat, z_star) -> tuple[torch.Tensor, dict]:
        """Sums the enabled embedding-space terms between prediction and target.

        Args:
            z_hat  : Normalized predicted embedding of shape (batch, embedding_dim, rows, cols).
            z_star : Normalized target embedding of the same shape.

        Returns:
            Tuple of the weighted total as a scalar tensor and a dict mapping each
            enabled term name to its unweighted scalar value.
        """
        cfg        = self.emb_cfg
        components = {}
        total      = torch.zeros((), dtype=z_hat.dtype, device=z_hat.device)

        if cfg.use_embedding_mse:
            val = CurveLoss.mse_diff(z_hat - z_star)
            components["embedding_mse"] = val
            total = total + cfg.weight_embedding_mse * val

        if cfg.use_embedding_cosine:
            val = CurveLoss.cosine(z_hat, z_star, axis=1)
            components["embedding_cosine"] = val
            total = total + cfg.weight_embedding_cosine * val

        if cfg.use_embedding_smoothl1:
            val = CurveLoss.smooth_l1_diff(z_hat - z_star, cfg.smoothl1_beta)
            components["embedding_smoothl1"] = val
            total = total + cfg.weight_embedding_smoothl1 * val

        if cfg.use_embedding_huber:
            val = CurveLoss.huber_diff(z_hat - z_star, cfg.embedding_huber_delta)
            components["embedding_huber"] = val
            total = total + cfg.weight_embedding_huber * val

        if cfg.use_embedding_charbonnier:
            val = CurveLoss.charbonnier_diff(z_hat - z_star, cfg.embedding_charbonnier_eps)
            components["embedding_charbonnier"] = val
            total = total + cfg.weight_embedding_charbonnier * val

        if cfg.use_embedding_variance:
            val = self._variance_hinge(z_hat)
            components["embedding_variance"] = val
            total = total + cfg.weight_embedding_variance * val

        if cfg.use_embedding_covariance:
            val = self._covariance_penalty(z_hat)
            components["embedding_covariance"] = val
            total = total + cfg.weight_embedding_covariance * val

        return total, components

    @staticmethod
    def _flatten_pixels(z: torch.Tensor) -> torch.Tensor:
        """Flattens an embedding map of shape (batch, dim, ...) to (pixels, dim)."""
        return z.movedim(1, -1).reshape(-1, z.shape[1])

    def _variance_hinge(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Returns the hinge holding each embedding dimension's spread above the floor.

        Args:
            z_hat : Normalized predicted embedding of shape (batch, dim, ...).

        Returns:
            Scalar mean of relu(variance_gamma - std) over the dimensions, with
            the standard deviation taken across every pixel of the batch.
        """
        flat = self._flatten_pixels(z_hat)
        std  = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-6)

        return torch.relu(self.emb_cfg.variance_gamma - std).mean()

    def _covariance_penalty(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Returns the mean squared off-diagonal covariance of the predicted embedding.

        Args:
            z_hat : Normalized predicted embedding of shape (batch, dim, ...).

        Returns:
            Scalar sum of squared off-diagonal covariance entries divided by the
            embedding dimension, with the covariance taken across every pixel.
        """
        flat     = self._flatten_pixels(z_hat)
        centered = flat - flat.mean(dim=0, keepdim=True)
        cov      = centered.t() @ centered / max(1, flat.shape[0] - 1)
        dim      = cov.shape[0]

        off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
        return (off_diagonal ** 2).sum() / dim

    @staticmethod
    def _select_masked(t: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
        """Gathers the masked pixels of a channel map into a pixel batch.

        Args:
            t          : Map of shape (batch, channels, rows, cols).
            pixel_mask : Boolean mask of shape (batch, rows, cols), True on the pixels kept.

        Returns:
            Tensor of shape (n_masked, channels, 1, 1) holding the kept pixels.
        """
        selected = t.permute(0, 2, 3, 1)[pixel_mask]
        return selected.unsqueeze(-1).unsqueeze(-1)

    def __call__(self, z_hat, gt_params_norm, pixel_mask=None) -> dict:
        """Evaluates the embedding-match loss for one batch.

        Args:
            z_hat          : Raw predicted embedding from the backbone, of shape (batch, embedding_dim, rows, cols).
            gt_params_norm : Normalized ground-truth Gaussian parameters of shape (batch, params_per_gaussian * n_gaussians, rows, cols).
            pixel_mask     : Optional boolean mask of shape (batch, rows, cols); when given, every term is evaluated on the True pixels only.

        Returns:
            Dict with the scalar "total_loss", the per-term "components" dict, and
            empty "monitor", "occupancy" and "physical" dicts kept for the trainer's
            uniform loss protocol.

        Raises:
            ValueError : If the configured curve reconstruction kind is unknown.
        """
        if pixel_mask is not None:
            z_hat          = self._select_masked(z_hat, pixel_mask)
            gt_params_norm = self._select_masked(gt_params_norm, pixel_mask)

        with torch.no_grad():
            gt_phys    = self.norm_stats.denormalize_output(gt_params_norm.float())
            gt_curve   = GaussianCurve.reconstruct(gt_phys, self.z_axis, self.ppg).to(z_hat.dtype)
            gt_curve_n = self.profile_normalizer.normalize(gt_curve)

        z_hat_n    = self.autoencoder.normalize_embedding(z_hat)
        z_star_raw = self.target_provider.target(self.autoencoder.encoder, gt_curve_n)
        z_star_n   = self.autoencoder.normalize_embedding(z_star_raw)

        total, components = self._embedding_terms(z_hat_n, z_star_n)

        if self.emb_cfg.use_curve_recon or self.emb_cfg.use_curve_sobolev:
            cfg       = self.emb_cfg
            curve_hat = self.autoencoder.decode(z_hat_n)
            diff      = curve_hat - gt_curve_n

            if cfg.use_curve_recon:
                if   cfg.curve_kind == "mse":         c_val = CurveLoss.mse_diff(diff)
                elif cfg.curve_kind == "l1":          c_val = CurveLoss.l1_diff(diff)
                elif cfg.curve_kind == "huber":       c_val = CurveLoss.huber_diff(diff, cfg.huber_delta)
                elif cfg.curve_kind == "charbonnier": c_val = CurveLoss.charbonnier_diff(diff, cfg.charbonnier_eps)
                else:                                 raise ValueError(f"Unknown curve loss kind '{cfg.curve_kind}'. Available: mse, l1, huber, charbonnier")

                components["curve_recon"] = c_val
                total = total + cfg.weight_curve_recon * c_val

            if cfg.use_curve_sobolev:
                s_val = CurveLoss.mse_diff(CurveLoss.second_difference(diff))

                components["curve_sobolev"] = s_val
                total = total + cfg.weight_curve_sobolev * s_val

        return {"total_loss": total, "components": components, "monitor": {}, "occupancy": {}, "physical": {}}
