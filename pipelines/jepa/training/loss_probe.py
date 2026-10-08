"""Pre-training loss-scale probe for the JEPA embedding regime.

Measures the raw magnitude of every embedding-loss term at unit weight over a
few batches, reusing the collection, filtering and reporting machinery of the
backbone loss-scale probe.
"""

from __future__ import annotations

import copy

from pipelines.backbone.training.loss_probe import LossScaleProbe
from pipelines.jepa.training.loss           import Loss as EmbeddingLoss


class JepaEmbeddingLossProbe(LossScaleProbe):
    """Measures each embedding-loss term at unit weight and suggests balancing weights.

    Attributes:
        EMBEDDING_TERMS    : Pairs of use flag and weight field of every probed term.
        autoencoder        : Coupled profile autoencoder of the run.
        target_provider    : TargetProvider producing the target embedding branch.
        norm_stats         : Output normalizer of the training dataset.
        ppg                : Number of parameters per Gaussian.
        profile_normalizer : Curve normalizer of the profile autoencoder run.
        loss_cfg           : Copy of the embedding loss with every probed term enabled at weight 1.0.
    """

    SHAPING_NOTE = "Raw values are shaped by the embedding-loss settings (smoothl1_beta, huber_delta, charbonnier_eps) and the coupled autoencoder's embedding normalization."

    EMBEDDING_TERMS = (
        ("use_embedding_mse",         "weight_embedding_mse"),
        ("use_embedding_cosine",      "weight_embedding_cosine"),
        ("use_embedding_smoothl1",    "weight_embedding_smoothl1"),
        ("use_embedding_huber",       "weight_embedding_huber"),
        ("use_embedding_charbonnier", "weight_embedding_charbonnier"),
        ("use_embedding_variance",    "weight_embedding_variance"),
        ("use_embedding_covariance",  "weight_embedding_covariance"),
        ("use_curve_recon",           "weight_curve_recon"),
        ("use_curve_sobolev",         "weight_curve_sobolev"),
    )

    def __init__(self, probe_cfg, embedding_cfg, autoencoder, target_provider, norm_stats, params_per_gaussian: int, profile_normalizer, logger=None):
        """Builds an embedding-loss configuration with the probed terms at unit weight.

        Args:
            probe_cfg           : Probe configuration; its `enabled_losses` overrides the default on/off state per term.
            embedding_cfg       : Embedding loss the probe copies its remaining settings from.
            autoencoder         : Coupled profile autoencoder.
            target_provider     : TargetProvider for the target embedding branch.
            norm_stats          : Output normalizer of the training dataset.
            params_per_gaussian : Number of parameters per Gaussian.
            profile_normalizer  : Curve normalizer of the profile autoencoder run.
            logger              : Logger receiving the probe report.

        Raises:
            ValueError : If `enabled_losses` names a flag no embedding term declares.
        """
        self.probe_cfg          = probe_cfg
        self.autoencoder        = autoencoder
        self.target_provider    = target_provider
        self.norm_stats         = norm_stats
        self.ppg                = params_per_gaussian
        self.profile_normalizer = profile_normalizer
        self.logger             = logger
        self.loss_cfg           = copy.deepcopy(embedding_cfg)

        known   = {use_flag for use_flag, _ in self.EMBEDDING_TERMS}
        unknown = sorted(set(probe_cfg.enabled_losses) - known)
        if unknown:
            raise ValueError(f"Unknown probe enabled_losses keys {unknown}; valid flags for the embedding regime: {sorted(known)}")

        for use_flag, weight_key in self.EMBEDDING_TERMS:
            override = probe_cfg.enabled_losses.get(use_flag)
            setattr(self.loss_cfg, use_flag, override if override is not None else True)

            if getattr(self.loss_cfg, use_flag):
                setattr(self.loss_cfg, weight_key, 1.0)

    def _build_criterion(self, z_axis):
        """Builds the embedding-match loss over the probe configuration.

        Args:
            z_axis : Z sample positions in metres, of shape (profile_length,).

        Returns:
            An EmbeddingLoss with every probed term at unit weight.
        """
        return EmbeddingLoss(self.autoencoder, self.target_provider, self.loss_cfg, z_axis, self.norm_stats, self.ppg, self.profile_normalizer, self.logger)

    @staticmethod
    def _criterion_result(criterion, prediction, targets) -> dict:
        """Evaluates the embedding criterion.

        Args:
            criterion  : The unit-weight embedding loss under measurement.
            prediction : Raw predicted embedding batch.
            targets    : Normalized ground-truth parameter batch.

        Returns:
            The criterion's loss dict.
        """
        return criterion(prediction, targets)
