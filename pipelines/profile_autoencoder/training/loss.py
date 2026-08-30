"""Reconstruction loss of the standalone profile autoencoder."""

from __future__ import annotations

from tools.loss.reconstruction_loss import ReconstructionLoss, ReconstructionTerm


class Loss(ReconstructionLoss):
    """Summed curve-space loss configured from the profile autoencoder settings.

    Attributes:
        TERM_SPECS : Term kind, use flag and weight field of every available term, in evaluation order; the reported component key is ``curve_<kind>``.
    """

    TERM_SPECS = (
        ("mse",         "use_mse_curve",         "weight_mse_curve"),
        ("l1",          "use_l1_curve",          "weight_l1_curve"),
        ("huber",       "use_huber_curve",       "weight_huber_curve"),
        ("charbonnier", "use_charbonnier_curve", "weight_charbonnier_curve"),
        ("cosine",      "use_cosine_curve",      "weight_cosine_curve"),
        ("log_energy",  "use_log_energy_curve",  "weight_log_energy_curve"),
        ("tv",          "use_tv_curve",          "weight_tv_curve"),
        ("sobolev",     "use_sobolev",           "weight_sobolev"),
    )

    def __init__(self, ae_cfg) -> None:
        """Builds the enabled term set from the loss configuration.

        Args:
            ae_cfg : Profile autoencoder loss configuration supplying the ``use_*``/``weight_*`` term flags, ``huber_delta`` and ``charbonnier_eps``.

        Raises:
            ValueError : If no term is enabled.
        """
        terms = tuple(ReconstructionTerm(kind, f"curve_{kind}", getattr(ae_cfg, weight_key)) for kind, use_flag, weight_key in self.TERM_SPECS if getattr(ae_cfg, use_flag))

        super().__init__(terms, ae_cfg.huber_delta, ae_cfg.charbonnier_eps)
