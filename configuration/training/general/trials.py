"""Catalog of the named experiment arms swept by the training launchers."""

from __future__ import annotations


class TrialCatalog:
    """Named loss, presence, architecture, augmentation and input sweep arms.

    Each factory returns the config overrides for one experiment axis, keyed by
    the short label that ends up in the run name.

    Attributes:
        CURVE_TERMS   : Maps a curve-loss label to its (enable flag, weight) field names.
        CURVE_WEIGHTS : Curve-loss weights each term is swept over.
    """


    CURVE_TERMS = {
        "mse"   : ("use_mse_curve",         "weight_mse_curve"),
        "l1"    : ("use_l1_curve",          "weight_l1_curve"),
        "huber" : ("use_huber_curve",       "weight_huber_curve"),
        "charb" : ("use_charbonnier_curve", "weight_charbonnier_curve"),
        "cos"   : ("use_cosine_curve",      "weight_cosine_curve"),
    }

    CURVE_WEIGHTS = [0.01, 0.05, 0.02]

    @staticmethod
    def warmup_losses() -> dict:
        """Returns the single param-L1 loss arm used during the curriculum warmup phase."""
        return {
            "pL11" : {"use_param_l1": True, "weight_param_l1": 1.0},
        }

    @classmethod
    def complete_losses(cls) -> dict:
        """Builds the loss arms pairing param-L1 with each curve term at each weight.

        Returns:
            Mapping from arm label to the loss-config overrides that enable
            param-L1 at weight 1.0 plus one curve term at one of `CURVE_WEIGHTS`.
        """
        losses = {}

        for label, (use_key, weight_key) in cls.CURVE_TERMS.items():
            for weight in cls.CURVE_WEIGHTS:
                losses[f"pL11-{label}{weight:g}"] = {
                    "use_param_l1"    : True,
                    "weight_param_l1" : 1.0,
                    use_key           : True,
                    weight_key        : weight,
                }

        return losses

    @staticmethod
    def presence_trials() -> dict:
        """Returns the four on/off combinations of active normalization and presence balance."""
        return {
            "none" : {"use_active_normalization": False, "presence_balance": False},
            "A"    : {"use_active_normalization": True,  "presence_balance": False},
            "B"    : {"use_active_normalization": False, "presence_balance": True},
            "AB"   : {"use_active_normalization": True,  "presence_balance": True},
        }

    @staticmethod
    def context_trials() -> list:
        """Returns the constant-depth rungs walking the context ladder.

        The pixel-MLP rung mirrors ``cnn01``'s exact 1x1 convolution stack on
        the ``pixel_mlp`` backbone, so the classic no-context baseline enters
        the ladder at the same depth and parameter budget as the CNN rungs.
        """
        return [
            {"label" : "mlp",   "backbone" : "pixel_mlp", "features" : [1277] * 20},
            {"label" : "cnn01", "backbone" : "local_cnn", "features" : [1277] * 10, "block_kernels" : [1] * 10},
            {"label" : "cnn09", "backbone" : "local_cnn", "features" : [848]  * 10, "block_kernels" : [3] * 2 + [1] * 8},
            {"label" : "cnn29", "backbone" : "local_cnn", "features" : [502]  * 10, "block_kernels" : [3] * 7 + [1] * 3},
            {"label" : "cnn41", "backbone" : "local_cnn", "features" : [426]  * 10, "block_kernels" : [3] * 10},
        ]

    @staticmethod
    def reach_rungs() -> list:
        """Returns the local-CNN and U-Net arms of the spatial-reach experiment."""
        return [
            {"label" : "cnn33", "backbone" : "local_cnn", "features" : [478] * 8, "dropout" : 0.15, "trunk_wd" : 1e-4},
            {"label" : "unet",  "backbone" : "unet",      "dropout" : 0.15},
        ]

    INPUT_TRIAL_BASES = {
        "amp-allsec-noderived" : {"tracks": "all",     "use_primary": True,  "use_secondaries": True,  "use_derived": False},
        "noamp-allsec-derived" : {"tracks": "all",     "use_primary": False, "use_secondaries": False, "use_derived": True},
        "amp-allsec-derived"   : {"tracks": "all",     "use_primary": True,  "use_secondaries": True,  "use_derived": True},
        "amp-redsec-derived"   : {"tracks": "reduced", "use_primary": True,  "use_secondaries": True,  "use_derived": True},
        "amp-redsec-noderived" : {"tracks": "reduced", "use_primary": True,  "use_secondaries": True,  "use_derived": False},
        "noamp-redsec-derived" : {"tracks": "reduced", "use_primary": False, "use_secondaries": False, "use_derived": True},
    }

    @classmethod
    def input_trials(cls) -> dict:
        """Builds the input-channel arms crossing amplitudes, secondaries and derived channels with the aux channel.

        Returns:
            Mapping from arm label to its input specification, every base arm
            of `INPUT_TRIAL_BASES` duplicated as an explicit no-aux and
            with-aux variant.
        """
        trials = {}

        for label, spec in cls.INPUT_TRIAL_BASES.items():
            trials[f"{label}-noaux"] = {**spec, "use_aux": False}
            trials[f"{label}-aux"]   = {**spec, "use_aux": True}

        return trials


class ProfileAeTrialCatalog:
    """Named architecture and loss sweep arms of the profile-autoencoder entry."""

    @staticmethod
    def arch_trials() -> list:
        """Returns one rung per registered profile-autoencoder family at its default settings."""
        return [
            {"label" : "mlp",      "ae_model_name" : "mlp_ae"},
            {"label" : "conv1d",   "ae_model_name" : "conv1d_ae"},
            {"label" : "tcn",      "ae_model_name" : "tcn_ae"},
            {"label" : "gru",      "ae_model_name" : "gru_ae"},
            {"label" : "resmlp",   "ae_model_name" : "resmlp_ae"},
            {"label" : "trans1d",  "ae_model_name" : "transformer1d_ae"},
            {"label" : "cnn_attn", "ae_model_name" : "cnn_attn_ae"},
        ]

    @staticmethod
    def loss_trials() -> dict:
        """Returns the reconstruction-loss arms comparing the pure penalties and mse-anchored shape pairings."""
        return {
            "mse"        : {"use_mse_curve": True, "weight_mse_curve": 1.0},
            "l1"         : {"use_l1_curve": True, "weight_l1_curve": 1.0},
            "huber"      : {"use_huber_curve": True, "weight_huber_curve": 1.0},
            "charb"      : {"use_charbonnier_curve": True, "weight_charbonnier_curve": 1.0},
            "mse-cos0.5" : {"use_mse_curve": True, "weight_mse_curve": 1.0, "use_cosine_curve": True, "weight_cosine_curve": 0.5},
            "mse-lE0.1"  : {"use_mse_curve": True, "weight_mse_curve": 1.0, "use_log_energy_curve": True, "weight_log_energy_curve": 0.1},
            "mse-tv0.1"  : {"use_mse_curve": True, "weight_mse_curve": 1.0, "use_tv_curve": True, "weight_tv_curve": 0.1},
        }


class JepaTrialCatalog:
    """Named embedding-loss sweep arms of the JEPA entry."""

    @staticmethod
    def embedding_losses() -> dict:
        """Returns the latent-loss arms comparing pure penalties and mse-anchored regularizer pairings."""
        return {
            "mse1"                : {"use_embedding_mse": True, "weight_embedding_mse": 1.0},
            "sl1"                 : {"use_embedding_smoothl1": True, "weight_embedding_smoothl1": 1.0},
            "hub1"                : {"use_embedding_huber": True, "weight_embedding_huber": 1.0},
            "cha1"                : {"use_embedding_charbonnier": True, "weight_embedding_charbonnier": 1.0},
            "mse1-cos0.5"         : {"use_embedding_mse": True, "weight_embedding_mse": 1.0, "use_embedding_cosine": True, "weight_embedding_cosine": 0.5},
            "mse1-var0.1"         : {"use_embedding_mse": True, "weight_embedding_mse": 1.0, "use_embedding_variance": True, "weight_embedding_variance": 0.1},
            "mse1-cov0.05"        : {"use_embedding_mse": True, "weight_embedding_mse": 1.0, "use_embedding_covariance": True, "weight_embedding_covariance": 0.05},
            "mse1-var0.1-cov0.05" : {"use_embedding_mse": True, "weight_embedding_mse": 1.0, "use_embedding_variance": True, "weight_embedding_variance": 0.1, "use_embedding_covariance": True, "weight_embedding_covariance": 0.05},
        }
