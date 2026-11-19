
from __future__ import annotations


class TrialCatalog:


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
        return {
            "pL11" : {"use_param_l1": True, "weight_param_l1": 1.0},
        }

    @classmethod
    def complete_losses(cls) -> dict:
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
        return {
            "none" : {"use_active_normalization": False, "presence_balance": False},
            "A"    : {"use_active_normalization": True,  "presence_balance": False},
            "B"    : {"use_active_normalization": False, "presence_balance": True},
            "AB"   : {"use_active_normalization": True,  "presence_balance": True},
        }

    @staticmethod
    def context_trials() -> list:
        return [
            {"label" : "mlp",   "backbone" : "pixel_mlp", "features" : [1277] * 20},
            {"label" : "cnn01", "backbone" : "local_cnn", "features" : [1277] * 10, "block_kernels" : [1] * 10},
            {"label" : "cnn09", "backbone" : "local_cnn", "features" : [848]  * 10, "block_kernels" : [3] * 2 + [1] * 8},
            {"label" : "cnn29", "backbone" : "local_cnn", "features" : [502]  * 10, "block_kernels" : [3] * 7 + [1] * 3},
            {"label" : "cnn41", "backbone" : "local_cnn", "features" : [426]  * 10, "block_kernels" : [3] * 10},
        ]

    @staticmethod
    def reach_rungs() -> list:
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
        trials = {}

        for label, spec in cls.INPUT_TRIAL_BASES.items():
            trials[f"{label}-noaux"] = {**spec, "use_aux": False}
            trials[f"{label}-aux"]   = {**spec, "use_aux": True}

        return trials
