
from __future__ import annotations


class TrialCatalog:


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
