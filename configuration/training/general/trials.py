from __future__ import annotations


class TrialCatalog:

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
