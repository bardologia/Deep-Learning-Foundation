
from __future__ import annotations

import os


class InheritedCudaMask:

    STASH_VAR = "MLTEMPLATE_INHERITED_CUDA_MASK"

    @classmethod
    def restored(cls) -> dict:
        env     = dict(os.environ)
        stashed = env.pop(cls.STASH_VAR, None)

        if stashed == "":
            env.pop("CUDA_VISIBLE_DEVICES", None)
        elif stashed is not None:
            env["CUDA_VISIBLE_DEVICES"] = stashed

        return env
