from __future__ import annotations

import os
import sys
from pathlib import Path


class PyRatEnvironment:
    @staticmethod
    def ensure_conda_lib_on_ld_path() -> None:
        conda_lib = os.path.join(sys.prefix, "lib")
        ld_path   = os.environ.get("LD_LIBRARY_PATH", "")

        if conda_lib not in ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = conda_lib + (":" + ld_path if ld_path else "")

    @staticmethod
    def ensure_root_on_sys_path(pyrat_root: str) -> None:
        if pyrat_root not in sys.path:
            sys.path.insert(0, pyrat_root)

    @staticmethod
    def ensure(pyrat_root: str | Path) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        PyRatEnvironment.ensure_conda_lib_on_ld_path()
        PyRatEnvironment.ensure_root_on_sys_path(str(pyrat_root))
