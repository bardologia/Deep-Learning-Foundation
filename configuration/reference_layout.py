"""Filesystem layout the configuration defaults describe: the checkout itself.

Every path default under ``configuration/`` is relative to the repository root and
sits below one of the roots named here. The machine profile written by the installer
rebases those roots onto the directories of the current machine, so a new
installation never edits a dataclass default, and a machine without a profile keeps
working out of the checkout.
"""

from __future__ import annotations

from pathlib import Path


class ReferenceLayout:
    """Checkout-relative roots the configuration defaults are written against.

    Attributes:
        DATA_ROOT    : Directory holding every dataset.
        RUNS_ROOT    : Root of the run-output tree.
        RESULTS_ROOT : Root of the analysis outputs.
        DATASET_DIR  : Example dataset every training default points at.
        PARAMS_RUN   : Name of the Gaussian parameter-extraction run inside that dataset.
        PARAMETERS   : Parameter array of that run.
        CONSOLE_HOST : Address the web console listens on.
        CONSOLE_PORT : Port the web console listens on.
    """

    DATA_ROOT    = Path("datasets")
    RUNS_ROOT    = Path("runs")
    RESULTS_ROOT = Path("results")
    DATASET_DIR  = DATA_ROOT / "synthetic_small"
    PARAMS_RUN   = "params_k2"
    PARAMETERS   = DATASET_DIR / "params" / PARAMS_RUN / "parameters.npy"
    CONSOLE_HOST = "127.0.0.1"
    CONSOLE_PORT = 8772
