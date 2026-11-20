
from __future__ import annotations

from pathlib import Path


class ReferenceLayout:

    DATA_ROOT    = Path("datasets")
    RUNS_ROOT    = Path("runs")
    RESULTS_ROOT = Path("results")
    DATASET_DIR  = DATA_ROOT / "synthetic"
    CONSOLE_HOST = "127.0.0.1"
    CONSOLE_PORT = 8772
