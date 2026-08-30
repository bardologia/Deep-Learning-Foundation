"""Command-line entry point for single-trunk backbone training runs.

Reads the ``--gpu`` selector before any CUDA-aware import so that the process
is pinned to a single device, then hands control to
``BackboneTrainingLauncher``, which owns configuration parsing and the seed
sweep.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Pins the selected GPU and launches backbone training.

    Parses ``--gpu`` ahead of the torch import, sets the visible device with
    expandable CUDA segments, and runs ``BackboneTrainingLauncher`` for this
    entry script.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)

    from pipelines.backbone.training.launcher import BackboneTrainingLauncher
    BackboneTrainingLauncher(entry_script=Path(__file__).resolve()).run()


if __name__ == "__main__":
    main()
