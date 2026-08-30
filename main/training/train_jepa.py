"""Command-line entry point for JEPA predictor training runs.

Reads the ``--gpu`` selector before any CUDA-aware import so that the process
is pinned to a single device, then drives the JEPA launcher, which trains a
single run, fans seeds out, or schedules a trial sweep.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Pins the selected GPU and launches the JEPA predictor training entry.

    Parses ``--gpu`` ahead of the torch import, sets the visible device with
    expandable CUDA segments, and runs ``JepaTrainingLauncher`` over
    ``JepaEntryConfig``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)

    from pipelines.jepa.training.launcher import JepaTrainingLauncher

    JepaTrainingLauncher(entry_script=pathlib.Path(__file__).resolve()).run()


if __name__ == "__main__":
    main()
