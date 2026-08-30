"""Command-line entry point for image-autoencoder training runs.

Reads the ``--gpu`` selector before any CUDA-aware import so that the process
is pinned to a single device, then drives the shared ``SeedSweepLauncher`` with
the image-autoencoder entry configuration and single-run trainer.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Pins the selected GPU and launches the image-autoencoder seed sweep.

    Parses ``--gpu`` ahead of the torch import, sets the visible device with
    expandable CUDA segments, and runs ``SeedSweepLauncher`` over
    ``ImageAeEntryConfig``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)

    from configuration.training.image_autoencoder    import ImageAeEntryConfig
    from pipelines.image_autoencoder.training.pipeline import SingleTrainRunner
    from pipelines.shared.training.training_launcher            import SeedSweepLauncher

    SeedSweepLauncher(ImageAeEntryConfig(), SingleTrainRunner, "Image autoencoder training", entry_script=pathlib.Path(__file__).resolve()).run()


if __name__ == "__main__":
    main()
