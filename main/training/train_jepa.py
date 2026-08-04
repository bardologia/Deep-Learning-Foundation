from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse

from _bootstrap import EnvironmentPinner


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)

    from configuration.training.jepa       import JepaEntryConfig
    from pipelines.jepa.training.pipeline   import SingleTrainRunner
    from pipelines.shared.training.training_launcher import SeedSweepLauncher

    SeedSweepLauncher(JepaEntryConfig(), SingleTrainRunner, "JEPA predictor training", entry_script=pathlib.Path(__file__).resolve()).run()


if __name__ == "__main__":
    main()
