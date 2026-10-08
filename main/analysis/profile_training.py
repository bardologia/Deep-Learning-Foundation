"""Entry point profiling full training steps of one model family with torch.profiler.

Thin CLI wrapper: parses a ``TrainingProfileEntryConfig`` and hands it to
``TrainingProfilePipeline``, which builds the family's feed target and records
forward, backward and optimizer steps into a chrome trace, an operator table
and a summary.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Pins the requested GPU, parses the profiling config and records the trace."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)
    EnvironmentPinner.threads()

    from configuration.benchmark.dataloader_tuning import TrainingProfileEntryConfig
    from pipelines.dataloader_tuning.profiling     import TrainingProfilePipeline
    from tools.monitoring.logger                   import Logger
    from tools.runtime.config_cli                  import ConfigCli

    entry = ConfigCli(TrainingProfileEntryConfig(), description="Build the configured family's real (dataset, model, loss) feed and profile full training steps with torch.profiler: warmup steps are observed, active steps recorded, and the run writes a chrome trace for chrome://tracing, a per-operator time table and a machine-readable summary").apply()

    with Logger(log_dir="logs", name="profile_training") as logger:
        TrainingProfilePipeline(entry, logger).run()


if __name__ == "__main__":
    main()
