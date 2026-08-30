"""Entry point sweeping DataLoader settings to find the configuration that keeps the GPU fed.

Runs on one pinned GPU with the host thread pool limited to a single thread, so measured
throughput reflects the loader settings (batch size, workers, prefetch, pin-memory).
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse

from _bootstrap import EnvironmentPinner


def main() -> None:
    """Pins the requested GPU and runs the DataLoader tuning sweep."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()

    EnvironmentPinner.gpu(args.gpu, expandable_segments=True)
    EnvironmentPinner.threads(1)

    from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
    from pipelines.dataloader_tuning.pipeline      import DataLoaderTuningPipeline
    from tools.runtime.config_cli                  import ConfigCli

    config = ConfigCli(DataLoaderTuningEntryConfig(), description="Sweep DataLoader settings (batch size, workers, prefetch, pin-memory) to find the configuration that keeps the GPU fed").apply()

    DataLoaderTuningPipeline(config).run()


if __name__ == "__main__":
    main()
