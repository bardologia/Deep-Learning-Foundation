from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner

from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from tools.runtime.config_cli                  import ConfigCli


def main() -> None:
    config = ConfigCli(DataLoaderTuningEntryConfig(), description="Sweep DataLoader settings (batch size, workers, prefetch, pin-memory) to find the configuration that keeps the GPU fed").apply()

    EnvironmentPinner.gpu(config.gpu, expandable_segments=True)
    EnvironmentPinner.threads(1)

    from pipelines.dataloader_tuning.pipeline import DataLoaderTuningPipeline

    DataLoaderTuningPipeline(config).run()


if __name__ == "__main__":
    main()
