from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner

from configuration.synthesis  import GenerateDatasetEntryConfig
from tools.runtime.config_cli import ConfigCli


def main() -> None:
    config = ConfigCli(GenerateDatasetEntryConfig(), description="Generate a synthetic Gaussian-mixture dataset with the on-disk layout the training and inference pipelines expect").apply()

    EnvironmentPinner.threads()

    from pipelines.processing.synthesis.pipeline import SyntheticDatasetSession

    SyntheticDatasetSession(config.synthesis, log_dir=Path(config.log_dir)).execute()


if __name__ == "__main__":
    main()
