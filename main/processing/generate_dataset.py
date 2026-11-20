from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _bootstrap import EnvironmentPinner

from configuration.generation import GenerateDatasetEntryConfig
from tools.runtime.config_cli import ConfigCli


def main() -> None:
    config = ConfigCli(GenerateDatasetEntryConfig(), description="Generate the synthetic dense regression dataset: smooth input fields and target fields given by a known nonlinear, spatially local function of them").apply()

    EnvironmentPinner.threads()

    from pipelines.processing.generation import GenerateDatasetSession

    GenerateDatasetSession(config).execute()


if __name__ == "__main__":
    main()
