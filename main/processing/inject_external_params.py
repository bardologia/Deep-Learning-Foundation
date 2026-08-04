from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner

from configuration.param_extraction import InjectExternalParamsEntryConfig
from tools.runtime.config_cli       import ConfigCli
from tools.monitoring.logger        import Logger


def main() -> None:
    config = ConfigCli(InjectExternalParamsEntryConfig(), description="Convert externally fitted Gaussian parameter files into parameter runs inside preprocessed datasets").apply()

    EnvironmentPinner.threads()

    from pipelines.processing.external_params.injection import ExternalParamsInjectionPipeline
    from pipelines.processing.external_params.sources   import ExternalSourceResolver
    from pipelines.shared.dataset.dataset_queue         import DatasetQueueResolver

    logger       = Logger(log_dir="logs", name="inject_external_params")
    base_path    = Path(config.dataset_base_path)
    dataset_dirs = DatasetQueueResolver(base_path, config.dataset_filter).resolve()
    sources      = ExternalSourceResolver(config.source_files, config.source_windows, logger).resolve()

    logger.section("Injection queue")
    logger.kv_table({
        "Datasets"      : len(dataset_dirs),
        "Queue"         : ", ".join(d.name for d in dataset_dirs),
        "Sources"       : len(sources),
        "Source order"  : config.source_order,
        "Slots"         : config.k_slots or "as many as the sources carry",
        "Run name"      : f"{config.output_prefix}_{config.output_suffix}",
        "Base path"     : str(base_path),
        "Filter"        : config.dataset_filter or "all dataset directories",
        "Overwrite"     : config.overwrite,
    }, title="Configuration")

    for index, dataset_dir in enumerate(dataset_dirs):
        logger.section(f"[Dataset {index + 1}/{len(dataset_dirs)}] {dataset_dir.name}")

        outputs = ExternalParamsInjectionPipeline(config, dataset_dir, sources).run()

        logger.kv_table({name: str(path) for name, path in outputs.items()}, title=f"Outputs {dataset_dir.name}")

    logger.close()


if __name__ == "__main__":
    main()
