from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner


def _resolve_run_tag(config) -> str:
    if config.run_tag is not None:
        return config.run_tag

    base       = Path(config.paths.log_base_dir)
    candidates = sorted(d.name for d in base.iterdir() if d.is_dir() and (d / "training").is_dir()) if base.is_dir() else []

    if not candidates:
        sys.exit(f"ERROR: no benchmark runs found under {base}")

    return candidates[-1]


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.comparison   import ComparisonEntryConfig
    from pipelines.benchmark.stages import ComparisonStage
    from tools.runtime.config_cli   import ConfigCli
    from tools.monitoring.logger    import Logger

    config = ConfigCli(ComparisonEntryConfig(), description="Standalone benchmark comparison").apply()
    tag    = _resolve_run_tag(config)

    logger = Logger(log_dir=str(Path(config.paths.log_base_dir) / tag / "pipeline"), name="compare_runs")

    stage   = ComparisonStage(config=config, run_tag=tag, logger=logger, reference_model=config.reference_model, embed_images=config.embed_images)
    out_dir = stage.run()

    logger.info(f"Comparison written to: {out_dir}")
    logger.close()

if __name__ == "__main__":
    main()
