
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner


def _resolve_run_tag(config) -> str:
    if config.run_tag is not None:
        return config.run_tag

    from tools.runtime.run_tag import RunTag

    base       = Path(config.paths.log_base_dir)
    runs       = sorted(d.name for d in base.iterdir() if d.is_dir() and (d / "training").is_dir()) if base.is_dir() else []
    candidates = [name for name in runs if RunTag.is_tag(name)]

    if not candidates and runs:
        sys.exit(f"ERROR: no timestamp-tagged benchmark run under {base}; the run(s) {runs} carry custom tags, whose alphabetical order says nothing about which is the latest. Name the one to compare with --run-tag")
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

    with Logger(log_dir=str(Path(config.paths.log_base_dir) / tag / "pipeline"), name="compare_runs") as logger:
        stage   = ComparisonStage(config=config, run_tag=tag, logger=logger, embed_images=config.embed_images, inference_split=config.inference_split)
        out_dir = stage.run()

        logger.info(f"Comparison written to: {out_dir}")


if __name__ == "__main__":
    main()
