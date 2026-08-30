"""Entry point running the benchmark comparison stage on its own for one benchmark run.

Re-runs only the comparison of an existing benchmark run tree, so the report can be
regenerated without repeating training or inference.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner


def _resolve_run_tag(config) -> str:
    """Returns the benchmark run tag to compare, defaulting to the latest timestamped one.

    Args:
        config : Resolved ``ComparisonEntryConfig`` carrying ``run_tag`` and the log base directory to scan for benchmark run directories.

    Returns:
        The explicit ``config.run_tag`` when given, otherwise the alphabetically last
        timestamp-tagged run directory that contains a ``training`` subdirectory.

    Raises:
        SystemExit : If the base directory holds no benchmark run, or holds only custom-tagged runs whose ordering cannot identify the latest.
    """
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
    """Runs the benchmark comparison stage for the resolved run tag."""
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
