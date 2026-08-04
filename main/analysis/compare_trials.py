from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

from _bootstrap import EnvironmentPinner

from tools.runtime.run_tag import RunTag


def main() -> None:
    EnvironmentPinner.threads()

    from configuration.comparison            import TrialComparisonConfig
    from pipelines.comparison.trial_collector        import TrialCollector
    from pipelines.comparison.comparison_report      import TrialComparisonReport
    from tools.runtime.config_cli                    import ConfigCli
    from tools.monitoring.logger                     import Logger

    config = ConfigCli(TrialComparisonConfig(), description="Compare inference results across multiple training trials").apply()

    runs_dir = Path(config.runs_dir)
    base_out = Path(config.output_dir) if config.output_dir else runs_dir / "_comparison"
    out_dir  = base_out / RunTag.now()

    with Logger(log_dir=str(out_dir / "logs"), name="compare_trials") as logger:
        logger.section("Trial comparison")
        logger.kv_table({
            "Runs dir"       : str(runs_dir),
            "Trials"         : len(config.run_tags),
            "Compare images" : config.compare_images,
            "Compare GIFs"   : config.compare_gifs,
            "Output dir"     : str(out_dir),
        }, title="Configuration")

        collector = TrialCollector(runs_dir=runs_dir, run_tags=config.run_tags, logger=logger)
        records   = collector.collect()

        report = TrialComparisonReport(
            records         = records,
            out_dir         = out_dir,
            compare_images  = config.compare_images,
            compare_gifs    = config.compare_gifs,
            embed_images    = config.embed_images,
            logger          = logger,
            seed_dispersion = collector.seed_dispersion,
        )

        written = report.write_all()

        logger.subsection("Reports written")
        for path in written:
            logger.info(str(path))

        logger.info(f"\nComparison written to: {out_dir}")


if __name__ == "__main__":
    main()
