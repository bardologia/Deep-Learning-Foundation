from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    EnvironmentPinner.threads()

    from configuration.patch_sweep       import PatchSweepConfig
    from configuration.training          import CurriculumInheritance, default_curriculum
    from pipelines.patch_sweep.pipeline  import PatchSweepPipeline
    from tools.runtime.config_cli        import ConfigCli

    cli    = ConfigCli(PatchSweepConfig(), description="Patch-size sweep per dataset")
    config = cli.apply()

    CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

    pipeline = PatchSweepPipeline(config=config, entry_script=Path(__file__).resolve())
    pipeline.run()


def _worker(unit: str, seed: int | None, gpu_id: int, run_tag: str, run_dir: str | None) -> None:
    EnvironmentPinner.gpu(gpu_id)

    from configuration.patch_sweep      import PatchSweepConfig
    from pipelines.patch_sweep.workers  import SweepTrainingWorker
    from tools.runtime.config_cli       import ConfigCli

    config = ConfigCli.load_worker_config(PatchSweepConfig(), run_tag, run_dir)

    SweepTrainingWorker(config=config, run_tag=run_tag).run(unit_name=unit, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",  type=str, default=None, choices=["train"])
    parser.add_argument("--unit",    type=str, default=None)
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.unit is None or args.run_tag is None:
            sys.exit("ERROR: --worker requires --unit and --run-tag")
        _worker(unit=args.unit, seed=args.seed, gpu_id=args.gpu, run_tag=args.run_tag, run_dir=args.run_dir)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
