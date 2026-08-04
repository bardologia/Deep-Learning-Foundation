from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    EnvironmentPinner.threads()

    from configuration.cross_validation      import CrossValidationConfig
    from configuration.training              import CurriculumInheritance, default_curriculum
    from pipelines.cross_validation.pipeline import CrossValidationPipeline
    from tools.runtime.config_cli            import ConfigCli

    cli    = ConfigCli(CrossValidationConfig(), description="K-fold cross-validation")
    config = cli.apply()

    CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

    pipeline = CrossValidationPipeline(config=config, entry_script=Path(__file__).resolve())
    pipeline.run()


def _worker(stage: str, fold_index: int, seed: int | None, split: str | None, gpu_id: int, run_tag: str, run_dir: str | None) -> None:
    EnvironmentPinner.gpu(gpu_id)

    from configuration.cross_validation import CrossValidationConfig
    from pipelines.cross_validation.workers                import FoldInferenceWorker, FoldTrainingWorker
    from tools.runtime.config_cli                          import ConfigCli

    config = ConfigCli.load_worker_config(CrossValidationConfig(), run_tag, run_dir)

    if stage == "train":
        FoldTrainingWorker(config=config, run_tag=run_tag).run(fold_index=fold_index, seed=seed)
    else:
        FoldInferenceWorker(config=config, run_tag=run_tag).run(fold_index=fold_index, split=split, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",  type=str, default=None, choices=["train", "infer"])
    parser.add_argument("--fold",    type=int, default=None)
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--split",   type=str, default=None)
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.fold is None or args.run_tag is None:
            sys.exit("ERROR: --worker requires --fold and --run-tag")
        if args.worker == "infer" and args.split is None:
            sys.exit("ERROR: --worker infer requires --split")
        _worker(stage=args.worker, fold_index=args.fold, seed=args.seed, split=args.split, gpu_id=args.gpu, run_tag=args.run_tag, run_dir=args.run_dir)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
