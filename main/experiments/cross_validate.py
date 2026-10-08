"""Entry point for K-fold cross-validation, in both scheduler and worker roles.

Without ``--worker`` the process schedules every fold; with it, the same script is
re-invoked per GPU to train or run inference for one fold and seed.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    """Applies curriculum inheritance and runs the cross-validation pipeline, which logs the inherited fields.

    With ``resume_from`` set, the configuration the run recorded is loaded
    first and the command-line overrides are applied on top, so the run is
    finished with the settings it was launched with unless told otherwise.
    """
    EnvironmentPinner.threads()

    from configuration.cross_validation             import CrossValidationConfig
    from configuration.training                     import CurriculumInheritance, default_curriculum
    from pipelines.cross_validation.pipeline        import CrossValidationPipeline
    from pipelines.shared.orchestration.resumed_run import ResumedRun
    from tools.runtime.config_cli                   import ConfigCli

    cli    = ConfigCli(CrossValidationConfig(), description="K-fold cross-validation; point resume_from at an existing run to finish its missing folds, seeds and inference splits with the configuration it recorded")
    config = cli.apply()

    resumed = None
    if config.resume_from:
        resumed = ResumedRun(CrossValidationConfig, config, cli.overrides)
        config  = resumed.load()

    inherited = CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

    pipeline = CrossValidationPipeline(config=config, entry_script=Path(__file__).resolve(), resumed=resumed, inherited=inherited)
    pipeline.run()


def _worker(stage: str, fold_index: int, seed: int | None, gpu_id: int, run_tag: str, run_dir: str | None, run_name: str | None) -> None:
    """Trains or runs inference for one cross-validation fold on a single pinned GPU.

    Args:
        stage      : ``train`` to fit the fold, anything else to run fold inference.
        fold_index : Zero-based index of the fold to process.
        seed       : Training seed for this fold, or None for the pipeline default.
        gpu_id     : Index of the GPU this process is pinned to.
        run_tag    : Tag of the cross-validation run whose resolved config is loaded.
        run_dir    : Explicit run directory holding the resolved config, or None to derive it.
        run_name   : Directory of the unit relative to the folds directory, or None to compose it from the fold and seed.
    """
    EnvironmentPinner.gpu(gpu_id, expandable_segments=True)

    from configuration.cross_validation     import CrossValidationConfig
    from pipelines.cross_validation.workers import FoldInferenceWorker, FoldTrainingWorker
    from tools.runtime.config_cli           import ConfigCli

    config = ConfigCli.load_worker_config(CrossValidationConfig(), run_tag, run_dir)

    if stage == "train":
        FoldTrainingWorker(config=config, run_tag=run_tag).run(fold_index=fold_index, seed=seed, run_name=run_name)
    else:
        FoldInferenceWorker(config=config, run_tag=run_tag).run(fold_index=fold_index, seed=seed, run_name=run_name)


def main() -> None:
    """Dispatches to the cross-validation scheduler or to a single fold worker.

    Raises:
        SystemExit : If ``--worker`` is given without both ``--fold`` and ``--run-tag``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",   type=str, default=None, choices=["train", "infer"])
    parser.add_argument("--fold",     type=int, default=None)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--gpu",      type=int, default=0)
    parser.add_argument("--run-tag",  type=str, default=None)
    parser.add_argument("--run-dir",  type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.fold is None or args.run_tag is None:
            sys.exit("ERROR: --worker requires --fold and --run-tag")
        _worker(stage=args.worker, fold_index=args.fold, seed=args.seed, gpu_id=args.gpu, run_tag=args.run_tag, run_dir=args.run_dir, run_name=args.run_name)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
