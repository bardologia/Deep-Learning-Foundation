"""Entry point for Optuna hyperparameter tuning, in both scheduler and worker roles.

Without ``--worker`` the process creates or resumes the studies and fans workers out; with
it, the same script attaches to one study through its storage URL and runs trials on a GPU.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner

from tools.runtime.run_tag import RunTag


def main() -> None:
    """Runs one tuning worker against an existing study, or schedules the studies.

    In worker mode the worker flags are checked before the GPU is pinned, then the
    resolved config is read from the given run directory and trials are run on the
    pinned GPU. In scheduler mode the config comes from the CLI, is optionally reloaded
    from the run's resolved config when resuming, and curriculum inheritance is applied
    before scheduling.

    A ``resume_from`` path continues the tuning run holding it with the
    configuration that run recorded, the command-line overrides applied on
    top; ``--resume`` with a run tag does the same for a run named by its tag.

    Raises:
        SystemExit : If worker mode lacks ``--model``, ``--study-name``, ``--storage-url``, ``--n-trials``, ``--worker-index``, ``--run-tag`` or ``--run-dir``, or if ``--resume`` is given without a run tag or with no resolved config at the expected path.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",       action="store_true")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--model",        type=str,  default=None, help="(scheduler) tune only this model; (worker) model being tuned")
    parser.add_argument("--gpu",          type=int,  default=0)
    parser.add_argument("--n-trials",     type=int,  default=None)
    parser.add_argument("--worker-index", type=int,  default=None)
    parser.add_argument("--study-name",   type=str,  default=None)
    parser.add_argument("--storage-url",  type=str,  default=None)
    parser.add_argument("--run-tag",      type=str,  default=None)
    parser.add_argument("--run-dir",      type=str,  default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.model is None:
            sys.exit("ERROR: --worker requires --model")
        if args.study_name is None or args.storage_url is None:
            sys.exit("ERROR: --worker requires --study-name and --storage-url")
        if args.n_trials is None:
            sys.exit("ERROR: --worker requires --n-trials")
        if args.worker_index is None:
            sys.exit("ERROR: --worker requires --worker-index")
        if args.run_tag is None or args.run_dir is None:
            sys.exit("ERROR: --worker requires --run-tag and --run-dir")

        EnvironmentPinner.gpu(args.gpu, expandable_segments=True)
    else:
        EnvironmentPinner.threads()

    from configuration.training                     import CurriculumInheritance, default_curriculum
    from configuration.tuning                       import TuningEntryConfig
    from pipelines.shared.orchestration.resumed_run import TuningResumedRun
    from pipelines.tuning.pipeline                  import TuningScheduler
    from pipelines.tuning.workers                   import TuningWorker
    from tools.runtime.config_cli                   import ConfigCli

    entry_script = Path(__file__).resolve()

    if args.worker:
        config = ConfigCli.load_resolved(TuningEntryConfig(), Path(args.run_dir) / "resolved_config.json")

        worker = TuningWorker(tag=args.run_tag, config=config)
        worker.run_worker(
            model_name   = args.model,
            gpu_id       = args.gpu,
            worker_index = args.worker_index,
            n_trials     = args.n_trials,
            study_name   = args.study_name,
            storage_url  = args.storage_url,
        )

    else:
        cli    = ConfigCli(TuningEntryConfig(), description="Optuna hyperparameter tuning; point resume_from at an existing tuning run to top its studies up with the configuration it recorded")
        config = cli.apply()

        resumed = None
        if config.resume_from:
            resumed     = TuningResumedRun(TuningEntryConfig, config, cli.overrides)
            config      = resumed.load()
            args.resume = True

        tag = args.run_tag or config.run_tag or RunTag.now()

        if args.resume and resumed is None:
            if args.run_tag is None and config.run_tag is None:
                sys.exit("ERROR: --resume requires --run-tag")

            resolved_path = Path(config.paths.log_base_dir) / tag / "resolved_config.json"
            if not resolved_path.exists():
                sys.exit(f"ERROR: --resume given but no resolved config found at {resolved_path}")

            config = ConfigCli.load_resolved(TuningEntryConfig(), resolved_path)
            config = ConfigCli.apply_overrides(config, cli.overrides)

        CurriculumInheritance(config.curriculum, default_curriculum(), cli.overrides).apply()

        scheduler = TuningScheduler(tag=tag, config=config, entry_script=entry_script, resumed=resumed)
        scheduler.schedule(target_model=args.model, resume=args.resume)


if __name__ == "__main__":
    main()
