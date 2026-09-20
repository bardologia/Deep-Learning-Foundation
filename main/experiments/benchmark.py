"""Entry point for the architecture benchmark, in both scheduler and worker roles.

Without ``--worker`` the process schedules the whole benchmark; with it, the same script is
re-invoked per GPU to run one max-batch probe, training or inference unit.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    """Runs the benchmark pipeline that fans work out to worker invocations of this script.

    With ``resume_from`` set, the configuration the run recorded is loaded
    first and the command-line overrides are applied on top, so the run is
    finished with the settings it was launched with unless told otherwise.
    """
    EnvironmentPinner.threads()

    from configuration.benchmark                    import BenchmarkConfig
    from pipelines.benchmark.pipeline               import BenchmarkPipeline
    from pipelines.shared.orchestration.resumed_run import ResumedRun
    from tools.runtime.config_cli                   import ConfigCli

    cli    = ConfigCli(BenchmarkConfig(), description="Full architecture benchmark; point resume_from at an existing run to finish its missing units and inference passes with the configuration it recorded")
    config = cli.apply()

    resumed = None
    if config.resume_from:
        resumed = ResumedRun(BenchmarkConfig, config, cli.overrides)
        config  = resumed.load()

    pipeline = BenchmarkPipeline(config=config, entry_script=Path(__file__).resolve(), resumed=resumed)
    pipeline.run()


def _worker(stage: str, model_name: str, seed: int | None, gpu_id: int, run_tag: str, run_dir: str | None, loss_component: str | None, run_name: str | None) -> None:
    """Runs one benchmark work unit on a single pinned GPU.

    Args:
        stage          : Which worker to run, one of ``maxbatch``, ``train`` or ``infer``.
        model_name     : Architecture name the unit operates on.
        seed           : Training seed for the train and infer stages; ignored by ``maxbatch``.
        gpu_id         : Index of the GPU this process is pinned to.
        run_tag        : Tag of the benchmark run whose resolved config is loaded.
        run_dir        : Explicit run directory holding the resolved config, or None to derive it.
        loss_component : Loss-component variant for the train and infer stages, or None.
        run_name       : Directory of the unit relative to the training directory, or None to compose it.
    """
    EnvironmentPinner.gpu(gpu_id, expandable_segments=True)

    from configuration.benchmark     import BenchmarkConfig
    from pipelines.benchmark.workers import InferenceWorker, MaxBatchWorker, TrainingWorker
    from tools.runtime.config_cli    import ConfigCli

    config = ConfigCli.load_worker_config(BenchmarkConfig(), run_tag, run_dir)

    workers = {
        "maxbatch" : MaxBatchWorker,
        "train"    : TrainingWorker,
        "infer"    : InferenceWorker,
    }

    worker = workers[stage](config=config, run_tag=run_tag)

    if stage == "maxbatch":
        worker.run(model_name=model_name)
    else:
        worker.run(model_name=model_name, seed=seed, loss_component=loss_component, run_name=run_name)


def main() -> None:
    """Dispatches to the benchmark scheduler or to a single worker unit.

    Raises:
        SystemExit : If ``--worker`` is given without both ``--model`` and ``--run-tag``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",         type=str, default=None, choices=["maxbatch", "train", "infer"])
    parser.add_argument("--model",          type=str, default=None)
    parser.add_argument("--loss-component", type=str, default=None)
    parser.add_argument("--seed",           type=int, default=None)
    parser.add_argument("--gpu",            type=int, default=0)
    parser.add_argument("--run-tag",        type=str, default=None)
    parser.add_argument("--run-dir",        type=str, default=None)
    parser.add_argument("--run-name",       type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.model is None or args.run_tag is None:
            sys.exit("ERROR: --worker requires --model and --run-tag")
        _worker(stage=args.worker, model_name=args.model, seed=args.seed, gpu_id=args.gpu, run_tag=args.run_tag, run_dir=args.run_dir, loss_component=args.loss_component, run_name=args.run_name)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
