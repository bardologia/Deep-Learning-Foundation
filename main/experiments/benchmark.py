from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from _bootstrap import EnvironmentPinner


def _scheduler() -> None:
    EnvironmentPinner.threads()

    from configuration.benchmark import BenchmarkConfig
    from pipelines.benchmark.pipeline               import BenchmarkPipeline
    from tools.runtime.config_cli                   import ConfigCli

    config = ConfigCli(BenchmarkConfig(), description="Full architecture benchmark").apply()

    pipeline = BenchmarkPipeline(config=config, entry_script=Path(__file__).resolve())
    pipeline.run()


def _worker(stage: str, model_name: str, seed: int | None, gpu_id: int, run_tag: str, run_dir: str | None, loss_component: str | None) -> None:
    EnvironmentPinner.gpu(gpu_id)

    from configuration.benchmark import BenchmarkConfig
    from pipelines.benchmark.workers                import InferenceWorker, MaxBatchWorker, TrainingWorker
    from tools.runtime.config_cli                   import ConfigCli

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
        worker.run(model_name=model_name, seed=seed, loss_component=loss_component)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker",         type=str, default=None, choices=["maxbatch", "train", "infer"])
    parser.add_argument("--model",          type=str, default=None)
    parser.add_argument("--loss-component", type=str, default=None)
    parser.add_argument("--seed",           type=int, default=None)
    parser.add_argument("--gpu",            type=int, default=0)
    parser.add_argument("--run-tag",        type=str, default=None)
    parser.add_argument("--run-dir",        type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.worker:
        if args.model is None or args.run_tag is None:
            sys.exit("ERROR: --worker requires --model and --run-tag")
        _worker(stage=args.worker, model_name=args.model, seed=args.seed, gpu_id=args.gpu, run_tag=args.run_tag, run_dir=args.run_dir, loss_component=args.loss_component)
    else:
        _scheduler()


if __name__ == "__main__":
    main()
