from __future__ import annotations

import sys
from copy    import deepcopy
from pathlib import Path
from typing  import Callable

from tools.orchestration      import ExperimentStage, GpuJob
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from tools.runtime.run_tag    import RunTag


class SeedSet:
    @staticmethod
    def resolve(seeds, default_seed: int) -> list[int]:
        return list(seeds or []) or [default_seed]

    @staticmethod
    def run_name(base: str, seed: int) -> str:
        return f"{base}/seed{seed}"

    @staticmethod
    def base(run_name: str) -> str:
        return run_name.split("/seed")[0]

    @staticmethod
    def cli_args(seed: int | None) -> list[str]:
        return ["--seed", str(seed)] if seed is not None else []

    @staticmethod
    def units(bases, seeds) -> list[tuple[str, int | None, str]]:
        seeds = list(seeds or [])

        if not seeds:
            return [(base, None, base) for base in bases]

        return [(base, seed, SeedSet.run_name(base, seed)) for base in bases for seed in seeds]


class SeedSweepRunner:
    def __init__(self, config, runner_factory: Callable[[object], object]) -> None:
        self.config         = config
        self.runner_factory = runner_factory

    @staticmethod
    def base_run_name(config, base_label: str | None = None) -> str:
        if config.run_name:
            return config.run_name

        timestamp = RunTag.now()
        return f"{base_label}_{timestamp}" if base_label else timestamp

    def run(self):
        seeds = SeedSet.resolve(self.config.seeds, self.config.seed)

        if len(seeds) > 1:
            raise ValueError(f"SeedSweepRunner trains exactly one seed, got {seeds}; multi-seed runs fan out across the GPU pool via SeedFanoutScheduler")

        config      = deepcopy(self.config)
        config.seed = seeds[0]
        return self.runner_factory(config).run()


class SeedFanoutScheduler:

    SCHEDULER_FIELDS = ("gpus", "gpus_file", "poll_interval_s", "gpu", "seed", "seeds", "run_name", "infer_at_end")

    def __init__(self, config, cli_overrides: dict, entry_script: Path, base_name: str, run_dir: Path, infer_at_end: bool = False) -> None:
        self.config             = config
        self.entry_script       = Path(entry_script)
        self.base_name          = base_name
        self.infer_at_end       = infer_at_end
        self.run_dir            = Path(run_dir)
        self.log_dir            = self.run_dir / "batch_train_logs"
        self.results_path       = self.log_dir / "seed_sweep_results.json"
        self.infer_results_path = self.log_dir / "seed_sweep_infer_results.json"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self.logger = Logger(log_dir=str(self.log_dir), name="seed_sweep")
        self.stage  = ExperimentStage(config=config, run_tag="seed_sweep", logger=self.logger, entry_script=self.entry_script, run_dir=self.run_dir)

    @classmethod
    def for_runner(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None, infer_at_end: bool = False) -> "SeedFanoutScheduler":
        base           = SeedSweepRunner.base_run_name(config, base_label)
        probe          = deepcopy(config)
        probe.run_name = base
        run_dir        = Path(config.logdir) / runner_factory(probe)._resolve_run_name()

        return cls(config, cli_overrides, entry_script, base_name=base, run_dir=run_dir, infer_at_end=infer_at_end)

    def _job(self, run_name: str, seed: int) -> GpuJob:
        batched   = {"infer_after": False} if self.infer_at_end else {}
        overrides = {**self.forward_overrides, **batched, "logdir": str(self.config.logdir), "run_name": run_name, "seed": seed, "seeds": (seed,)}

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script)] + ConfigCli.to_argv(overrides),
            log_path = self.log_dir / f"seed{seed}.log",
        )

    def _inference_job(self, run_name: str, seed: int) -> GpuJob:
        overrides = {**self.forward_overrides, "infer_after": True, "resume": True, "logdir": str(self.config.logdir), "run_name": run_name, "seed": seed, "seeds": (seed,)}

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script)] + ConfigCli.to_argv(overrides),
            log_path = self.log_dir / f"seed{seed}_infer.log",
        )

    def _run_inference_pass(self, units: list[tuple[int, str]], training_results: list[dict]) -> list[dict]:
        trained = {result["name"] for result in training_results if result["status"] == "DONE"}
        todo    = [(seed, run_name) for seed, run_name in units if run_name in trained]

        self.logger.section("Batched inference")
        self.logger.kv_table({
            "Eligible" : len(todo),
            "Skipped"  : len(units) - len(todo),
            "GPUs"     : self.config.gpus,
            "Log dir"  : str(self.log_dir),
        }, title="Configuration")

        for _, run_name in units:
            if run_name not in trained:
                self.logger.warning(f"{run_name}: training failed, inference skipped")

        if not todo:
            return []

        jobs    = [self._inference_job(run_name, seed) for seed, run_name in todo]
        names   = [run_name for _, run_name in todo]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.infer_results_path)
        return results

    def run(self) -> None:
        seeds = SeedSet.resolve(self.config.seeds, self.config.seed)
        units = [(seed, SeedSet.run_name(self.base_name, seed)) for seed in seeds]

        self.logger.section("Seed fan-out")
        self.logger.kv_table({
            "Base run"      : self.base_name,
            "Run dir"       : str(self.run_dir),
            "Seeds"         : seeds,
            "GPUs"          : self.config.gpus,
            "GPU pool file" : str(self.stage.pool_file),
            "Infer at end"  : self.infer_at_end,
            "Forwarded overrides (scheduler options excluded)" : self.forward_overrides or "—",
            "Log dir"       : str(self.log_dir),
        }, title="Configuration")

        jobs    = [self._job(run_name, seed) for seed, run_name in units]
        names   = [run_name for _, run_name in units]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.results_path)

        infer_results = self._run_inference_pass(units, results) if self.infer_at_end else []

        self.logger.section("Summary")
        rows = [{"Seed": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in results]
        self.logger.metrics_table(rows, columns=["Seed", "Status", "Duration"])

        if infer_results:
            self.logger.subsection("Batched inference")
            infer_rows = [{"Seed": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in infer_results]
            self.logger.metrics_table(infer_rows, columns=["Seed", "Status", "Duration"])

        failed       = [r for r in results if r["status"] != "DONE"]
        infer_failed = [r for r in infer_results if r["status"] != "DONE"]
        self.stage._log_failures(failed + infer_failed)

        self.logger.close()

        problems = []
        if failed:
            problems.append(f"{len(failed)} of {len(results)} seed runs failed")
        if infer_failed:
            problems.append(f"{len(infer_failed)} of {len(infer_results)} inference jobs failed")

        if problems:
            raise SystemExit("; ".join(problems) + f"; see {self.log_dir}")
