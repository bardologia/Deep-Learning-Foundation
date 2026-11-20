
from __future__ import annotations

import re
import sys
from copy    import deepcopy
from pathlib import Path
from typing  import Callable

from pipelines.shared.inference.metadata    import CompletedInference
from pipelines.shared.training.sweep_resume import SweepResults, SweepResumePlan
from tools.orchestration      import ExperimentStage, GpuJob
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from tools.runtime.run_tag    import RunTag


class SeedSet:

    SEED_DIR = re.compile(r"seed\d+")

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

        config      = deepcopy(self.config)
        config.seed = seeds[0]
        return self.runner_factory(config).run()


class SeedFanoutScheduler:

    SCHEDULER_FIELDS = ("gpus", "gpus_file", "poll_interval_s", "gpu", "seed", "seeds", "run_name", "infer_at_end", "resume_from")

    def __init__(self, config, cli_overrides: dict, entry_script: Path, base_name: str, run_dir: Path, infer_at_end: bool = False, infer_after: bool = False) -> None:
        self.config             = config
        self.entry_script       = Path(entry_script)
        self.base_name          = base_name
        self.infer_at_end       = infer_at_end
        self.inference_split    = config.inference.split if (infer_at_end or infer_after) else None
        self.run_dir            = Path(run_dir)
        self.resuming           = bool(config.resume_from)
        self.log_dir            = self.run_dir / "batch_train_logs"
        self.results_path       = self.log_dir / "seed_sweep_results.json"
        self.infer_results_path = self.log_dir / "seed_sweep_infer_results.json"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self.logger = Logger(log_dir=str(self.log_dir), name="seed_sweep")
        self.stage  = ExperimentStage(config=config, run_tag="seed_sweep", logger=self.logger, entry_script=self.entry_script, run_dir=self.run_dir)

    @classmethod
    def _resume_run_dir(cls, resume_from: str) -> Path:
        target = Path(resume_from).expanduser()

        if SeedSet.SEED_DIR.fullmatch(target.name):
            return target.parent

        if any(child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) for child in target.iterdir()):
            return target

    @classmethod
    def for_runner(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None, infer_at_end: bool = False, infer_after: bool = False) -> "SeedFanoutScheduler":
        if config.resume_from:
            run_dir       = cls._resume_run_dir(config.resume_from)
            base          = run_dir.name
            config.logdir = run_dir.parent
            config.resume = True
        else:
            base           = SeedSweepRunner.base_run_name(config, base_label)
            probe          = deepcopy(config)
            probe.run_name = base
            run_dir        = Path(config.logdir) / runner_factory(probe)._resolve_run_name()

        return cls(config, cli_overrides, entry_script, base_name=base, run_dir=run_dir, infer_at_end=infer_at_end, infer_after=infer_after)

    def _unit_overrides(self, run_name: str, seed: int) -> dict:
        identity = {"seed": seed, "seeds": (seed,)}

        if self.resuming:
            return {**identity, "resume_from": str(self.run_dir / f"seed{seed}")}

        return {**identity, "logdir": str(self.config.logdir), "run_name": run_name}

    def _job(self, run_name: str, seed: int) -> GpuJob:
        batched   = {"infer_after": False} if self.infer_at_end else {}
        overrides = {**self.forward_overrides, **batched, **self._unit_overrides(run_name, seed)}

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + ConfigCli.to_argv(overrides),
            log_path = self.log_dir / f"seed{seed}.log",
        )

    def _inference_job(self, run_name: str, seed: int) -> GpuJob:
        overrides = {**self.forward_overrides, "infer_after": True, "resume": True, **self._unit_overrides(run_name, seed)}

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + ConfigCli.to_argv(overrides),
            log_path = self.log_dir / f"seed{seed}_infer.log",
        )

    def _inference_pending(self, seed: int) -> bool:
        return not CompletedInference.stamps(self.run_dir / f"seed{seed}" / "inference", self.inference_split)

    def _run_inference_pass(self, units: list[tuple[int, str]], training_results: list[dict]) -> list[dict]:
        failed = {result["name"] for result in training_results if result["status"] != "DONE"}
        todo   = [(seed, run_name) for seed, run_name in units if run_name not in failed and self._inference_pending(seed)]

        self.logger.section("Batched inference")
        self.logger.kv_table({
            "Split"             : self.inference_split,
            "Planned"           : len(units),
            "Eligible"          : len(todo),
            "Inference present" : len(units) - len(todo) - len(failed),
            "Training failed"   : len(failed),
            "GPUs"              : self.config.gpus,
            "Log dir"           : str(self.log_dir),
        }, title="Configuration")

        for _, run_name in units:
            if run_name in failed:
                self.logger.warning(f"{run_name}: training failed, inference skipped")

        if not todo:
            return []

        self.logger.info(f"{len(todo)} seed(s) lack '{self.inference_split}' inference -> queued: {', '.join(run_name for _, run_name in todo)}")

        jobs    = [self._inference_job(run_name, seed) for seed, run_name in todo]
        names   = [run_name for _, run_name in todo]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(SweepResults.merge(results, self.infer_results_path, self.logger), self.infer_results_path)
        return results

    def _resume_units(self, units: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], "SweepResumePlan | None"]:
        if not self.resuming:
            return units, None

        plan = SweepResumePlan(self.run_dir, [f"seed{seed}" for seed, _ in units], split=self.inference_split).build()
        done = set(plan.completed)

        return [(seed, run_name) for seed, run_name in units if f"seed{seed}" not in done], plan

    def _record(self) -> Path:
        from pipelines.shared.training.resume_source import SweepRecording

        return SweepRecording.write(self.run_dir, self.config)

    def _warn_existing_root(self, planned: list[tuple[int, str]]) -> None:
        plan = SweepResumePlan(self.run_dir, [f"seed{seed}" for seed, _ in planned], split=self.inference_split).build()
        plan.warn_existing_root(self.logger, unit_resume=self.config.resume, trainer_resume=self.config.training.resume)

    def _train_seeds(self, units: list[tuple[int, str]]) -> list[dict]:
        if not units:
            self.logger.ok("every requested seed is already complete; no training job launched")
            return []

        jobs    = [self._job(run_name, seed) for seed, run_name in units]
        names   = [run_name for _, run_name in units]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(SweepResults.merge(results, self.results_path, self.logger), self.results_path)
        return results

    def run(self) -> None:
        with self.logger:
            seeds       = SeedSet.resolve(self.config.seeds, self.config.seed)
            planned     = [(seed, SeedSet.run_name(self.base_name, seed)) for seed in seeds]
            units, plan = self._resume_units(planned)
            recording   = self._record()

            self.logger.section("Seed fan-out")
            self.logger.kv_table({
                "Base run"        : self.base_name,
                "Run dir"         : str(self.run_dir),
                "Seeds"           : seeds,
                "GPUs"            : self.config.gpus,
                "GPU pool file"   : str(self.stage.pool_file),
                "Infer at end"    : self.infer_at_end,
                "Inference split" : self.inference_split or "none (the sweep infers nothing)",
                "Resume from"     : self.config.resume_from or "—",
                "Recording"       : str(recording),
                "Forwarded overrides (scheduler options excluded)" : self.forward_overrides or "—",
                "Log dir"         : str(self.log_dir),
            }, title="Configuration")

            if plan:
                plan.log(self.logger)
            else:
                self._warn_existing_root(planned)

            results       = self._train_seeds(units)
            infer_results = self._run_inference_pass(planned, results) if self.inference_split is not None else []

            self.logger.section("Summary")

            if results:
                rows = [{"Seed": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in results]
                self.logger.metrics_table(rows, columns=["Seed", "Status", "Duration"])

            if infer_results:
                self.logger.subsection("Batched inference")
                infer_rows = [{"Seed": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in infer_results]
                self.logger.metrics_table(infer_rows, columns=["Seed", "Status", "Duration"])

            failed       = [r for r in results if r["status"] != "DONE"]
            infer_failed = [r for r in infer_results if r["status"] != "DONE"]
            self.stage._log_failures(failed + infer_failed)

            problems = []
            if failed:
                problems.append(f"{len(failed)} of {len(results)} seed runs failed")
            if infer_failed:
                problems.append(f"{len(infer_failed)} of {len(infer_results)} inference jobs failed")

            if problems:
                raise SystemExit("; ".join(problems) + f"; see {self.log_dir}")
