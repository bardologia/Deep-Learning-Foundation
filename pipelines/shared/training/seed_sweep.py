"""Seed-set naming and multi-seed fan-out across the local GPU pool.

A single-seed request trains in process; a multi-seed request is fanned out as
one subprocess per seed onto the GPU queue, optionally followed by a batched
inference pass. A resumed sweep reconciles the existing seed directories
against the plan before launching anything and hands every seed job its exact
directory, so the sweep's naming scheme never has to be reproduced.
"""

from __future__ import annotations

import re
import sys
from copy    import deepcopy
from pathlib import Path
from typing  import Callable

from pipelines.shared.training.sweep_resume import SweepResumePlan
from tools.orchestration      import ExperimentStage, GpuJob
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from tools.runtime.run_tag    import RunTag


class SeedSet:
    """Naming conventions for the seed dimension of a sweep.

    Attributes:
        SEED_DIR : Pattern matching a ``seed<N>`` directory name.
    """

    SEED_DIR = re.compile(r"seed\d+")

    @staticmethod
    def resolve(seeds, default_seed: int) -> list[int]:
        """Returns the explicit seed list, falling back to the single default seed."""
        return list(seeds or []) or [default_seed]

    @staticmethod
    def run_name(base: str, seed: int) -> str:
        """Returns the ``<base>/seed<N>`` run name of one seed."""
        return f"{base}/seed{seed}"

    @staticmethod
    def base(run_name: str) -> str:
        """Returns the sweep base name of a ``<base>/seed<N>`` run name."""
        return run_name.split("/seed")[0]

    @staticmethod
    def cli_args(seed: int | None) -> list[str]:
        """Returns the ``--seed`` CLI arguments, empty when no seed is pinned."""
        return ["--seed", str(seed)] if seed is not None else []

    @staticmethod
    def units(bases, seeds) -> list[tuple[str, int | None, str]]:
        """Expands base names across the seed set into sweep units.

        Args:
            bases : Base run names to expand.
            seeds : Seeds to fan out over; empty means no seed dimension.

        Returns:
            List of (base, seed, run name) triples, with seed None and the run
            name equal to the base when no seed is requested.
        """
        seeds = list(seeds or [])

        if not seeds:
            return [(base, None, base) for base in bases]

        return [(base, seed, SeedSet.run_name(base, seed)) for base in bases for seed in seeds]


class SeedSweepRunner:
    """Trains the single requested seed in the current process.

    Attributes:
        config         : Entry configuration of the run.
        runner_factory : Callable turning a config into a single-run runner.
    """

    def __init__(self, config, runner_factory: Callable[[object], object]) -> None:
        """Stores the entry config and the single-run runner factory."""
        self.config         = config
        self.runner_factory = runner_factory

    @staticmethod
    def base_run_name(config, base_label: str | None = None) -> str:
        """Returns the sweep base name, from the config or a timestamp.

        Args:
            config     : Entry configuration, whose ``run_name`` wins when set.
            base_label : Optional label prefixed to the timestamp.

        Returns:
            The base run name.
        """
        if config.run_name:
            return config.run_name

        timestamp = RunTag.now()
        return f"{base_label}_{timestamp}" if base_label else timestamp

    def run(self):
        """Trains the single seed and returns whatever the run runner returns.

        Raises:
            ValueError : If more than one seed is requested, which belongs to SeedFanoutScheduler.
        """
        seeds = SeedSet.resolve(self.config.seeds, self.config.seed)

        if len(seeds) > 1:
            raise ValueError(f"SeedSweepRunner trains exactly one seed, got {seeds}; multi-seed runs fan out across the GPU pool via SeedFanoutScheduler")

        config      = deepcopy(self.config)
        config.seed = seeds[0]
        return self.runner_factory(config).run()


class SeedFanoutScheduler:
    """Fans a multi-seed run out as one subprocess per seed on the GPU pool.

    Attributes:
        SCHEDULER_FIELDS  : Top-level config fields owned by the scheduler and therefore never forwarded to the per-seed jobs.
        config            : Entry configuration of the sweep.
        entry_script      : Entry script each seed job re-invokes.
        base_name         : Sweep base run name.
        infer_at_end      : Whether inference is batched after all training finishes.
        run_dir           : Sweep root directory holding the per-seed run directories.
        resuming          : Whether the sweep continues an existing root, in which case every seed job is handed its own directory.
        log_dir           : Directory holding the per-seed logs, the result JSONs and the sweep recording.
        forward_overrides : CLI overrides forwarded to each seed job.
        logger            : Logger of the sweep.
        stage             : Experiment stage owning the GPU queue and result bookkeeping.
    """

    SCHEDULER_FIELDS = ("gpus", "gpus_file", "poll_interval_s", "gpu", "seed", "seeds", "run_name", "infer_at_end", "resume_from")

    def __init__(self, config, cli_overrides: dict, entry_script: Path, base_name: str, run_dir: Path, infer_at_end: bool = False) -> None:
        """Prepares the sweep directories, forwarded overrides and GPU stage.

        Args:
            config        : Entry configuration of the sweep.
            cli_overrides : All CLI overrides; scheduler-owned fields are filtered out before forwarding.
            entry_script  : Entry script each seed job re-invokes.
            base_name     : Sweep base run name.
            run_dir       : Sweep root directory.
            infer_at_end  : Whether to batch inference after training finishes.
        """
        self.config             = config
        self.entry_script       = Path(entry_script)
        self.base_name          = base_name
        self.infer_at_end       = infer_at_end
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
        """Resolves the pasted resume path to the sweep root directory.

        Args:
            resume_from : Path of the sweep root or of one of its seed runs.

        Returns:
            The sweep root directory holding the ``seed<N>`` runs.

        Raises:
            FileNotFoundError : If the path does not exist.
            SystemExit        : If the path is neither a seed run nor a sweep root.
        """
        target = Path(resume_from).expanduser()

        if not target.is_dir():
            raise FileNotFoundError(f"resume_from points at '{target}', which does not exist; paste the sweep base directory holding the seed runs, or one of its seed directories")

        if SeedSet.SEED_DIR.fullmatch(target.name):
            return target.parent

        if any(child.is_dir() and SeedSet.SEED_DIR.fullmatch(child.name) for child in target.iterdir()):
            return target

        raise SystemExit(f"resume_from '{target}' is neither a seed run directory nor a sweep base directory holding seed runs; paste the base directory whose children are seed<N> runs, or one of those seed directories")

    @classmethod
    def for_runner(cls, config, cli_overrides: dict, entry_script: Path, runner_factory: Callable[[object], object], base_label: str | None = None, infer_at_end: bool = False) -> "SeedFanoutScheduler":
        """Builds the scheduler for an entry, resuming or naming a fresh sweep.

        A resumed sweep keeps the pasted root as it is, whatever it is called:
        the seed jobs are handed their directories directly, so the root name
        never has to be recomposed from the current settings.

        Args:
            config         : Entry configuration of the sweep.
            cli_overrides  : All CLI overrides seen by the entry.
            entry_script   : Entry script each seed job re-invokes.
            runner_factory : Callable turning a config into a single-run runner, used to compose run names.
            base_label     : Optional label prefixed to the timestamped base name.
            infer_at_end   : Whether to batch inference after training finishes.

        Returns:
            The configured scheduler.
        """
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

        return cls(config, cli_overrides, entry_script, base_name=base, run_dir=run_dir, infer_at_end=infer_at_end)

    def _unit_overrides(self, run_name: str, seed: int) -> dict:
        """Returns the overrides pinning one seed job's identity and directory.

        A resumed sweep hands the job its exact directory through
        ``resume_from``; a fresh sweep names it through the log directory and
        the nested run name its naming scheme composes back into the root.

        Args:
            run_name : ``<base>/seed<N>`` run name of the seed.
            seed     : Seed value pinned in the job's overrides.
        """
        identity = {"seed": seed, "seeds": (seed,)}

        if self.resuming:
            return {**identity, "resume_from": str(self.run_dir / f"seed{seed}")}

        return {**identity, "logdir": str(self.config.logdir), "run_name": run_name}

    def _job(self, run_name: str, seed: int) -> GpuJob:
        """Builds the training job for one seed.

        The job carries the ``--trial`` flag, which marks it as one unit of a
        sweep: it trains the single seed in place and, when handed its
        directory, reads that directory as a run rather than as a sweep.

        Args:
            run_name : ``<base>/seed<N>`` run name of the seed.
            seed     : Seed value pinned in the job's overrides.

        Returns:
            The GPU job re-invoking the entry script for this seed.
        """
        batched   = {"infer_after": False} if self.infer_at_end else {}
        overrides = {**self.forward_overrides, **batched, **self._unit_overrides(run_name, seed)}

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + ConfigCli.to_argv(overrides),
            log_path = self.log_dir / f"seed{seed}.log",
        )

    def _inference_job(self, run_name: str, seed: int) -> GpuJob:
        """Builds the batched inference job for one already-trained seed.

        Args:
            run_name : ``<base>/seed<N>`` run name of the seed.
            seed     : Seed value pinned in the job's overrides.

        Returns:
            The GPU job re-invoking the entry script with inference enabled.
        """
        overrides = {**self.forward_overrides, "infer_after": True, "resume": True, **self._unit_overrides(run_name, seed)}

        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + ConfigCli.to_argv(overrides),
            log_path = self.log_dir / f"seed{seed}_infer.log",
        )

    def _run_inference_pass(self, units: list[tuple[int, str]], training_results: list[dict]) -> list[dict]:
        """Runs inference for every seed whose training finished.

        Args:
            units            : (seed, run name) pairs that were launched for training.
            training_results : Per-job training result records.

        Returns:
            The inference result records, empty when no seed is eligible.
        """
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

    def _resume_units(self, units: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], "SweepResumePlan | None"]:
        """Drops completed seeds when resuming.

        Args:
            units : (seed, run name) pairs planned for the sweep.

        Returns:
            Tuple of the units still to launch and the reconciliation plan, or
            the units unchanged and None when not resuming.
        """
        if not self.resuming:
            return units, None

        plan = SweepResumePlan(self.run_dir, [f"seed{seed}" for seed, _ in units]).build()
        done = set(plan.completed)

        return [(seed, run_name) for seed, run_name in units if f"seed{seed}" not in done], plan

    def _record(self) -> Path:
        """Records the sweep's effective configuration under its log directory, for a later resume to load."""
        from pipelines.shared.training.resume_source import SweepRecording

        return SweepRecording.write(self.run_dir, self.config)

    def run(self) -> None:
        """Launches every outstanding seed, then optionally the inference pass.

        Writes the training and inference result JSONs, logs the per-seed
        summary tables and reports failures.

        Raises:
            SystemExit : If any training or inference job did not finish DONE.
        """
        seeds       = SeedSet.resolve(self.config.seeds, self.config.seed)
        units, plan = self._resume_units([(seed, SeedSet.run_name(self.base_name, seed)) for seed in seeds])
        recording   = self._record()

        self.logger.section("Seed fan-out")
        self.logger.kv_table({
            "Base run"      : self.base_name,
            "Run dir"       : str(self.run_dir),
            "Seeds"         : seeds,
            "GPUs"          : self.config.gpus,
            "GPU pool file" : str(self.stage.pool_file),
            "Infer at end"  : self.infer_at_end,
            "Resume from"   : self.config.resume_from or "—",
            "Recording"     : str(recording),
            "Forwarded overrides (scheduler options excluded)" : self.forward_overrides or "—",
            "Log dir"       : str(self.log_dir),
        }, title="Configuration")

        if plan:
            plan.log(self.logger)

        if not units:
            self.logger.ok("every requested seed is already complete; nothing to launch")
            self.logger.close()
            return

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
