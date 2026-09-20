"""Entry-agnostic scheduler running a planned trial sweep over the GPU pool.

Holds the shared machinery every trial fan-out uses: seed expansion of the
planned units, sweep-root and resume resolution, per-unit argv construction,
GPU-queue submission and the summary reporting. Entries subclass it with their
mode table, their planner dispatch and their model key.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.inference.metadata       import CompletedInference
from pipelines.shared.training.ddp_fanout      import DdpDispatch
from pipelines.shared.training.seed_sweep      import SeedSet
from pipelines.shared.training.sweep_resume    import SweepResumePlan, SweepResumeRoot
from tools.orchestration                       import ExperimentStage, GpuJob
from tools.monitoring.logger                   import Logger
from tools.runtime.config_cli                  import ConfigCli


class TrialSweepScheduler:
    """Expands a trial mode into per-unit GPU jobs and runs them as a sweep.

    Every trial mode maps to a planner and a run subdirectory; the resulting
    units are fanned out over the GPU pool, with inference either folded into
    each unit or batched after all training finishes when the entry stages it.

    Attributes:
        config             : Resolved entry config of the training entry.
        entry_script       : Path to the training entry script the jobs re-invoke.
        resume_plan        : Plan of already-completed units when resuming, else None.
        runs_root          : Directory holding all run directories of this sweep.
        log_dir            : Directory for the scheduler log and result JSONs.
        results_path       : JSON file with the training job results.
        infer_results_path : JSON file with the batched inference job results.
        forward_overrides  : CLI overrides forwarded to each unit, scheduler-only fields removed.
    """

    SCHEDULER_FIELDS = ()

    MODE_SUBDIRS = {}

    def __init__(self, config, cli_overrides: dict, entry_script: Path) -> None:
        """Resolves the sweep root, the resume plan and the overrides forwarded to units.

        Args:
            config        : Resolved entry config naming the trial mode.
            cli_overrides : Override paths and values parsed from the command line.
            entry_script  : Path to the training entry script each unit re-invokes.

        Raises:
            ValueError : If `config.trials_mode` is not a known mode, or if `ddp` is set, which a trial sweep cannot combine with.
        """
        DdpDispatch.reject_trials(config, trial_unit=False)

        self.config       = config
        self.entry_script = Path(entry_script)
        self._planner     = None

        if config.trials_mode not in self.MODE_SUBDIRS:
            raise ValueError(f"Unknown trials_mode '{config.trials_mode}', expected one of {sorted(self.MODE_SUBDIRS)}")

        self.resume_plan = None
        self.runs_root   = Path(config.logdir) / self.MODE_SUBDIRS[config.trials_mode]

        if config.resume_from:
            names            = [run_name for run_name, _ in self.units()]
            self.runs_root   = SweepResumeRoot.resolve(config.resume_from, names)
            self.resume_plan = SweepResumePlan(self.runs_root, names).build()
            config.resume    = True

        self.log_dir            = self.runs_root / "batch_train_logs"
        self.results_path       = self.log_dir / "train_scheduler_results.json"
        self.infer_results_path = self.log_dir / "infer_scheduler_results.json"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self._logger  = None
        self._stage   = None

    @property
    def entry_label(self) -> str:
        """The training family owning the sweep, read off the entry script name."""
        return self.entry_script.stem.removeprefix("train_")

    @property
    def sweep_name(self) -> str:
        """The cluster sweep name, qualifying the trial mode with the entry that owns it."""
        return f"sweep_{self.entry_label}_{self.config.trials_mode}"

    @property
    def entry_flags(self) -> tuple:
        """The flags marking a re-invocation of the entry script as a single trial."""
        return ("--trial",)

    @property
    def staged_inference(self) -> bool:
        """Whether inference runs as a separate stage after all training units."""
        return False

    def setup_argv(self) -> list[str] | None:
        """Returns the preparation-stage argv, None because this sweep needs none."""
        return None

    def finalize_argv(self) -> list[str] | None:
        """Returns the finalization-stage argv, None because a trial sweep stages no compare job.

        The mode report is written by `run`, which a cluster sweep never
        executes; re-running the launch locally against the finished sweep root
        writes it, since a fully resumed sweep still reports.
        """
        return None

    @property
    def logger(self) -> Logger:
        """The scheduler logger, created on first use."""
        if self._logger is None:
            self._logger = Logger(log_dir=str(self.log_dir), name="train_scheduler")

        return self._logger

    @property
    def stage(self) -> ExperimentStage:
        """The GPU-queue experiment stage backing this sweep, created on first use."""
        if self._stage is None:
            self._stage = ExperimentStage(config=self.config, run_tag="batch_train", logger=self.logger, entry_script=self.entry_script, run_dir=self.runs_root)

        return self._stage

    def planner(self):
        """Returns the trial planner for the configured mode, built on first use."""
        if self._planner is None:
            self._planner = self._build_planner()

        return self._planner

    def _build_planner(self):
        """Builds the planner matching the configured trial mode.

        Raises:
            NotImplementedError : Always; every entry scheduler dispatches its own planner set.
        """
        raise NotImplementedError("Entry schedulers must map their trial modes to planners")

    def _model_key(self) -> str:
        """Returns the model label of the base configuration.

        Raises:
            NotImplementedError : Always; every entry scheduler names its own base model.
        """
        raise NotImplementedError("Entry schedulers must name their base model")

    def _seed_units(self, experiments: list[tuple[str, dict]], seeds: list[int]) -> list[tuple[str, dict]]:
        """Expands each planned experiment into one unit per seed.

        Args:
            experiments : Planned `(run_name, overrides)` pairs.
            seeds       : Seeds to train each experiment at.

        Returns:
            The experiments unchanged for a single seed, otherwise one
            seed-suffixed unit per experiment and seed.
        """
        if len(seeds) == 1:
            return list(experiments)

        return [(SeedSet.run_name(base, seed), {**overrides, "seed": seed, "seeds": (seed,)}) for base, overrides in experiments for seed in seeds]

    def planned_units(self) -> list[tuple[str, dict]]:
        """Returns every seed-expanded unit of the sweep, including the ones a resume marks complete."""
        seeds = SeedSet.resolve(self.config.seeds, self.config.seed)

        return self._seed_units(self.planner().plan(), seeds)

    def units(self) -> list[tuple[str, dict]]:
        """Returns the seed-expanded units still to train, dropping those the resume plan marks complete."""
        units = self.planned_units()

        if self.resume_plan is None:
            return units

        done = set(self.resume_plan.completed)
        return [(run_name, overrides) for run_name, overrides in units if run_name not in done]

    def trial_argv(self, run_name: str, overrides: dict, batched_inference: bool = False) -> list[str]:
        """Builds the command-line arguments training one unit of the sweep.

        Args:
            run_name          : Name of the unit's run directory.
            overrides         : Per-unit config overrides from the planner.
            batched_inference : Whether the caller submits the inference pass separately, as staged inference requires.

        Returns:
            The argv list for the entry script.

        Raises:
            ValueError : If inference is staged but the caller did not declare that it batches inference itself.
        """
        if self.staged_inference and not batched_inference:
            raise ValueError(f"infer_at_end batches inference after every training unit finishes, so '{run_name}' is built with infer_after disabled; the caller must submit the inference pass itself via inference_argv(), or set infer_after instead to run inference inside each unit")

        batched = {"infer_after": False} if self.staged_inference else {}

        return ConfigCli.to_argv({**self.forward_overrides, **overrides, **batched, **self._unit_overrides(run_name)})

    def _unit_overrides(self, run_name: str) -> dict:
        """Returns the overrides placing one unit in its directory.

        A unit whose run directory the resume plan found is handed that
        directory through ``resume_from``, whatever the current naming scheme
        would call it; any other unit is named through the runs root and its
        run name, which the entry's naming scheme composes into a fresh
        directory.

        Args:
            run_name : Name of the unit.

        Raises:
            SystemExit : If several run directories under the root match the unit, which leaves it ambiguous.
        """
        if self.resume_plan is None or run_name in self.resume_plan.missing:
            return {"run_name": run_name, "logdir": str(self.runs_root)}

        hits = SweepResumeRoot.matches(self.runs_root, run_name)
        if len(hits) != 1:
            raise SystemExit(f"unit '{run_name}' matches {len(hits)} run directories under {self.runs_root}: {[str(hit) for hit in hits]}; remove or move the extra ones before resuming")

        return {"resume_from": str(hits[0])}

    def inference_argv(self, run_name: str, overrides: dict) -> list[str]:
        """Returns the argv running inference over an already-trained unit."""
        return ConfigCli.to_argv({**self.forward_overrides, **overrides, "infer_after": True, "resume": True, **self._unit_overrides(run_name)})

    def _job(self, run_name: str, overrides: dict) -> GpuJob:
        """Returns the GPU job training one unit."""
        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + self.trial_argv(run_name, overrides, batched_inference=True),
            log_path = self.log_dir / f"{run_name}.log",
        )

    def _train_units(self, units: list[tuple[str, dict]]) -> list[dict]:
        """Trains the pending units over the GPU pool and writes their results.

        Args:
            units : The `(run_name, overrides)` pairs still to train.

        Returns:
            Per-unit training job results, empty when a resume left nothing to train.
        """
        if not units:
            self.logger.ok("every planned unit is already trained; no training job launched")
            return []

        jobs    = [self._job(run_name, overrides) for run_name, overrides in units]
        names   = [run_name for run_name, overrides in units]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.results_path)
        return results

    def _inference_job(self, run_name: str, overrides: dict) -> GpuJob:
        """Returns the GPU job running inference over one trained unit."""
        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + self.inference_argv(run_name, overrides),
            log_path = self.log_dir / f"{run_name}_infer.log",
        )

    def _inference_pending(self, run_name: str) -> bool:
        """Returns whether a planned unit still lacks the inference output of the configured split.

        Args:
            run_name : Name of the unit's run directory.

        Returns:
            True when no run directory of the unit carries a completed
            inference stamp, which covers a unit just trained with per-unit
            inference disabled and a reused unit whose staged pass never ran.
        """
        if not self.runs_root.is_dir():
            return True

        split = self.config.inference.split

        return not any(CompletedInference.stamps(directory / "inference", split) for directory in SweepResumeRoot.matches(self.runs_root, run_name))

    def _run_inference_pass(self, units: list[tuple[str, dict]], training_results: list[dict]) -> list[dict]:
        """Runs the batched inference stage over every planned unit still missing its inference output.

        Args:
            units            : The `(run_name, overrides)` pairs of every planned unit, reused ones included.
            training_results : Per-unit training job results of this launch.

        Returns:
            The inference job results, empty when every unit either failed
            training or already carries its inference output.
        """
        failed = {result["name"] for result in training_results if result["status"] != "DONE"}
        todo   = [(run_name, overrides) for run_name, overrides in units if run_name not in failed and self._inference_pending(run_name)]

        self.logger.section("Batched inference")
        self.logger.kv_table({
            "Planned"           : len(units),
            "Eligible"          : len(todo),
            "Inference present" : len(units) - len(todo) - len(failed),
            "Training failed"   : len(failed),
            "GPUs"              : self.config.gpus,
            "Log dir"           : str(self.log_dir),
        }, title="Configuration")

        for run_name, _ in units:
            if run_name in failed:
                self.logger.warning(f"{run_name}: training failed, inference skipped")

        if not todo:
            return []

        jobs    = [self._inference_job(run_name, overrides) for run_name, overrides in todo]
        names   = [run_name for run_name, _ in todo]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(results, self.infer_results_path)
        return results

    def _inference_rows(self) -> dict:
        """Returns the inference-related configuration rows, empty for entries without an inference pass."""
        return {}

    def _write_mode_report(self) -> None:
        """Writes the mode-specific sweep report, a no-op unless an entry defines one."""

    def _record(self) -> Path:
        """Records the sweep's effective configuration under its log directory, for a later resume to load."""
        from pipelines.shared.training.resume_source import SweepRecording

        return SweepRecording.write(self.runs_root, self.config)

    def run(self) -> None:
        """Runs the whole sweep: plan units, train the pending ones, infer over those still missing it, then write the mode report.

        Raises:
            SystemExit : If any training or inference job did not finish DONE.
        """
        planner     = self.planner()
        experiments = planner.plan()
        seeds       = SeedSet.resolve(self.config.seeds, self.config.seed)
        planned     = self.planned_units()
        units       = self.units()
        recording   = self._record()

        self.logger.section(f"Training trials: {self.config.trials_mode}")
        self.logger.kv_table({
            "Model (base config)"  : f"{self._model_key()} (per-trial settings may override the model)",
            "Mode"                 : self.config.trials_mode,
            **planner.summary(),
            "Trials"               : len(experiments),
            "Seeds"                : seeds,
            "Planned units"        : len(planned),
            "GPU jobs"             : len(units),
            "GPUs"                 : self.config.gpus,
            "GPU pool file"        : str(self.stage.pool_file),
            **self._inference_rows(),
            "Unit resume"          : f"{self.config.resume} (reuse completed run dirs, delete unfinished ones)",
            "Trainer resume"       : self.config.training.resume,
            "Resume from"          : self.config.resume_from or "—",
            "Recording"            : str(recording),
            "Forwarded overrides (scheduler options excluded; per-trial settings win)" : self.forward_overrides or "—",
            "Log dir"              : str(self.log_dir),
        }, title="Configuration")

        if self.resume_plan:
            self.resume_plan.log(self.logger)

        results       = self._train_units(units)
        infer_results = self._run_inference_pass(planned, results) if self.staged_inference else []

        self._write_mode_report()

        self.logger.section("Summary")
        rows = [{"Trial": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in results]
        self.logger.metrics_table(rows, columns=["Trial", "Status", "Duration"])

        if infer_results:
            self.logger.subsection("Batched inference")
            infer_rows = [{"Trial": r["name"], "Status": r["status"], "Duration": f"{r['duration_s'] / 60:.1f} min"} for r in infer_results]
            self.logger.metrics_table(infer_rows, columns=["Trial", "Status", "Duration"])

        failed       = [r for r in results if r["status"] != "DONE"]
        infer_failed = [r for r in infer_results if r["status"] != "DONE"]
        self.stage._log_failures(failed + infer_failed)

        self.logger.close()

        problems = []
        if failed:
            problems.append(f"{len(failed)} of {len(results)} training trials failed")
        if infer_failed:
            problems.append(f"{len(infer_failed)} of {len(infer_results)} inference jobs failed")

        if problems:
            raise SystemExit("; ".join(problems) + f"; see {self.log_dir}")
