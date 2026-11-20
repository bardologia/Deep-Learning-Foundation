
from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.inference.metadata       import CompletedInference
from pipelines.shared.training.seed_sweep      import SeedSet
from pipelines.shared.training.sweep_resume    import SweepResults, SweepResumePlan, SweepResumeRoot
from tools.orchestration                       import ExperimentStage, GpuJob
from tools.monitoring.logger                   import Logger
from tools.runtime.config_cli                  import ConfigCli


class TrialSweepScheduler:

    SCHEDULER_FIELDS = ()

    MODE_SUBDIRS = {}

    def __init__(self, config, cli_overrides: dict, entry_script: Path) -> None:
        self.config       = config
        self.entry_script = Path(entry_script)
        self._planner     = None

        self.resume_plan = None
        self.runs_root   = Path(config.logdir) / self.MODE_SUBDIRS[config.trials_mode]

        if config.resume_from:
            names            = [run_name for run_name, _ in self.units()]
            self.runs_root   = SweepResumeRoot.resolve(config.resume_from, names)
            self.resume_plan = SweepResumePlan(self.runs_root, names, split=self.inference_split).build()
            config.resume    = True

        self.log_dir            = self.runs_root / "batch_train_logs"
        self.results_path       = self.log_dir / "train_scheduler_results.json"
        self.infer_results_path = self.log_dir / "infer_scheduler_results.json"

        self.forward_overrides = {path: value for path, value in cli_overrides.items() if path.split(".")[0] not in self.SCHEDULER_FIELDS}

        self._logger  = None
        self._stage   = None

    @property
    def staged_inference(self) -> bool:
        return False

    @property
    def unit_inference(self) -> bool:
        return False

    @property
    def inference_split(self) -> str | None:
        return self.config.inference.split if self.staged_inference or self.unit_inference else None

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = Logger(log_dir=str(self.log_dir), name="train_scheduler")

        return self._logger

    @property
    def stage(self) -> ExperimentStage:
        if self._stage is None:
            self._stage = ExperimentStage(config=self.config, run_tag="batch_train", logger=self.logger, entry_script=self.entry_script, run_dir=self.runs_root)

        return self._stage

    def planner(self):
        if self._planner is None:
            self._planner = self._build_planner()

        return self._planner

    def _build_planner(self):
        raise NotImplementedError("Entry schedulers must map their trial modes to planners")

    def _model_key(self) -> str:
        raise NotImplementedError("Entry schedulers must name their base model")

    def _seed_units(self, experiments: list[tuple[str, dict]], seeds: list[int]) -> list[tuple[str, dict]]:
        if len(seeds) == 1:
            return list(experiments)

        return [(SeedSet.run_name(base, seed), {**overrides, "seed": seed, "seeds": (seed,)}) for base, overrides in experiments for seed in seeds]

    def planned_units(self) -> list[tuple[str, dict]]:
        seeds = SeedSet.resolve(self.config.seeds, self.config.seed)

        return self._seed_units(self.planner().plan(), seeds)

    def units(self) -> list[tuple[str, dict]]:
        units = self.planned_units()

        if self.resume_plan is None:
            return units

        done = set(self.resume_plan.completed)
        return [(run_name, overrides) for run_name, overrides in units if run_name not in done]

    def trial_argv(self, run_name: str, overrides: dict, batched_inference: bool = False) -> list[str]:
        batched = {"infer_after": False} if self.staged_inference else {}

        return ConfigCli.to_argv({**self.forward_overrides, **overrides, **batched, **self._unit_overrides(run_name)})

    def _unit_overrides(self, run_name: str) -> dict:
        if self.resume_plan is None or run_name in self.resume_plan.missing:
            return {"run_name": run_name, "logdir": str(self.runs_root)}

        hits = SweepResumeRoot.matches(self.runs_root, run_name)

        return {"resume_from": str(hits[0])}

    def inference_argv(self, run_name: str, overrides: dict) -> list[str]:
        return ConfigCli.to_argv({**self.forward_overrides, **overrides, "infer_after": True, "resume": True, **self._unit_overrides(run_name)})

    def _job(self, run_name: str, overrides: dict) -> GpuJob:
        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + self.trial_argv(run_name, overrides, batched_inference=True),
            log_path = self.log_dir / f"{run_name}.log",
        )

    def _train_units(self, units: list[tuple[str, dict]]) -> list[dict]:
        if not units:
            self.logger.ok("every planned unit is already trained; no training job launched")
            return []

        jobs    = [self._job(run_name, overrides) for run_name, overrides in units]
        names   = [run_name for run_name, overrides in units]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(SweepResults.merge(results, self.results_path, self.logger), self.results_path)
        return results

    def _inference_job(self, run_name: str, overrides: dict) -> GpuJob:
        return GpuJob(
            name     = run_name,
            command  = [sys.executable, str(self.entry_script), "--trial"] + self.inference_argv(run_name, overrides),
            log_path = self.log_dir / f"{run_name}_infer.log",
        )

    def _inference_pending(self, run_name: str) -> bool:
        if not self.runs_root.is_dir():
            return True

        return not any(CompletedInference.stamps(directory / "inference", self.inference_split) for directory in SweepResumeRoot.matches(self.runs_root, run_name))

    def _run_inference_pass(self, units: list[tuple[str, dict]], training_results: list[dict]) -> list[dict]:
        failed = {result["name"] for result in training_results if result["status"] != "DONE"}
        todo   = [(run_name, overrides) for run_name, overrides in units if run_name not in failed and self._inference_pending(run_name)]

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

        for run_name, _ in units:
            if run_name in failed:
                self.logger.warning(f"{run_name}: training failed, inference skipped")

        if not todo:
            return []

        self.logger.info(f"{len(todo)} unit(s) lack '{self.inference_split}' inference -> queued: {', '.join(run_name for run_name, _ in todo)}")

        jobs    = [self._inference_job(run_name, overrides) for run_name, overrides in todo]
        names   = [run_name for run_name, _ in todo]
        results = self.stage._order_results(self.stage._run_queue(jobs), names)

        self.stage._write_results(SweepResults.merge(results, self.infer_results_path, self.logger), self.infer_results_path)
        return results

    def _inference_rows(self) -> dict:
        return {}

    def _resume_rows(self) -> dict:
        unit    = "completed run dirs are reused" if self.config.resume else "completed run dirs are retrained in place"
        trainer = "per unit: continues from last.pt where it exists, restarts from scratch otherwise (see Sweep resume)" if self.resume_plan else f"{self.config.training.resume} ({'unfinished run dirs continue from last.pt' if self.config.training.resume else 'unfinished run dirs are deleted and restart'})"

        return {
            "Unit resume"    : f"{self.config.resume} ({unit})",
            "Trainer resume" : trainer,
        }

    def _write_mode_report(self) -> None:
        pass

    def _record(self) -> Path:
        from pipelines.shared.training.resume_source import SweepRecording

        return SweepRecording.write(self.runs_root, self.config)

    def _warn_existing_root(self, planned: list[tuple[str, dict]]) -> None:
        if not self.runs_root.is_dir():
            return

        plan = SweepResumePlan(self.runs_root, [run_name for run_name, _ in planned], split=self.inference_split).build()
        plan.warn_existing_root(self.logger, unit_resume=self.config.resume, trainer_resume=self.config.training.resume)

    def run(self) -> None:
        with self.logger:
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
                "Inference split"      : self.inference_split or "none (the sweep infers nothing)",
                **self._resume_rows(),
                "Resume from"          : self.config.resume_from or "—",
                "Recording"            : str(recording),
                "Forwarded overrides (scheduler options excluded; per-trial settings win)" : self.forward_overrides or "—",
                "Log dir"              : str(self.log_dir),
            }, title="Configuration")

            if self.resume_plan:
                self.resume_plan.log(self.logger)
            else:
                self._warn_existing_root(planned)

            results       = self._train_units(units)
            infer_results = self._run_inference_pass(planned, results) if self.inference_split is not None else []

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

            problems = []
            if failed:
                problems.append(f"{len(failed)} of {len(results)} training trials failed")
            if infer_failed:
                problems.append(f"{len(infer_failed)} of {len(infer_results)} inference jobs failed")

            if problems:
                raise SystemExit("; ".join(problems) + f"; see {self.log_dir}")
