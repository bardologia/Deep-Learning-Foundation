
from __future__ import annotations

import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState

from configuration.tuning                       import TuningConfig
from models                                     import BACKBONE_CONFIG_REGISTRY, BACKBONE_HEADS
from pipelines.shared.model.model_builder       import ModelBuilder
from pipelines.shared.orchestration.resumed_run import TuningResumedRun
from pipelines.tuning.plots                     import StudyPlotter
from pipelines.tuning.storage                   import TuningStorage
from pipelines.tuning.tuners                    import BestConfigWriter, SearchSpace, StudyProgress, StudyStopRules, StudyTargetStop
from tools.data.io                              import FileIO
from tools.orchestration.gpu_queue              import GpuJob, GpuPoolFile, GpuQueue
from tools.monitoring.logger                    import Logger
from tools.runtime.config_cli                   import ConfigCli
from tools.runtime.run_tag                      import RunTag


class TuningScheduler:

    STUDY_KEYS      = ("heads", "skip_models", "tuning", "curriculum", "paths.dataset_path", "paths.parameters_path", "paths.secondary_labels")
    MUTABLE_KEYS    = ("tuning.n_trials", "tuning.plateau_patience", "tuning.plateau_min_delta", "tuning.time_budget_h")
    LAUNCH_KEYS     = (*TuningResumedRun.LAUNCH_FIELDS, *TuningResumedRun.LOCATION_FIELDS, "n_workers")
    HISTORY_DIR     = "config_history"
    RECORD_KEYS     = ("model", "status", "trials_completed", "trials_pruned", "trials_failed", "trials_running", "target", "stop_reason", "val_loss", "best_config")
    STAGES          = ("full", "prepare", "compare")

    def __init__(self, tag: str, config, entry_script: Path, resumed=None) -> None:
        self.tag          = tag
        self.config       = config
        self.entry_script = Path(entry_script)
        self.resumed      = resumed
        self.run_dir      = Path(config.paths.log_base_dir) / tag
        self.storage_url  = f"sqlite:///{self.run_dir / 'optuna.db'}"
        self.summary_path = self.run_dir / "tuning_results.json"
        self.db_path      = self.run_dir / "optuna.db"

        self.logger   = None
        self.results  = []
        self.prepared = {"created": [], "loaded": []}
        self.reported = []

        self._quiet_optuna()

    def _quiet_optuna(self) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _defines_study(self, key: str) -> bool:
        if key in self.MUTABLE_KEYS:
            return False

        return any(key == prefix or key.startswith(f"{prefix}.") for prefix in self.STUDY_KEYS)

    def _guard_resumed_config(self, resolved_path: Path) -> dict:
        stored  = FileIO.load_json(resolved_path)
        current = ConfigCli.to_mapping(self.config)
        changed = {key: (stored[key], value) for key, value in current.items() if self._defines_study(key) and stored[key] != value}

        if changed:
            sys.exit(f"ERROR: --resume cannot change the fields that define the study '{self.tag}'; trials already stored were optimised under {', '.join(f'{key}={old!r} (now {new!r})' for key, (old, new) in sorted(changed.items()))}; start a new run tag instead")

        return {key: (stored[key], value) for key, value in sorted(current.items()) if key in stored and stored[key] != value and not self._defines_study(key) and key not in self.MUTABLE_KEYS and key not in self.LAUNCH_KEYS}

    def _archive_previous_config(self, resolved_path: Path) -> Path | None:
        if not resolved_path.is_file() or FileIO.load_json(resolved_path) == ConfigCli.to_mapping(self.config):
            return None

        archived = self.run_dir / self.HISTORY_DIR / f"{RunTag.now()}.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.replace(archived)

        return archived

    def _distribute_trials(self, total: int, n_workers: int) -> list[int]:
        base  = total // n_workers
        extra = total % n_workers
        return [base + (1 if i < extra else 0) for i in range(n_workers)]

    def worker_count(self) -> int:
        return self.config.n_workers or max(len(self.config.gpus), 1)

    def _study_name(self, model_name: str) -> str:
        return f"{model_name}_{self.tag}"

    def _model_keys(self) -> list[str]:
        unknown = [head for head in self.config.heads if head not in BACKBONE_HEADS]
        if unknown:
            sys.exit(f"ERROR: unknown head(s) {unknown}; valid: {', '.join(BACKBONE_HEADS)}")

        return [ModelBuilder.model_key(name, head) for name in BACKBONE_CONFIG_REGISTRY for head in self.config.heads]

    def _search_space(self, model_name: str) -> dict:
        config_cls = BACKBONE_CONFIG_REGISTRY[ModelBuilder.split_key(model_name)[0]]

        return SearchSpace.of(config_cls, self.config.tuning)

    def selected_models(self, target_model: str | None = None) -> list[str]:
        registry = BACKBONE_CONFIG_REGISTRY

        if target_model is None:
            keys    = self._model_keys()
            skip    = set(self.config.skip_models)
            unknown = sorted(skip - set(keys) - set(registry))
            if unknown:
                sys.exit(f"ERROR: skip_models entries match no model: {unknown}; registry keys: {', '.join(registry)}; model keys: {', '.join(keys)}")

            return [key for key in keys if key not in skip and ModelBuilder.split_key(key)[0] not in skip]

        name, head = ModelBuilder.split_key(target_model)
        if name not in registry:
            sys.exit(f"ERROR: unknown model '{name}'. Available: {list(registry.keys())}")
        if head not in BACKBONE_HEADS:
            sys.exit(f"ERROR: unknown head '{head}'. Available: {list(BACKBONE_HEADS)}")

        return [target_model]

    def planned_workers(self) -> list[tuple[str, str, int, int]]:
        workers = []
        for model_name in self.selected_models():
            counts = self._distribute_trials(self.config.tuning.n_trials, self.worker_count())
            workers += [(model_name, self._study_name(model_name), count, index) for index, count in enumerate(counts) if count > 0]

        return workers

    def worker_argv(self, model_name: str, study_name: str, n_trials: int, worker_index: int) -> list[str]:
        return [
            "--worker",
            "--model",        model_name,
            "--n-trials",     str(n_trials),
            "--worker-index", str(worker_index),
            "--study-name",   study_name,
            "--storage-url",  self.storage_url,
            "--run-tag",      self.tag,
            "--run-dir",      str(self.run_dir),
        ]

    def _worker_job(self, model_name: str, n_trials: int, worker_index: int) -> GpuJob:
        command = [sys.executable, str(self.entry_script), *self.worker_argv(model_name, self._study_name(model_name), n_trials, worker_index)]

        return GpuJob(
            name     = f"{model_name}_w{worker_index}",
            command  = command,
            log_path = self.run_dir / model_name / f"worker{worker_index}.log",
        )

    def _dispatch_workers(self, model_name: str, trial_counts: list[int]) -> bool:
        jobs = []
        for worker_index, n_trials in enumerate(trial_counts):
            if n_trials == 0:
                continue

            jobs.append(self._worker_job(model_name, n_trials, worker_index))

        queue   = GpuQueue(gpus=self.config.gpus, logger=self.logger, pool_file=GpuPoolFile.resolve(self.config.gpus_file, self.run_dir))
        results = queue.run(jobs)

        self.config.gpus = list(queue.gpus)

        return all(result.status == "DONE" for result in results)

    def _study_exists(self, model_name: str) -> bool:
        return self._study_name(model_name) in optuna.get_all_study_names(TuningStorage.open(self.storage_url))

    def _load_or_create_study(self, model_name: str) -> optuna.Study:
        return optuna.create_study(
            study_name     = self._study_name(model_name),
            storage        = TuningStorage.open(self.storage_url),
            direction      = "minimize",
            load_if_exists = True,
        )

    def _fail_stale_trials(self, study: optuna.Study) -> list[int]:
        running = [trial.number for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))]
        optuna.storages.fail_stale_trials(study)
        failed  = {trial.number for trial in study.get_trials(deepcopy=False, states=(TrialState.FAIL,))}

        return [number for number in running if number in failed]

    def _count_state(self, study: optuna.Study, state: TrialState) -> int:
        return len(study.get_trials(deepcopy=False, states=(state,)))

    def _store(self, record: dict) -> None:
        self.results = [kept for kept in self.results if kept["model"] != record["model"]] + [record]
        FileIO.save_json(self.results, self.summary_path, indent=2)

    def _plot_study(self, study: optuna.Study, model_name: str) -> None:
        plot_dir = self.run_dir / model_name / "study_plots"
        plotter  = StudyPlotter(self.logger)
        saved    = plotter.render(study, plot_dir)
        self.logger.info(f"Saved {len(saved)} study plots to {plot_dir}")

    def _record(self, model_name: str, status: str, study: optuna.Study, payload, path, stop_reason: str | None) -> dict:
        return {
            "model"            : model_name,
            "status"           : status,
            "trials_completed" : self._count_state(study, TrialState.COMPLETE),
            "trials_pruned"    : self._count_state(study, TrialState.PRUNED),
            "trials_failed"    : self._count_state(study, TrialState.FAIL),
            "trials_running"   : self._count_state(study, TrialState.RUNNING),
            "target"           : self.config.tuning.n_trials,
            "stop_reason"      : stop_reason,
            "val_loss"         : payload["val_loss"] if payload is not None else None,
            "best_config"      : str(path) if path is not None else None,
        }

    def _warn_running(self, study: optuna.Study, model_name: str) -> None:
        running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        if not running or self.config.stage == "compare":
            return

        listed = ", ".join(f"#{trial.number} (started {trial.datetime_start:%Y-%m-%d %H:%M})" for trial in running)
        self.logger.warning(f"{model_name}: {len(running)} trial(s) still RUNNING; their heartbeat is recent, so a worker of another launch may still be running them: {listed}")

    def _prepare_model(self, model_name: str) -> optuna.Study:
        self.logger.section(f"[{model_name}]")

        existed = self._study_exists(model_name)
        study   = self._load_or_create_study(model_name)

        if existed:
            self.prepared["loaded"].append(model_name)
            self.logger.warning(f"{model_name}: existing study '{study.study_name}' holds {StudyProgress.phrase(StudyProgress.counts(study), self.config.tuning.n_trials)}; this launch tops it up")
        else:
            self.prepared["created"].append(model_name)
            self.logger.info(f"{model_name}: created study '{study.study_name}' in {self.storage_url}")

        stale = self._fail_stale_trials(study)
        if stale:
            self.logger.warning(f"{model_name}: marked {len(stale)} trial(s) whose heartbeat stopped as failed, each left by a worker that died mid-trial: {', '.join(f'#{number}' for number in stale)}")

        self._warn_running(study, model_name)

        return study

    def _run_model(self, study: optuna.Study, model_name: str, tune_cfg: TuningConfig) -> bool:
        counted   = StudyTargetStop.describe(study, tune_cfg.n_trials)
        remaining = max(0, tune_cfg.n_trials - StudyTargetStop.counted(study))
        shares    = self._distribute_trials(remaining, self.worker_count())

        if remaining == 0:
            self.logger.subsection(f"Target reached: {counted}; no worker dispatched")
            return True

        self.logger.subsection(f"{counted}; running {remaining} across {sum(1 for share in shares if share > 0)} worker(s)")

        return self._dispatch_workers(model_name, shares)

    def _stop_reason(self, study: optuna.Study, model_name: str, tune_cfg: TuningConfig) -> str | None:
        met = StudyStopRules(tune_cfg, self.logger).already_met(study)
        if met is not None:
            return met

        snapshot = self.run_dir / model_name / StudyProgress.FILENAME
        if not snapshot.exists():
            return None

        return FileIO.load_json(snapshot)["stop_reason"]

    def _warn_trials(self, study: optuna.Study, model_name: str) -> None:
        failed  = self._count_state(study, TrialState.FAIL)
        running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))

        if failed:
            self.logger.warning(f"{model_name}: {failed} trial(s) failed; each worker log names the error on the trial's line")

        if running:
            self.logger.warning(f"{model_name}: {len(running)} trial(s) still RUNNING after the workers ended, left by workers that died mid-trial: {', '.join(f'#{trial.number}' for trial in running)}")

    def _status(self, model_name: str, payload, stop_reason: str | None, ok: bool, counts: str) -> str:
        if payload is None:
            return "FAILED"

        if ok and stop_reason in StudyStopRules.STUDY_ENDS:
            return "DONE"

        if not ok:
            why = "a worker of this launch failed"
        elif stop_reason is None:
            why = "no stop rule fired and no worker recorded why it ended"
        else:
            why = f"the workers ended on the {stop_reason} stop before the target or a stop rule was reached"

        self.logger.warning(f"{model_name}: PARTIAL, {counts}; {why}. Resuming the run tops the study up")

        return "PARTIAL"

    def _report_model(self, study: optuna.Study, model_name: str, tune_cfg: TuningConfig, ok: bool) -> None:
        writer      = BestConfigWriter(model_name, self._search_space(model_name), self.run_dir / model_name / "best_config.json")
        payload     = writer.write(study)
        stop_reason = self._stop_reason(study, model_name, tune_cfg)
        counts      = StudyProgress.phrase(StudyProgress.counts(study), tune_cfg.n_trials)

        self.reported.append(model_name)
        self._warn_trials(study, model_name)

        status = self._status(model_name, payload, stop_reason, ok, counts)

        if payload is None:
            self.logger.error(f"No completed trials for {model_name}: {counts}")
            self._store(self._record(model_name, status, study, None, None, stop_reason))
            return

        self.logger.subsection(f"Best, trial {payload['trial']}, best validation loss = {payload['val_loss']:.6f}")
        self.logger.kv_table(payload["params"], title="Best Params")

        if tune_cfg.emit_study_plots:
            self._plot_study(study, model_name)

        self._store(self._record(model_name, status, study, payload, writer.path, stop_reason))

    def _tune_model(self, model_name: str, tune_cfg: TuningConfig) -> None:
        study = self._prepare_model(model_name)

        if self.config.stage == "prepare":
            return

        ok = self._run_model(study, model_name, tune_cfg) if self.config.stage == "full" else True

        self._report_model(study, model_name, tune_cfg, ok)

    def _configuration(self, models_to_run: list[str], resume: bool) -> dict:
        tune_cfg = self.config.tuning

        return {
            "Run tag"          : self.tag,
            "Stage"            : self.config.stage,
            "Mode"             : "resume" if resume else "new",
            "Models"           : f"{len(models_to_run)}: {', '.join(models_to_run)}",
            "Skipped"          : ", ".join(self.config.skip_models) or "none",
            "GPUs"             : self.config.gpus,
            "Trial target"     : tune_cfg.n_trials,
            "Search groups"    : SearchSpace.describe(tune_cfg),
            "Plateau patience" : tune_cfg.plateau_patience or "off",
            "Time budget"      : f"{tune_cfg.time_budget_h} h" if tune_cfg.time_budget_h else "none",
            "Storage"          : self.storage_url,
            "Log dir"          : str(self.run_dir),
        }

    def _log_resume_changes(self, changed: dict, archived: Path | None) -> None:
        if changed:
            self.logger.warning(f"{len(changed)} recorded leaves changed on resume; trials from this launch train under the new values, so the study mixes trials scored under both")
            self.logger.metrics_table([{"Leaf": key, "Recorded": repr(old), "Now": repr(new)} for key, (old, new) in changed.items()], columns=["Leaf", "Recorded", "Now"], title="Changed on resume")

        if archived is not None:
            self.logger.info(f"The resolved config of the earlier launch was kept at {archived}")

    def _load_results(self) -> None:
        if self.config.stage == "prepare" or not self.summary_path.is_file():
            return

        stored = FileIO.load_json(self.summary_path)
        stale  = [record["model"] for record in stored if set(record) != set(self.RECORD_KEYS)]
        if stale:
            sys.exit(f"ERROR: {self.summary_path} holds records in an older layout for {', '.join(stale)}; delete it, and the compare or full launches rewrite it from the studies")

        self.results = stored
        self.logger.info(f"{self.summary_path.name} holds the records of {len(stored)} model(s) from earlier launches ({', '.join(record['model'] for record in stored)}); this launch replaces those of the models it reports")

    def _log_summary(self) -> list[str]:
        self.logger.section("Summary")
        if not self.results:
            return []

        rows = {}
        for record in self.results:
            val_loss              = f"{record['val_loss']:.6f}" if record["val_loss"] is not None else "none"
            rows[record["model"]] = f"status={record['status']}, {record['trials_completed']} complete, {record['trials_pruned']} pruned, {record['trials_failed']} failed, {record['trials_running']} running of target {record['target']}, stop={record['stop_reason'] or 'none'}, val_loss={val_loss}"
        self.logger.kv_table(rows, title="Models")

        unfinished = [f"{record['model']} {record['status']}" for record in self.results if record["model"] in self.reported and record["status"] != "DONE"]
        if unfinished:
            self.logger.error(f"{len(unfinished)} model(s) did not finish their study: {', '.join(unfinished)}; the run exits non-zero so no caller carries a short study forward")

        return unfinished

    def _log_done(self) -> None:
        if self.config.stage == "prepare":
            self.logger.kv_table({
                "Studies prepared" : len(self.prepared["created"]) + len(self.prepared["loaded"]),
                "Created"          : ", ".join(self.prepared["created"]) or "none",
                "Loaded"           : ", ".join(self.prepared["loaded"]) or "none",
                "Storage"          : self.storage_url,
            }, title="Done")
            return

        kept = [record["model"] for record in self.results if record["model"] not in self.reported]
        done = {"Models reported": len(self.reported), "Records kept from earlier launches": ", ".join(kept) or "none"}
        if self.reported:
            done["Results saved"] = str(self.summary_path)

        self.logger.kv_table(done, title="Done")

    def schedule(self, target_model: str | None = None, resume: bool = False) -> None:
        if self.config.stage not in self.STAGES:
            sys.exit(f"ERROR: unknown tuning stage '{self.config.stage}'; valid: {', '.join(self.STAGES)}")

        if self.config.stage == "full" and not self.config.gpus:
            sys.exit(f"ERROR: gpus is empty; {GpuQueue.EMPTY_POOL_MESSAGE}")

        if self.config.stage == "compare" and not self.db_path.exists():
            sys.exit(f"ERROR: stage 'compare' found no study database at {self.db_path}; run the prepare stage and the trial workers first")

        if self.config.stage == "full" and resume and not self.db_path.exists():
            sys.exit(f"ERROR: --resume given but no study found at {self.db_path}")

        if self.config.stage in ("full", "prepare") and not resume and self.db_path.exists():
            sys.exit(f"ERROR: run tag '{self.tag}' already has a study at {self.db_path}; pass --resume, or point resume_from at the run, to top it up under its recorded definition")

        tune_cfg      = self.config.tuning
        resolved_path = self.run_dir / "resolved_config.json"

        changed = self._guard_resumed_config(resolved_path) if resume else {}

        self.run_dir.mkdir(parents=True, exist_ok=True)
        archived = self._archive_previous_config(resolved_path)
        ConfigCli.save_resolved(self.config, resolved_path)

        self.logger = Logger(log_dir=str(self.run_dir), name="tune_scheduler")

        with self.logger:
            self.logger.adopt("optuna")

            models_to_run = self.selected_models(target_model)

            self.logger.section("Hyperparameter Tuning")
            self.logger.kv_table(self._configuration(models_to_run, resume), title="Configuration")

            if self.resumed is not None:
                self.resumed.log(self.logger)

            self._log_resume_changes(changed, archived)
            self._load_results()

            for model_name in models_to_run:
                self._tune_model(model_name, tune_cfg)

            unfinished = self._log_summary()
            self._log_done()

        if unfinished:
            sys.exit(f"ERROR: tuning run '{self.tag}' ended with model(s) not DONE: {', '.join(unfinished)}; see {self.summary_path}")
