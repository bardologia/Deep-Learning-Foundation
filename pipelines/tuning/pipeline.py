"""Scheduler driving Optuna studies for every model in a tuning run.

One study per model key is kept in a shared SQLite storage; trials are executed
by worker subprocesses fanned out across the GPU pool. The scheduler tops each
study up to its trial target, writes the best configuration and the study
plots, and refuses to resume a run whose study-defining settings changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState

from configuration.tuning                 import TuningConfig
from models                               import BACKBONE_HEADS, config_registry
from pipelines.shared.model.model_builder import ModelBuilder
from pipelines.tuning.plots               import StudyPlotter
from pipelines.tuning.storage             import TuningStorage
from pipelines.tuning.tuners              import BestConfigWriter
from tools.data.io                        import FileIO
from tools.orchestration.gpu_queue        import GpuJob, GpuPoolFile, GpuQueue
from tools.monitoring.logger              import Logger
from tools.runtime.config_cli             import ConfigCli


class TuningScheduler:
    """Runs and resumes the Optuna studies of a tuning run, model by model.

    Attributes:
        SUPPORTED_TYPES : Training types that can be tuned.
        STUDY_KEYS      : Config keys whose values define a study and may not change on resume.
        MUTABLE_KEYS    : Study keys exempted from that rule.
        tag             : Run tag naming the tuning directory and the studies.
        config          : Tuning entry configuration.
        entry_script    : Entry script the worker subprocesses re-invoke.
        resumed         : Description of the recorded run this launch continues, or None.
        run_dir         : Directory holding the study database, logs and results.
        storage_url     : SQLAlchemy URL of the shared Optuna storage.
        summary_path    : Path of the per-model results JSON.
        db_path         : Path of the Optuna SQLite database.
        logger          : Logger of the scheduler, created when scheduling starts.
        results         : Per-model result records collected during the run.
    """

    SUPPORTED_TYPES = ("backbone", "image_autoencoder", "jepa", "profile_autoencoder")
    STUDY_KEYS      = ("training_type", "heads", "skip_models", "tuning", "curriculum", "paths.dataset_path", "paths.parameters_path", "paths.secondary_labels")
    MUTABLE_KEYS    = ("tuning.n_trials",)

    def __init__(self, tag: str, config, entry_script: Path, resumed=None) -> None:
        """Resolves the tuning paths and quiets Optuna's own logging.

        Args:
            tag          : Run tag naming the tuning directory and its studies.
            config       : Tuning entry configuration.
            entry_script : Entry script the worker subprocesses re-invoke.
            resumed      : Resumed-run description logged when scheduling starts, or None.
        """
        self.tag          = tag
        self.config       = config
        self.entry_script = Path(entry_script)
        self.resumed      = resumed
        self.run_dir      = Path(config.paths.log_base_dir) / tag
        self.storage_url  = f"sqlite:///{self.run_dir / 'optuna.db'}"
        self.summary_path = self.run_dir / "tuning_results.json"
        self.db_path      = self.run_dir / "optuna.db"

        self.logger  = None
        self.results = []

        self._quiet_optuna()

    def _quiet_optuna(self) -> None:
        """Raises Optuna's log threshold so only warnings reach the console."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _defines_study(self, key: str) -> bool:
        """Returns whether a config key defines the study and is frozen on resume."""
        if key in self.MUTABLE_KEYS:
            return False

        return any(key == prefix or key.startswith(f"{prefix}.") for prefix in self.STUDY_KEYS)

    def _guard_resumed_config(self, resolved_path: Path) -> None:
        """Aborts the resume when a study-defining setting changed.

        Args:
            resolved_path : Path of the resolved config saved by the first run.

        Raises:
            SystemExit : If any study-defining key differs from the stored value.
        """
        stored  = FileIO.load_json(resolved_path)
        current = ConfigCli.to_mapping(self.config)
        changed = {key: (stored[key], value) for key, value in current.items() if self._defines_study(key) and stored[key] != value}

        if changed:
            sys.exit(f"ERROR: --resume cannot change the fields that define the study '{self.tag}'; trials already stored were optimised under {', '.join(f'{key}={old!r} (now {new!r})' for key, (old, new) in sorted(changed.items()))} — start a new run tag instead")

    def _distribute_trials(self, total: int, n_workers: int) -> list[int]:
        """Splits the outstanding trials as evenly as possible over the workers."""
        base  = total // n_workers
        extra = total % n_workers
        return [base + (1 if i < extra else 0) for i in range(n_workers)]

    def _study_name(self, model_name: str) -> str:
        """Returns the Optuna study name of one model under this run tag."""
        return f"{model_name}_{self.tag}"

    def _registry(self) -> dict:
        """Returns the model-name to config-class registry of the training type."""
        return config_registry(self.config.training_type)

    def _model_keys(self) -> list[str]:
        """Returns every model key to tune, expanding backbones over the heads.

        Returns:
            The registry keys for autoencoder tuning, or the
            ``<backbone>[-<head>]`` keys for backbone and JEPA tuning.

        Raises:
            SystemExit : If an unknown head is requested, or a structured head is combined with a coupled profile autoencoder in JEPA tuning.
        """
        if self.config.training_type not in ("backbone", "jepa"):
            return list(self._registry().keys())

        unknown = [head for head in self.config.heads if head not in BACKBONE_HEADS]
        if unknown:
            sys.exit(f"ERROR: unknown head(s) {unknown}; valid: {', '.join(BACKBONE_HEADS)}")

        structured = [head for head in self.config.heads if head != "conv"]
        if self.config.training_type == "jepa" and self.config.jepa.profile_autoencoder_run and structured:
            sys.exit(f"ERROR: head(s) {structured} are invalid for JEPA tuning with a coupled profile autoencoder; the backbone predicts the embedding and only the conv head projects to it. Set heads=['conv'] or drop jepa.profile_autoencoder_run.")

        return [ModelBuilder.model_key(name, head) for name in self._registry() for head in self.config.heads]

    def _search_space(self, model_name: str) -> dict:
        """Returns the model's tunable learning-rate and architecture parameters."""
        config_cls = self._registry()[ModelBuilder.split_key(model_name)[0]]
        return {**config_cls.tunable_lr_params(), **config_cls.tunable_arch_params()}

    def _worker_job(self, model_name: str, n_trials: int, worker_index: int) -> GpuJob:
        """Builds the GPU job running one tuning worker.

        Args:
            model_name   : Model key the worker tunes.
            n_trials     : Trials this worker is responsible for.
            worker_index : Index distinguishing the workers of a model.

        Returns:
            The GPU job invoking the entry script in worker mode.
        """
        command = [
            sys.executable, str(self.entry_script),
            "--worker",
            "--model",        model_name,
            "--n-trials",     str(n_trials),
            "--study-name",   self._study_name(model_name),
            "--storage-url",  self.storage_url,
            "--run-tag",      self.tag,
            "--run-dir",      str(self.run_dir),
        ]

        return GpuJob(
            name     = f"{model_name}_w{worker_index}",
            command  = command,
            log_path = self.run_dir / model_name / f"worker{worker_index}.log",
        )

    def _dispatch_workers(self, model_name: str, trial_counts: list[int]) -> bool:
        """Runs the tuning workers of one model across the GPU pool.

        The pool the queue ends with replaces the configured selection, so a
        resize applied while this model was tuning sets how many workers the
        next model spreads its trials over.

        Args:
            model_name   : Model key being tuned.
            trial_counts : Trials assigned to each worker; zero entries are skipped.

        Returns:
            True when every dispatched worker finished successfully.
        """
        jobs = []
        for worker_index, n_trials in enumerate(trial_counts):
            if n_trials == 0:
                continue

            jobs.append(self._worker_job(model_name, n_trials, worker_index))

        queue   = GpuQueue(gpus=self.config.gpus, logger=self.logger, pool_file=GpuPoolFile.resolve(self.config.gpus_file, self.run_dir))
        results = queue.run(jobs)

        self.config.gpus = list(queue.gpus)

        return all(result.status == "DONE" for result in results)

    def _load_or_create_study(self, model_name: str) -> optuna.Study:
        """Opens the model's minimisation study, creating it if it does not exist."""
        return optuna.create_study(
            study_name     = self._study_name(model_name),
            storage        = TuningStorage.open(self.storage_url),
            direction      = "minimize",
            load_if_exists = True,
        )

    def _fail_stale_trials(self, study: optuna.Study) -> int:
        """Marks heartbeat-stale running trials as failed and returns how many."""
        before = self._count_state(study, TrialState.FAIL)
        optuna.storages.fail_stale_trials(study)

        return self._count_state(study, TrialState.FAIL) - before

    def _count_done(self, study: optuna.Study) -> int:
        """Returns the number of trials that completed or were pruned."""
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))

    def _count_state(self, study: optuna.Study, state: TrialState) -> int:
        """Returns the number of trials in the given state."""
        return len(study.get_trials(deepcopy=False, states=(state,)))

    def _save_results(self) -> None:
        """Writes the per-model result records collected so far to disk."""
        FileIO.save_json(self.results, self.summary_path, indent=2)

    def _plot_study(self, study: optuna.Study, model_name: str) -> None:
        """Renders the study plots for a model into its ``study_plots`` directory."""
        plot_dir = self.run_dir / model_name / "study_plots"
        plotter  = StudyPlotter(self.logger)
        saved    = plotter.render(study, plot_dir)
        self.logger.info(f"Saved {len(saved)} study plots to {plot_dir}")

    def _record(self, model_name: str, status: str, study: optuna.Study, payload, path) -> dict:
        """Builds the summary record of one tuned model.

        Args:
            model_name : Model key that was tuned.
            status     : Outcome label (DONE, PARTIAL or FAILED).
            study      : The model's Optuna study, read for trial counts.
            payload    : Best-trial payload, or None when no trial completed.
            path       : Path of the written best-config file, or None.

        Returns:
            The result record appended to the tuning summary.
        """
        return {
            "model"           : model_name,
            "status"          : status,
            "trials_completed": self._count_state(study, TrialState.COMPLETE),
            "trials_pruned"   : self._count_state(study, TrialState.PRUNED),
            "trials_failed"   : self._count_state(study, TrialState.FAIL),
            "val_loss"        : payload["val_loss"] if payload is not None else None,
            "best_config"     : str(path) if path is not None else None,
        }

    def _tune_model(self, model_name: str, tune_cfg: TuningConfig) -> None:
        """Tops one model's study up to its trial target and records the outcome.

        Stale trials are failed first, the outstanding trials are split across
        the GPUs, and the best configuration, plots and summary record are
        written afterwards.

        Args:
            model_name : Model key to tune.
            tune_cfg   : Tuning settings holding the trial target and plot flag.
        """
        self.logger.section(f"[{model_name}]")

        study = self._load_or_create_study(model_name)
        stale = self._fail_stale_trials(study)
        if stale > 0:
            self.logger.info(f"Marked {stale} stale running trials as failed")

        done      = self._count_done(study)
        remaining = max(0, tune_cfg.n_trials - done)
        n_gpus    = len(self.config.gpus)
        ok        = True

        if remaining == 0:
            self.logger.subsection(f"Target reached — {done}/{tune_cfg.n_trials} trials already complete")
        else:
            self.logger.subsection(f"{done}/{tune_cfg.n_trials} trials done — running {remaining} across {n_gpus} GPUs")
            counts = self._distribute_trials(remaining, n_gpus)
            ok     = self._dispatch_workers(model_name, counts)

        writer  = BestConfigWriter(model_name, self._search_space(model_name), self.run_dir / model_name / "best_config.json")
        payload = writer.write(study)

        if payload is None:
            self.logger.error(f"No completed trials for {model_name}")
            self.results.append(self._record(model_name, "FAILED", study, None, None))
            self._save_results()
            return

        self.logger.subsection(f"Best — trial {payload['trial']}  best validation loss under the {self.config.training_type} objective = {payload['val_loss']:.6f}")
        self.logger.kv_table(payload["params"], title="Best Params")

        if tune_cfg.emit_study_plots:
            self._plot_study(study, model_name)

        self.results.append(self._record(model_name, "DONE" if ok else "PARTIAL", study, payload, writer.path))
        self._save_results()

    def schedule(self, target_model: str | None = None, resume: bool = False) -> None:
        """Validates the request, then tunes every selected model in turn.

        Args:
            target_model : Single model key to tune, or None to tune all models that are not skipped.
            resume       : Whether to continue an existing study database.

        Raises:
            SystemExit : If the training type is unsupported, the GPU pool is empty, the resume flag disagrees with the database on disk, or an unknown model or head was requested.
        """
        if self.config.training_type not in self.SUPPORTED_TYPES:
            sys.exit(f"ERROR: tuning does not support training_type '{self.config.training_type}'; supported: {', '.join(self.SUPPORTED_TYPES)}")

        if not self.config.gpus:
            sys.exit(f"ERROR: gpus is empty; {GpuQueue.EMPTY_POOL_MESSAGE}")

        if resume and not self.db_path.exists():
            sys.exit(f"ERROR: --resume given but no study found at {self.db_path}")
        if not resume and self.db_path.exists():
            sys.exit(f"ERROR: run tag '{self.tag}' already has a study at {self.db_path} — pass --resume to continue it")

        tune_cfg      = self.config.tuning
        resolved_path = self.run_dir / "resolved_config.json"

        if resume:
            self._guard_resumed_config(resolved_path)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        ConfigCli.save_resolved(self.config, resolved_path)

        self.logger = Logger(log_dir=str(self.run_dir), name="tune_scheduler")

        registry   = self._registry()
        all_models = self._model_keys()
        if target_model is not None:
            name, head = ModelBuilder.split_key(target_model)
            if name not in registry:
                self.logger.close()
                sys.exit(f"ERROR: unknown model '{name}'. Available: {list(registry.keys())}")
            if head not in BACKBONE_HEADS:
                self.logger.close()
                sys.exit(f"ERROR: unknown head '{head}'. Available: {list(BACKBONE_HEADS)}")
            models_to_run = [target_model]
        else:
            skip          = set(self.config.skip_models)
            models_to_run = [m for m in all_models if m not in skip and ModelBuilder.split_key(m)[0] not in skip]

        self.logger.section("Hyperparameter Tuning")
        self.logger.kv_table({
            "Run tag"      : self.tag,
            "Mode"         : "resume" if resume else "new",
            "Models"       : len(models_to_run),
            "GPUs"         : self.config.gpus,
            "Trial target" : tune_cfg.n_trials,
            "Storage"      : self.storage_url,
            "Log dir"      : str(self.run_dir),
        }, title="Configuration")

        if self.resumed is not None:
            self.resumed.log(self.logger)

        for model_name in models_to_run:
            self._tune_model(model_name, tune_cfg)

        self.logger.section("Summary")
        self.logger.kv_table({
            "Models tuned" : len(self.results),
            "Results saved": str(self.summary_path),
        }, title="Done")
        self.logger.close()
