"""Tests for the tuning scheduler that fans studies out over GPUs.

Covers trial distribution, study naming and storage layout, stale trial
handling, result summaries, best-config persistence and the guards on
training type, GPU pool and resumed configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from types   import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState

from configuration.architectures.backbone import UNetConfig
from pipelines.tuning.pipeline            import TuningScheduler
from pipelines.tuning.storage             import TuningStorage
from pipelines.tuning.tuners              import ParamSampler


class StubLogger:
    """Logger stub recording (kind, message) pairs."""
    def __init__(self):
        """Initialises the message buffer."""
        self.messages = []

    def section(self, t):     self.messages.append(("section", t))
    def subsection(self, t):  self.messages.append(("subsection", t))
    def info(self, t):        self.messages.append(("info", t))
    def warning(self, t):     self.messages.append(("warning", t))
    def error(self, t):       self.messages.append(("error", t))
    def kv_table(self, *a, **k): pass
    def close(self): pass


def _make_config(tmp_path: Path, n_trials: int = 10):
    """Returns a tuning entry config namespace rooted at `tmp_path`."""
    tuning = SimpleNamespace(
        n_trials                = n_trials,
        n_epochs                = 2,
        base_seed               = 42,
        early_stop_patience     = 2,
        pruner_n_startup_trials = 0,
        pruner_n_warmup_steps   = 0,
        search_lr               = True,
        search_arch             = True,
        search_batch            = False,
        batch_sizes             = [64, 128, 256],
        emit_trial_docs         = False,
        emit_study_plots        = False,
    )

    paths = SimpleNamespace(log_base_dir=tmp_path, secondary_labels=None, parameters_path=None)

    return SimpleNamespace(
        paths         = paths,
        tuning        = tuning,
        training_type = "backbone",
        stage         = "full",
        gpus          = [0, 1, 2],
        n_workers     = 0,
        gpus_file     = "",
        skip_models   = [],
    )


def _orchestrator(tmp_path: Path, n_trials: int = 10) -> TuningScheduler:
    """Returns a TuningScheduler for `tmp_path` with a stub logger attached."""
    cfg  = _make_config(tmp_path, n_trials=n_trials)
    orch = TuningScheduler("testtag", cfg, entry_script=tmp_path / "entry.py")
    orch.logger = StubLogger()
    return orch


def test_distribute_trials_balances_remainder(tmp_path):
    """Verifies that trials are split evenly with the remainder spread over the first GPUs."""
    orch = _orchestrator(tmp_path)

    assert orch._distribute_trials(10, 3) == [4, 3, 3]
    assert orch._distribute_trials(9, 3)  == [3, 3, 3]
    assert orch._distribute_trials(2, 4)  == [1, 1, 0, 0]
    assert sum(orch._distribute_trials(17, 4)) == 17


def test_study_name_combines_model_and_tag(tmp_path):
    """Verifies that the study name joins the model key and the run tag."""
    orch = _orchestrator(tmp_path)
    assert orch._study_name("unet") == "unet_testtag"


def test_search_space_matches_config_union(tmp_path):
    """Verifies that the search space is the union of tunable LR and architecture params."""
    orch  = _orchestrator(tmp_path)
    space = orch._search_space("unet")

    expected = {**UNetConfig.tunable_lr_params(), **UNetConfig.tunable_arch_params()}
    assert space == expected


def test_search_space_follows_the_switched_on_groups(tmp_path):
    """Verifies that the scheduler decodes against the groups the tuning settings switch on, the batch size included."""
    orch = _orchestrator(tmp_path)

    orch.config.tuning.search_arch = False
    assert orch._search_space("unet") == UNetConfig.tunable_lr_params()

    orch.config.tuning.search_arch  = True
    orch.config.tuning.search_lr    = False
    orch.config.tuning.search_batch = True
    assert orch._search_space("unet") == {**UNetConfig.tunable_arch_params(), "batch_size": {"type": "categorical", "choices": [64, 128, 256]}}

    orch.config.tuning.search_arch  = False
    orch.config.tuning.search_batch = False
    with pytest.raises(ValueError, match="every search group is off"):
        orch._search_space("unet")


def test_a_jepa_study_decodes_against_the_architecture_alone(tmp_path):
    """The JEPA tuner searches the architecture only, so the scheduler decodes its best trial against that group."""
    orch = _orchestrator(tmp_path)
    orch.config.training_type = "jepa"

    assert orch._search_space("unet") == UNetConfig.tunable_arch_params()


def test_model_keys_reject_structured_heads_for_profile_coupled_jepa(tmp_path):
    """Verifies that structured heads are rejected when JEPA uses a profile autoencoder."""
    orch = _orchestrator(tmp_path)
    orch.config.training_type = "jepa"
    orch.config.heads         = ["conv", "set_pred"]
    orch.config.jepa          = SimpleNamespace(profile_autoencoder_run="mlp_ae_run")

    with pytest.raises(SystemExit, match="only the conv head"):
        orch._model_keys()


def test_model_keys_allow_structured_heads_without_a_profile_ae(tmp_path):
    """Verifies that structured heads are allowed when no profile autoencoder is set."""
    orch = _orchestrator(tmp_path)
    orch.config.training_type = "jepa"
    orch.config.heads         = ["conv", "set_pred"]
    orch.config.jepa          = SimpleNamespace(profile_autoencoder_run=None)

    keys = orch._model_keys()

    assert "unet" in keys
    assert "unet-set_pred" in keys


def test_storage_and_paths_derived_from_tag(tmp_path):
    """Verifies that run directory, database and summary paths derive from the run tag."""
    orch = _orchestrator(tmp_path)

    assert orch.run_dir      == tmp_path / "testtag"
    assert orch.db_path      == tmp_path / "testtag" / "optuna.db"
    assert orch.summary_path == tmp_path / "testtag" / "tuning_results.json"
    assert orch.storage_url.startswith("sqlite:///")


def test_load_or_create_study_is_minimize_and_idempotent(tmp_path):
    """Verifies that the study minimizes and is reused rather than duplicated."""
    orch  = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    study1 = orch._load_or_create_study("unet")
    study2 = orch._load_or_create_study("unet")

    assert study1.direction == optuna.study.StudyDirection.MINIMIZE
    assert study1.study_name == study2.study_name


def test_count_done_counts_complete_and_pruned(tmp_path):
    """Verifies that done trials count complete and pruned states but not failures."""
    orch  = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    study = orch._load_or_create_study("unet")

    study.add_trial(optuna.trial.create_trial(value=0.5, state=TrialState.COMPLETE, params={}, distributions={}))
    study.add_trial(optuna.trial.create_trial(value=0.9, state=TrialState.PRUNED,   params={}, distributions={}))
    study.add_trial(optuna.trial.create_trial(state=TrialState.FAIL))

    assert orch._count_done(study) == 2
    assert orch._count_state(study, TrialState.FAIL) == 1


def test_running_trial_within_its_grace_period_is_left_alone(tmp_path):
    """Verifies that a running trial inside its heartbeat grace period is not failed."""
    orch  = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    study = orch._load_or_create_study("unet")

    study.add_trial(optuna.trial.create_trial(value=0.3, state=TrialState.COMPLETE, params={}, distributions={}))
    study.ask()

    failed = orch._fail_stale_trials(study)

    assert failed == 0
    assert orch._count_state(study, TrialState.FAIL) == 0
    assert orch._count_state(study, TrialState.RUNNING) == 1


def test_study_storage_carries_heartbeats(tmp_path):
    """Verifies that the study storage carries the configured heartbeat and grace period."""
    orch = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    assert orch._load_or_create_study("unet")._storage.get_heartbeat_interval() == TuningStorage.HEARTBEAT_INTERVAL_S

    storage = TuningStorage.open(orch.storage_url)

    assert storage.heartbeat_interval == TuningStorage.HEARTBEAT_INTERVAL_S
    assert storage.grace_period       == TuningStorage.GRACE_PERIOD_S


def test_record_summarizes_counts(tmp_path):
    """Verifies that a summary record carries model, status, trial count and best config."""
    orch  = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    study = orch._load_or_create_study("unet")

    study.add_trial(optuna.trial.create_trial(value=0.5, state=TrialState.COMPLETE, params={}, distributions={}))

    record = orch._record("unet", "DONE", study, {"val_loss": 0.5}, Path("/tmp/best.json"))

    assert record["model"]            == "unet"
    assert record["status"]           == "DONE"
    assert record["trials_completed"] == 1
    assert record["val_loss"]         == 0.5
    assert record["best_config"]      == "/tmp/best.json"


@pytest.mark.slow
def test_tune_model_extracts_synthetic_optimum_and_persists(tmp_path, monkeypatch):
    """Verifies that tuning recovers a synthetic optimum and persists best config and summary."""
    orch = _orchestrator(tmp_path, n_trials=30)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    space   = orch._search_space("unet")
    optimum = "gelu"

    def fake_dispatch(model_name, counts):
        """Runs the synthetic objective in-process instead of dispatching GPU workers."""
        sampler = ParamSampler()
        study   = orch._load_or_create_study(model_name)

        def objective(trial):
            """Returns a loss minimised by the target activation choice."""
            s    = sampler.sample(trial, space)
            base = 0.0 if s["activation"] == optimum else 1.0
            return base + 0.001 * s["dropout"]

        study.optimize(objective, n_trials=sum(counts), gc_after_trial=True)
        return True

    monkeypatch.setattr(orch, "_dispatch_workers", fake_dispatch)

    orch._tune_model("unet", orch.config.tuning)

    best_path = orch.run_dir / "unet" / "best_config.json"
    assert best_path.exists()

    payload = json.loads(best_path.read_text())
    assert payload["model"]              == "unet"
    assert payload["params"]["activation"] == optimum
    assert payload["val_loss"] < 0.5

    assert orch.summary_path.exists()
    results = json.loads(orch.summary_path.read_text())
    assert results[-1]["model"]  == "unet"
    assert results[-1]["status"] == "DONE"
    assert results[-1]["trials_completed"] == 30


@pytest.mark.slow
def test_tune_model_records_failure_with_no_trials(tmp_path, monkeypatch):
    """Verifies that a study with no completed trials is recorded as FAILED."""
    orch = _orchestrator(tmp_path, n_trials=5)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(orch, "_dispatch_workers", lambda model_name, counts: True)

    orch._tune_model("unet", orch.config.tuning)

    results = json.loads(orch.summary_path.read_text())
    assert results[-1]["status"]   == "FAILED"
    assert results[-1]["val_loss"] is None


@pytest.mark.slow
def test_tune_model_skips_dispatch_when_target_reached(tmp_path, monkeypatch):
    """Verifies that no workers are dispatched when the trial target is already met."""
    orch = _orchestrator(tmp_path, n_trials=2)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    study = orch._load_or_create_study("unet")
    for v in (0.4, 0.2):
        study.add_trial(optuna.trial.create_trial(value=v, state=TrialState.COMPLETE, params={"dropout": 0.1}, distributions={"dropout": optuna.distributions.FloatDistribution(0.0, 0.5)}))

    called = {"dispatch": False}

    def fake_dispatch(model_name, counts):
        """Records that dispatch was requested and reports success."""
        called["dispatch"] = True
        return True

    monkeypatch.setattr(orch, "_dispatch_workers", fake_dispatch)

    orch._tune_model("unet", orch.config.tuning)

    assert called["dispatch"] is False
    assert (orch.run_dir / "unet" / "best_config.json").exists()


def test_schedule_rejects_unsupported_training_type(tmp_path):
    """Verifies that scheduling an unsupported training type exits with an error."""
    cfg               = _make_config(tmp_path)
    cfg.training_type = "nonexistent_type"
    orch              = TuningScheduler("testtag", cfg, entry_script=tmp_path / "entry.py")

    with pytest.raises(SystemExit, match="nonexistent_type"):
        orch.schedule()


def test_schedule_rejects_an_empty_gpu_pool(tmp_path):
    """Verifies that scheduling with an empty GPU pool exits with an error."""
    cfg      = _make_config(tmp_path)
    cfg.gpus = []
    orch     = TuningScheduler("testtag", cfg, entry_script=tmp_path / "entry.py")

    with pytest.raises(SystemExit, match="gpus is empty"):
        orch.schedule()


def test_worker_rejects_unsupported_training_type(tmp_path):
    """Verifies that a worker rejects an unsupported training type."""
    from pipelines.tuning.workers import TuningWorker

    cfg               = _make_config(tmp_path)
    cfg.training_type = "nonexistent_type"
    worker            = TuningWorker("testtag", cfg)

    with pytest.raises(ValueError, match="nonexistent_type"):
        worker._build_tuner("gamma_net", cfg.tuning, StubLogger())


def test_image_ae_entry_template_forwards_normalization_and_augmentation(tmp_path):
    """Verifies that the image AE entry template forwards normalization and augmentation."""
    from configuration.dataset              import AugmentationConfig
    from configuration.normalization.general import NormalizationConfig
    from configuration.training             import ImageAeLossConfig
    from pipelines.tuning.workers           import TuningWorker

    cfg               = _make_config(tmp_path)
    cfg.image_ae_loss = ImageAeLossConfig()
    cfg.training      = SimpleNamespace()
    cfg.normalization = NormalizationConfig()
    cfg.augmentation  = AugmentationConfig()

    worker = TuningWorker("testtag", cfg)
    entry  = worker._image_ae_entry_template()

    assert entry.normalization is cfg.normalization
    assert entry.augmentation  is cfg.augmentation
    assert entry.ae_loss       is cfg.image_ae_loss


def _resumed_scheduler(tmp_path: Path):
    """Returns a scheduler whose run directory already holds a saved resolved config."""
    from configuration.tuning     import TuningEntryConfig
    from tools.runtime.config_cli import ConfigCli

    cfg                    = TuningEntryConfig()
    cfg.paths.log_base_dir = tmp_path

    orch = TuningScheduler("testtag", cfg, entry_script=tmp_path / "entry.py")
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    ConfigCli.save_resolved(cfg, orch.run_dir / "resolved_config.json")

    return orch


def test_resume_rejects_a_changed_objective(tmp_path):
    """Verifies that changing the epoch budget on resume is rejected."""
    orch = _resumed_scheduler(tmp_path)

    orch.config.tuning.n_epochs = 5

    with pytest.raises(SystemExit, match="tuning.n_epochs"):
        orch._guard_resumed_config(orch.run_dir / "resolved_config.json")


def test_resume_allows_more_trials_and_a_new_gpu_pool(tmp_path):
    """Verifies that raising the trial count and changing the GPU pool are allowed on resume."""
    orch = _resumed_scheduler(tmp_path)

    orch.config.tuning.n_trials = orch.config.tuning.n_trials + 50
    orch.config.gpus            = [0]

    orch._guard_resumed_config(orch.run_dir / "resolved_config.json")


def test_resume_rejects_a_changed_search_group(tmp_path):
    """The search groups define what the stored trials sampled, so a resume may not change them."""
    orch = _resumed_scheduler(tmp_path)

    orch.config.tuning.search_arch = False

    with pytest.raises(SystemExit, match="tuning.search_arch"):
        orch._guard_resumed_config(orch.run_dir / "resolved_config.json")


def test_prepare_stage_opens_the_study_without_dispatching_or_reporting(tmp_path, monkeypatch):
    """The prepare stage creates the study a separately submitted worker attaches to, and stops there."""
    orch = _orchestrator(tmp_path, n_trials=2)
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    orch.config.stage = "prepare"

    called = {"dispatch": False, "report": False}
    monkeypatch.setattr(orch, "_dispatch_workers", lambda model_name, counts: called.update(dispatch=True) or True)
    monkeypatch.setattr(orch, "_report_model", lambda study, model_name, tune_cfg, ok: called.update(report=True))

    orch._tune_model("unet", orch.config.tuning)

    assert called == {"dispatch": False, "report": False}
    assert orch.db_path.exists()


def test_compare_stage_reports_without_dispatching(tmp_path, monkeypatch):
    """The compare stage writes the reports of a study the workers filled without running more trials."""
    orch = _orchestrator(tmp_path, n_trials=2)
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    orch.config.stage = "compare"

    called = {"dispatch": False, "report": False}
    monkeypatch.setattr(orch, "_dispatch_workers", lambda model_name, counts: called.update(dispatch=True) or True)
    monkeypatch.setattr(orch, "_report_model", lambda study, model_name, tune_cfg, ok: called.update(report=True))

    orch._tune_model("unet", orch.config.tuning)

    assert called == {"dispatch": False, "report": True}


def test_worker_count_follows_the_configured_value_then_the_pool(tmp_path):
    """An explicit worker count wins, and without one there is a worker per device in the pool."""
    orch = _orchestrator(tmp_path)

    assert orch.worker_count() == 3

    orch.config.n_workers = 8
    assert orch.worker_count() == 8
