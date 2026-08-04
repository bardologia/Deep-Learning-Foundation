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
    def __init__(self):
        self.messages = []

    def section(self, t):     self.messages.append(("section", t))
    def subsection(self, t):  self.messages.append(("subsection", t))
    def info(self, t):        self.messages.append(("info", t))
    def warning(self, t):     self.messages.append(("warning", t))
    def error(self, t):       self.messages.append(("error", t))
    def kv_table(self, *a, **k): pass
    def close(self): pass


def _make_config(tmp_path: Path, n_trials: int = 10):
    tuning = SimpleNamespace(
        n_trials                = n_trials,
        n_epochs                = 2,
        base_seed               = 42,
        early_stop_patience     = 2,
        pruner_n_startup_trials = 0,
        pruner_n_warmup_steps   = 0,
        emit_trial_docs         = False,
        emit_study_plots        = False,
    )

    paths = SimpleNamespace(log_base_dir=tmp_path, secondary_labels=None, parameters_path=None)

    return SimpleNamespace(
        paths         = paths,
        tuning        = tuning,
        training_type = "backbone",
        gpus          = [0, 1, 2],
        gpus_file     = "",
        skip_models   = [],
    )


def _orchestrator(tmp_path: Path, n_trials: int = 10) -> TuningScheduler:
    cfg  = _make_config(tmp_path, n_trials=n_trials)
    orch = TuningScheduler("testtag", cfg, entry_script=tmp_path / "entry.py")
    orch.logger = StubLogger()
    return orch


def test_distribute_trials_balances_remainder(tmp_path):
    orch = _orchestrator(tmp_path)

    assert orch._distribute_trials(10, 3) == [4, 3, 3]
    assert orch._distribute_trials(9, 3)  == [3, 3, 3]
    assert orch._distribute_trials(2, 4)  == [1, 1, 0, 0]
    assert sum(orch._distribute_trials(17, 4)) == 17


def test_study_name_combines_model_and_tag(tmp_path):
    orch = _orchestrator(tmp_path)
    assert orch._study_name("unet") == "unet_testtag"


def test_search_space_matches_config_union(tmp_path):
    orch  = _orchestrator(tmp_path)
    space = orch._search_space("unet")

    expected = {**UNetConfig.tunable_lr_params(), **UNetConfig.tunable_arch_params()}
    assert space == expected


def test_storage_and_paths_derived_from_tag(tmp_path):
    orch = _orchestrator(tmp_path)

    assert orch.run_dir      == tmp_path / "testtag"
    assert orch.db_path      == tmp_path / "testtag" / "optuna.db"
    assert orch.summary_path == tmp_path / "testtag" / "tuning_results.json"
    assert orch.storage_url.startswith("sqlite:///")


def test_load_or_create_study_is_minimize_and_idempotent(tmp_path):
    orch  = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    study1 = orch._load_or_create_study("unet")
    study2 = orch._load_or_create_study("unet")

    assert study1.direction == optuna.study.StudyDirection.MINIMIZE
    assert study1.study_name == study2.study_name


def test_count_done_counts_complete_and_pruned(tmp_path):
    orch  = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    study = orch._load_or_create_study("unet")

    study.add_trial(optuna.trial.create_trial(value=0.5, state=TrialState.COMPLETE, params={}, distributions={}))
    study.add_trial(optuna.trial.create_trial(value=0.9, state=TrialState.PRUNED,   params={}, distributions={}))
    study.add_trial(optuna.trial.create_trial(state=TrialState.FAIL))

    assert orch._count_done(study) == 2
    assert orch._count_state(study, TrialState.FAIL) == 1


def test_running_trial_within_its_grace_period_is_left_alone(tmp_path):
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
    orch = _orchestrator(tmp_path)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    assert orch._load_or_create_study("unet")._storage.get_heartbeat_interval() == TuningStorage.HEARTBEAT_INTERVAL_S

    storage = TuningStorage.open(orch.storage_url)

    assert storage.heartbeat_interval == TuningStorage.HEARTBEAT_INTERVAL_S
    assert storage.grace_period       == TuningStorage.GRACE_PERIOD_S


def test_record_summarizes_counts(tmp_path):
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
    orch = _orchestrator(tmp_path, n_trials=30)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    space   = orch._search_space("unet")
    optimum = "gelu"

    def fake_dispatch(model_name, counts):
        sampler = ParamSampler()
        study   = orch._load_or_create_study(model_name)

        def objective(trial):
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
    orch = _orchestrator(tmp_path, n_trials=5)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(orch, "_dispatch_workers", lambda model_name, counts: True)

    orch._tune_model("unet", orch.config.tuning)

    results = json.loads(orch.summary_path.read_text())
    assert results[-1]["status"]   == "FAILED"
    assert results[-1]["val_loss"] is None


@pytest.mark.slow
def test_tune_model_skips_dispatch_when_target_reached(tmp_path, monkeypatch):
    orch = _orchestrator(tmp_path, n_trials=2)
    orch.run_dir.mkdir(parents=True, exist_ok=True)

    study = orch._load_or_create_study("unet")
    for v in (0.4, 0.2):
        study.add_trial(optuna.trial.create_trial(value=v, state=TrialState.COMPLETE, params={"dropout": 0.1}, distributions={"dropout": optuna.distributions.FloatDistribution(0.0, 0.5)}))

    called = {"dispatch": False}

    def fake_dispatch(model_name, counts):
        called["dispatch"] = True
        return True

    monkeypatch.setattr(orch, "_dispatch_workers", fake_dispatch)

    orch._tune_model("unet", orch.config.tuning)

    assert called["dispatch"] is False
    assert (orch.run_dir / "unet" / "best_config.json").exists()


def test_schedule_rejects_unsupported_training_type(tmp_path):
    cfg               = _make_config(tmp_path)
    cfg.training_type = "unrolled"
    orch              = TuningScheduler("testtag", cfg, entry_script=tmp_path / "entry.py")

    with pytest.raises(SystemExit, match="unrolled"):
        orch.schedule()


def test_worker_rejects_unsupported_training_type(tmp_path):
    from pipelines.tuning.workers import TuningWorker

    cfg               = _make_config(tmp_path)
    cfg.training_type = "unrolled"
    worker            = TuningWorker("testtag", cfg)

    with pytest.raises(ValueError, match="unrolled"):
        worker._build_tuner("gamma_net", cfg.tuning, StubLogger())


def test_image_ae_entry_template_forwards_normalization_and_augmentation(tmp_path):
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
