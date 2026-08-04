from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import optuna
import pytest


class FakeLogger:
    def __init__(self) -> None:
        self.sections    = []
        self.subsections = []
        self.infos       = []
        self.warnings    = []
        self.errors      = []
        self.tables      = []

    def section(self, title) -> None:
        self.sections.append(title)

    def subsection(self, title) -> None:
        self.subsections.append(title)

    def info(self, message) -> None:
        self.infos.append(message)

    def warning(self, message) -> None:
        self.warnings.append(message)

    def error(self, message) -> None:
        self.errors.append(message)

    def kv_table(self, data, title=None, key_header="Field", value_header="Value") -> None:
        self.tables.append((title, dict(data)))

    def close(self) -> None:
        pass


class TuneCfg:
    def __init__(self, n_epochs=2, early_stop_patience=2, base_seed=42) -> None:
        self.n_epochs                = n_epochs
        self.early_stop_patience     = early_stop_patience
        self.base_seed               = base_seed
        self.n_trials                = 12
        self.pruner_n_startup_trials = 0
        self.pruner_n_warmup_steps   = 0
        self.emit_trial_docs         = False
        self.emit_study_plots        = False


@pytest.fixture
def fake_logger():
    return FakeLogger()


@pytest.fixture
def tune_cfg():
    return TuneCfg()


@pytest.fixture
def seeded_sampler():
    return optuna.samplers.TPESampler(seed=0, n_startup_trials=4, multivariate=True)


@pytest.fixture
def quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return True
