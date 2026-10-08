"""Shared fixtures for the hyperparameter tuning tests.

Provides a recording logger stub, a minimal tuning settings object, a seeded
Optuna sampler and a quiet Optuna verbosity fixture.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import optuna
import pytest


class FakeLogger:
    """Logger stub recording every message it receives.

    Attributes:
        sections    : Section titles received.
        subsections : Subsection titles received.
        infos       : Informational messages received.
        warnings    : Warning messages received.
        errors      : Error messages received.
        tables      : (title, mapping) pairs received as key-value tables.
    """
    def __init__(self) -> None:
        """Initialises empty buffers for every recorded message kind."""
        self.sections    = []
        self.subsections = []
        self.infos       = []
        self.warnings    = []
        self.errors      = []
        self.tables      = []

    def section(self, title) -> None:
        """Records a section title."""
        self.sections.append(title)

    def subsection(self, title) -> None:
        """Records a subsection title."""
        self.subsections.append(title)

    def info(self, message) -> None:
        """Records an informational message."""
        self.infos.append(message)

    def warning(self, message) -> None:
        """Records a warning message."""
        self.warnings.append(message)

    def error(self, message) -> None:
        """Records an error message."""
        self.errors.append(message)

    def kv_table(self, data, title=None, key_header="Field", value_header="Value") -> None:
        """Records a key-value table together with its title."""
        self.tables.append((title, dict(data)))

    def close(self) -> None:
        """Performs no teardown."""
        pass


class TuneCfg:
    """Minimal tuning settings object accepted by the tuners."""
    def __init__(self, n_epochs=2, early_stop_patience=2, base_seed=42) -> None:
        """Stores the epoch budget, patience and seed alongside fixed pruner, search-space and stop-rule settings."""
        self.n_epochs                = n_epochs
        self.early_stop_patience     = early_stop_patience
        self.base_seed               = base_seed
        self.n_trials                = 12
        self.pruner_n_startup_trials = 0
        self.pruner_n_warmup_steps   = 0
        self.search_lr               = True
        self.search_arch             = True
        self.search_batch            = False
        self.batch_sizes             = [64, 128, 256, 512, 1024]
        self.plateau_patience        = 0
        self.plateau_min_delta       = 0.0
        self.time_budget_h           = 0.0
        self.emit_trial_docs         = False
        self.emit_study_plots        = False


@pytest.fixture
def fake_logger():
    """Returns a fresh recording logger."""
    return FakeLogger()


@pytest.fixture
def tune_cfg():
    """Returns a tuning settings object with two epochs and patience two."""
    return TuneCfg()


@pytest.fixture
def seeded_sampler():
    """Returns a TPE sampler seeded for reproducible trials."""
    return optuna.samplers.TPESampler(seed=0, n_startup_trials=4, multivariate=True)


@pytest.fixture
def quiet_optuna():
    """Silences Optuna below warning level and returns True."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return True
