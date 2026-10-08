"""Tests for the pre-training orchestrator that sizes batches, loaders and the learning rate.

Covers the ordering of batch search before loader tuning, the dataset size
ceiling, the LR-probe wiring, the all-flags-off no-op path and failure
propagation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import tools.training.pretraining.orchestrator as orchestrator_module
from tools.training.pretraining.orchestrator import PretrainContext, PretrainOrchestrator


class _Logger:
    """Logger stub swallowing the section and table calls made by the orchestrator."""
    def section(self, *args, **kwargs):
        """Ignores a section heading."""
        pass

    def subsection(self, *args, **kwargs):
        """Ignores a subsection heading."""
        pass

    def kv_table(self, *args, **kwargs):
        """Ignores a key-value table."""
        pass

    def warning(self, *args, **kwargs):
        """Ignores a warning."""
        pass

    def metrics_table(self, *args, **kwargs):
        """Ignores a metrics table."""
        pass


def _pretrain(**overrides):
    """Returns a pretrain settings namespace with the given fields overridden."""
    base = dict(
        find_batch_size      = False,
        tune_loader          = False,
        find_lr              = False,
        vram_budget_gb       = 10.0,
        max_batch            = 8,
        measure_steps        = 1,
        worker_counts        = (0,),
        prefetch_factors     = (2,),
        warmup_batches       = 1,
        timed_batches        = 1,
        data_wait_target     = 0.05,
        lr_scale_min         = 1e-2,
        lr_scale_max         = 1e2,
        lr_rungs             = 20,
        lr_rung_steps        = 2,
        lr_probe_batches     = 2,
        lr_divergence_factor = 4.0,
        lr_safety_factor     = 10.0,
        lr_min_improvement   = 0.02,
        lr_noise_sigmas      = 3.0,
        seed                 = 0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _training():
    """Returns a training settings namespace with single-sample batches."""
    return SimpleNamespace(batch_size=1, num_workers=0, prefetch_factor=2, scale_lr_with_batch=True, lr_multiplier=1.0)


def _context(order, training, dataset=None, trainer=None):
    """Returns a PretrainContext whose trial step records the batch sizes it sees."""
    def trial(batch_size):
        """Returns a peak memory that exceeds the budget above batch size four."""
        order.append(("trial", batch_size))
        return 1.0 if batch_size <= 4 else 100.0

    return PretrainContext(
        dataset        = dataset if dataset is not None else list(range(64)),
        model          = None,
        to_model_input = None,
        forward_loss   = None,
        trial_step     = trial,
        device         = torch.device("cpu"),
        use_amp        = False,
        context_gb     = 0.0,
        on_oom         = lambda: None,
        trainer        = trainer,
    )


def test_batch_finder_runs_before_tuner_and_feeds_resolved_batch(monkeypatch, tmp_path):
    """Verifies that the resolved batch size is found first and passed to the tuner."""
    order    = []
    training = _training()

    class _FakeTuner:
        """Loader tuner stub recording the batch size it was tuned for."""
        def __init__(self, **kwargs):
            """Accepts and discards the tuner keyword arguments."""
            pass

        def run(self, batch_size):
            """Records the tuning call and returns a fixed loader choice."""
            order.append(("tune", batch_size))
            self.records = [{"phase": "workers", "num_workers": 3, "prefetch_factor": 4, "status": "ok", "end_to_end_samples_per_s": 100.0}, {"phase": "workers", "num_workers": 6, "prefetch_factor": 4, "status": "oom"}]
            return {"num_workers": 3, "prefetch_factor": 4}

    monkeypatch.setattr(orchestrator_module, "LoaderTuner", _FakeTuner)

    pretrain = _pretrain(find_batch_size=True, tune_loader=True)

    PretrainOrchestrator(
        pretrain_config = pretrain,
        training_config = training,
        build_context   = lambda: _context(order, training),
        logger          = _Logger(),
        result_dir      = tmp_path,
        label           = "m",
    ).run()

    assert training.batch_size      == 4
    assert training.num_workers     == 3
    assert training.prefetch_factor == 4
    assert pretrain.find_batch_size is False
    assert pretrain.tune_loader     is False

    sweep = pd.read_csv(tmp_path / PretrainOrchestrator.LOADER_SWEEP)

    assert list(sweep["status"])      == ["ok", "oom"]
    assert list(sweep["num_workers"]) == [3, 6]

    tune_index = next(i for i, (kind, _) in enumerate(order) if kind == "tune")

    assert order[tune_index] == ("tune", 4)
    assert any(kind == "trial" for kind, _ in order[:tune_index])


def test_batch_finder_caps_ceiling_at_dataset_size(tmp_path):
    """Verifies that the search ceiling never exceeds the dataset length."""
    order    = []
    training = _training()

    PretrainOrchestrator(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = training,
        build_context   = lambda: _context(order, training, dataset=list(range(3))),
        logger          = _Logger(),
        result_dir      = tmp_path,
        label           = "m",
    ).run()

    assert training.batch_size == 2
    assert [size for kind, size in order if kind == "trial"] == [1, 2]


def test_all_flags_off_is_a_noop(tmp_path):
    """Verifies that the orchestrator runs no trials when every flag is off."""
    order    = []
    training = _training()

    PretrainOrchestrator(
        pretrain_config = _pretrain(),
        training_config = training,
        build_context   = lambda: _context(order, training),
        logger          = _Logger(),
        result_dir      = tmp_path,
    ).run()

    assert order == []
    assert training.batch_size  == 1
    assert training.num_workers == 0


def test_lr_probe_runs_at_the_resolved_batch_and_scales_the_multiplier(monkeypatch, tmp_path):
    """Verifies that the LR test sees the resolved batch size and multiplies lr_multiplier."""
    order    = []
    training = _training()

    training.lr_multiplier = 2.0

    class _FakeRangeTest:
        """Range-test stub recording the batch size it was probed at."""
        def __init__(self, **kwargs):
            """Accepts and discards the range-test keyword arguments."""
            pass

        def run(self, batch_size):
            """Records the probe call and returns a fixed selection."""
            order.append(("lr", batch_size))
            self.record = {"multiplier": 0.5, "losses": [1.0, 0.5]}
            return {"multiplier": 0.5, "min_loss_multiplier": 5.0, "divergence_multiplier": 20.0}

    monkeypatch.setattr(orchestrator_module, "LrRangeTest", _FakeRangeTest)

    pretrain = _pretrain(find_batch_size=True, find_lr=True)

    PretrainOrchestrator(
        pretrain_config = pretrain,
        training_config = training,
        build_context   = lambda: _context(order, training, trainer=object()),
        logger          = _Logger(),
        result_dir      = tmp_path,
        label           = "m",
    ).run()

    assert training.batch_size    == 4
    assert training.lr_multiplier == 2.0 * 0.5
    assert order[-1]              == ("lr", 4)
    assert pretrain.find_lr is False

    record = json.loads((tmp_path / PretrainOrchestrator.LR_RECORD).read_text())

    assert record["losses"]                   == [1.0, 0.5]
    assert record["lr_multiplier_configured"] == 2.0
    assert record["lr_multiplier_resolved"]   == 1.0


def test_lr_probe_keeps_the_multiplier_when_nothing_is_found(monkeypatch, tmp_path):
    """Verifies that a None selection leaves lr_multiplier untouched."""
    order    = []
    training = _training()

    class _EmptyRangeTest:
        """Range-test stub that never finds a stable multiplier."""
        def __init__(self, **kwargs):
            """Accepts and discards the range-test keyword arguments."""
            pass

        def run(self, batch_size):
            """Reports no selection."""
            self.record = {"multiplier": None, "losses": [1.0, 1.0]}
            return None

    monkeypatch.setattr(orchestrator_module, "LrRangeTest", _EmptyRangeTest)

    pretrain = _pretrain(find_lr=True)

    PretrainOrchestrator(
        pretrain_config = pretrain,
        training_config = training,
        build_context   = lambda: _context(order, training, trainer=object()),
        logger          = _Logger(),
        result_dir      = tmp_path,
    ).run()

    assert training.lr_multiplier == 1.0
    assert pretrain.find_lr is False

    record = json.loads((tmp_path / PretrainOrchestrator.LR_RECORD).read_text())

    assert record["multiplier"]               is None
    assert record["lr_multiplier_configured"] == record["lr_multiplier_resolved"] == 1.0


def test_loader_tuner_without_a_result_still_switches_itself_off(monkeypatch, tmp_path):
    """Verifies a sweep that found no configuration leaves the loader settings alone but records that it ran."""
    order    = []
    training = _training()

    class _EmptyTuner:
        """Loader tuner stub that never finds a configuration."""
        def __init__(self, **kwargs):
            """Accepts and discards the tuner keyword arguments."""
            pass

        def run(self, batch_size):
            """Reports no choice."""
            self.records = [{"phase": "workers", "num_workers": 0, "prefetch_factor": 0, "status": "oom"}]
            return None

    monkeypatch.setattr(orchestrator_module, "LoaderTuner", _EmptyTuner)

    pretrain = _pretrain(tune_loader=True)

    PretrainOrchestrator(
        pretrain_config = pretrain,
        training_config = training,
        build_context   = lambda: _context(order, training),
        logger          = _Logger(),
        result_dir      = tmp_path,
    ).run()

    assert training.num_workers  == 0
    assert pretrain.tune_loader is False
    assert list(pd.read_csv(tmp_path / PretrainOrchestrator.LOADER_SWEEP)["status"]) == ["oom"]


def test_context_requires_a_trainer():
    """Verifies that a probe context cannot be built without its trainer."""
    with pytest.raises(TypeError):
        PretrainContext(
            dataset        = [],
            model          = None,
            to_model_input = None,
            forward_loss   = None,
            trial_step     = lambda batch_size: 1.0,
            device         = torch.device("cpu"),
        )


def test_batch_finder_failure_raises_a_domain_error(monkeypatch, tmp_path):
    """Verifies that a failed batch search is re-raised as a RuntimeError."""
    order    = []
    training = _training()

    class _FailingFinder:
        """Batch finder stub that always reports a failed search."""
        def __init__(self, **kwargs):
            """Accepts and discards the finder keyword arguments."""
            pass

        def run(self):
            """Returns a FAIL result carrying an error message."""
            return {"status": "FAIL", "error": "every trial went OOM"}

    monkeypatch.setattr(orchestrator_module, "BatchSizeFinder", _FailingFinder)

    with pytest.raises(RuntimeError, match="every trial went OOM"):
        PretrainOrchestrator(
            pretrain_config = _pretrain(find_batch_size=True),
            training_config = training,
            build_context   = lambda: _context(order, training),
            logger          = _Logger(),
            result_dir      = tmp_path,
            label           = "m",
        ).run()
