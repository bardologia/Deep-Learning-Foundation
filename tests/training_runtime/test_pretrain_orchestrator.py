
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import tools.training.pretraining.orchestrator as orchestrator_module
from tools.training.pretraining.orchestrator import PretrainContext, PretrainOrchestrator


class _Logger:
    def section(self, *args, **kwargs):
        pass

    def subsection(self, *args, **kwargs):
        pass

    def kv_table(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def metrics_table(self, *args, **kwargs):
        pass


def _pretrain(**overrides):
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
    return SimpleNamespace(batch_size=1, num_workers=0, prefetch_factor=2, scale_lr_with_batch=True, lr_multiplier=1.0)


def _context(order, training, dataset=None, trainer=None):
    def trial(batch_size):
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
    order    = []
    training = _training()

    class _FakeTuner:
        def __init__(self, **kwargs):
            pass

        def run(self, batch_size):
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


def test_batch_finder_caps_ceiling_at_dataset_size(tmp_path, recording_logger):
    order    = []
    training = _training()

    PretrainOrchestrator(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = training,
        build_context   = lambda: _context(order, training, dataset=list(range(3))),
        logger          = recording_logger,
        result_dir      = tmp_path,
        label           = "m",
    ).run()

    assert training.batch_size == 2
    assert [size for kind, size in order if kind == "trial"] == [1, 2]
    assert ("warning", "Ceiling lowered from 8 to 3: the dataset holds 3 samples, so the batch-size finder cannot try the configured maximum") in recording_logger.messages


def test_all_flags_off_is_a_noop(tmp_path):
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
    order    = []
    training = _training()

    training.lr_multiplier = 2.0

    class _FakeRangeTest:
        def __init__(self, **kwargs):
            pass

        def run(self, batch_size):
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
    order    = []
    training = _training()

    class _EmptyRangeTest:
        def __init__(self, **kwargs):
            pass

        def run(self, batch_size):
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
    order    = []
    training = _training()

    class _EmptyTuner:
        def __init__(self, **kwargs):
            pass

        def run(self, batch_size):
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
    order    = []
    training = _training()

    class _FailingFinder:
        def __init__(self, **kwargs):
            pass

        def run(self):
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
