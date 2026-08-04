from __future__ import annotations

from types import SimpleNamespace

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


def _pretrain(**overrides):
    base = dict(
        find_batch_size  = False,
        tune_loader      = False,
        vram_budget_gb   = 10.0,
        max_batch        = 8,
        measure_steps    = 1,
        worker_counts    = (0,),
        prefetch_factors = (2,),
        warmup_batches   = 1,
        timed_batches    = 1,
        data_wait_target = 0.05,
        seed             = 0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _training():
    return SimpleNamespace(batch_size=1, num_workers=0, prefetch_factor=2, scale_lr_with_batch=True)


def _context(order, training, dataset=None):
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
    )


def test_batch_finder_runs_before_tuner_and_feeds_resolved_batch(monkeypatch):
    order    = []
    training = _training()

    class _FakeTuner:
        def __init__(self, **kwargs):
            pass

        def run(self, batch_size):
            order.append(("tune", batch_size))
            return {"num_workers": 3, "prefetch_factor": 4, "pin_memory": True}

    monkeypatch.setattr(orchestrator_module, "LoaderTuner", _FakeTuner)

    PretrainOrchestrator(
        pretrain_config = _pretrain(find_batch_size=True, tune_loader=True),
        training_config = training,
        build_context   = lambda: _context(order, training),
        logger          = _Logger(),
        label           = "m",
    ).run()

    assert training.batch_size      == 4
    assert training.num_workers     == 3
    assert training.prefetch_factor == 4

    tune_index = next(i for i, (kind, _) in enumerate(order) if kind == "tune")

    assert order[tune_index] == ("tune", 4)
    assert any(kind == "trial" for kind, _ in order[:tune_index])


def test_batch_finder_caps_ceiling_at_dataset_size():
    order    = []
    training = _training()

    PretrainOrchestrator(
        pretrain_config = _pretrain(find_batch_size=True),
        training_config = training,
        build_context   = lambda: _context(order, training, dataset=list(range(3))),
        logger          = _Logger(),
        label           = "m",
    ).run()

    assert training.batch_size == 2
    assert [size for kind, size in order if kind == "trial"] == [1, 2]


def test_all_flags_off_is_a_noop():
    order    = []
    training = _training()

    PretrainOrchestrator(
        pretrain_config = _pretrain(),
        training_config = training,
        build_context   = lambda: _context(order, training),
        logger          = _Logger(),
    ).run()

    assert order == []
    assert training.batch_size  == 1
    assert training.num_workers == 0
