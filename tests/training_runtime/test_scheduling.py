from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tools.training.scheduling import Scheduler, Warmup


def _warmup_config(enabled=True, steps=10, start_factor=0.1, mode="linear", poly_power=2.0):
    return SimpleNamespace(
        warmup=SimpleNamespace(
            warmup_steps        = steps,
            warmup_start_factor = start_factor,
            warmup_enabled      = enabled,
            warmup_mode         = mode,
            warmup_poly_power   = poly_power,
        )
    )


def _scheduler_config(stype="cosine_annealing", epochs=10, eta_min=0.0, step_size=30, gamma=0.1, power=1.0):
    return SimpleNamespace(
        scheduler=SimpleNamespace(
            type      = stype,
            epochs    = epochs,
            eta_min   = eta_min,
            step_size = step_size,
            gamma     = gamma,
            power     = power,
        )
    )


def test_warmup_disabled_returns_one(logger):
    warmup = Warmup(_warmup_config(enabled=False), logger)
    assert warmup.factor() == 1.0
    assert warmup.step()   == 1.0
    assert warmup.is_finished() is True


def test_warmup_linear_factor_progression(logger):
    warmup = Warmup(_warmup_config(steps=10, start_factor=0.0, mode="linear"), logger)

    warmup.current_step = 0
    assert warmup.factor() == pytest.approx(0.0)

    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.5)

    warmup.current_step = 10
    assert warmup.factor() == pytest.approx(1.0)


def test_warmup_linear_respects_start_factor(logger):
    warmup              = Warmup(_warmup_config(steps=4, start_factor=0.2, mode="linear"), logger)
    warmup.current_step = 2
    assert warmup.factor() == pytest.approx(0.2 + 0.8 * 0.5)


def test_warmup_cosine_factor(logger):
    warmup              = Warmup(_warmup_config(steps=8, start_factor=0.1, mode="cosine"), logger)
    warmup.current_step = 4
    progress            = 0.5
    cos_factor          = (1.0 - math.cos(math.pi * progress)) / 2.0
    expected            = 0.1 + 0.9 * cos_factor
    assert warmup.factor() == pytest.approx(expected)


def test_warmup_polynomial_factor(logger):
    warmup              = Warmup(_warmup_config(steps=10, start_factor=0.0, mode="polynomial", poly_power=2.0), logger)
    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.5 ** 2.0)


def test_warmup_exponential_factor(logger):
    warmup              = Warmup(_warmup_config(steps=10, start_factor=0.01, mode="exponential"), logger)
    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.01 ** 0.5)


def test_warmup_factor_one_after_completion(logger):
    warmup              = Warmup(_warmup_config(steps=10, mode="cosine"), logger)
    warmup.current_step = 11
    assert warmup.factor() == 1.0


def test_warmup_step_increments_and_finishes(logger):
    warmup = Warmup(_warmup_config(steps=3, start_factor=0.0, mode="linear"), logger)

    assert warmup.step() == pytest.approx(1.0 / 3.0)
    assert warmup.step() == pytest.approx(2.0 / 3.0)
    last = warmup.step()
    assert last == pytest.approx(1.0)
    assert warmup.warmup_finished is True
    assert warmup.step()          == 1.0


def test_warmup_reset_restores_initial_state(logger):
    warmup = Warmup(_warmup_config(steps=3, mode="linear"), logger)
    warmup.step()
    warmup.step()
    warmup.reset()

    assert warmup.current_step       == 0
    assert warmup.warmup_finished    is False
    assert warmup._logged_completion is False


def test_scheduler_cosine_endpoints(logger):
    config = _scheduler_config("cosine_annealing", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.01], warmup=None, config=config, logger=logger)

    start = sched.step(0)
    assert start[0] == pytest.approx(0.01)

    end = sched.step(10)
    assert end[0] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_cosine_midpoint(logger):
    config = _scheduler_config("cosine_annealing", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.02], warmup=None, config=config, logger=logger)

    mid = sched.step(5)
    assert mid[0] == pytest.approx(0.01)


def test_scheduler_cosine_respects_eta_min(logger):
    base    = 0.1
    eta_min = 0.01
    config  = _scheduler_config("cosine_annealing", epochs=10, eta_min=eta_min)
    sched   = Scheduler([base], warmup=None, config=config, logger=logger)

    end = sched.step(10)
    assert end[0] == pytest.approx(eta_min)


def test_scheduler_constant_returns_base(logger):
    config = _scheduler_config("constant", epochs=10)
    sched  = Scheduler([0.05, 0.1], warmup=None, config=config, logger=logger)

    for epoch in (0, 3, 9):
        lrs = sched.step(epoch)
        assert lrs == [0.05, 0.1]


def test_scheduler_linear_endpoints_and_midpoint(logger):
    config = _scheduler_config("linear", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(5)[0]  == pytest.approx(0.05)
    assert sched.step(10)[0] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_linear_respects_eta_min(logger):
    config = _scheduler_config("linear", epochs=10, eta_min=0.01)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)
    assert sched.step(10)[0] == pytest.approx(0.01)


def test_scheduler_polynomial_factor(logger):
    config = _scheduler_config("polynomial", epochs=10, eta_min=0.0, power=2.0)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(5)[0]  == pytest.approx(0.1 * (0.5 ** 2.0))
    assert sched.step(10)[0] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_exponential_endpoints(logger):
    config = _scheduler_config("exponential", epochs=10, eta_min=0.001)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(5)[0]  == pytest.approx(0.1 * ((0.001 / 0.1) ** 0.5))
    assert sched.step(10)[0] == pytest.approx(0.001)


def test_scheduler_step_decay(logger):
    config = _scheduler_config("step", epochs=100, eta_min=0.0, step_size=10, gamma=0.5)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(9)[0]  == pytest.approx(0.1)
    assert sched.step(10)[0] == pytest.approx(0.05)
    assert sched.step(20)[0] == pytest.approx(0.025)


def test_scheduler_step_floors_at_eta_min(logger):
    config = _scheduler_config("step", epochs=100, eta_min=0.05, step_size=10, gamma=0.5)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)
    assert sched.step(50)[0] == pytest.approx(0.05)


def test_scheduler_unknown_type_raises(logger):
    config = _scheduler_config("nonexistent", epochs=10)
    sched  = Scheduler([0.05], warmup=None, config=config, logger=logger)

    with pytest.raises(ValueError):
        sched.step(0)


def test_scheduler_reset_applies_epoch_offset(logger):
    config = _scheduler_config("cosine_annealing", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.02], warmup=None, config=config, logger=logger)

    sched.reset(epoch_offset=5)
    lrs = sched.step(5)
    assert lrs[0] == pytest.approx(0.02)


def test_effective_lrs_applies_warmup_factor(logger):
    warmup              = Warmup(_warmup_config(steps=10, start_factor=0.0, mode="linear"), logger)
    warmup.current_step = 5
    config              = _scheduler_config("constant", epochs=10)
    sched               = Scheduler([0.1], warmup=warmup, config=config, logger=logger)

    sched.step(0)
    eff = sched.effective_lrs()
    assert eff[0] == pytest.approx(0.05)


def test_effective_lrs_no_warmup_returns_current(logger):
    config = _scheduler_config("constant", epochs=10)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    sched.step(0)
    assert sched.effective_lrs() == [0.1]
