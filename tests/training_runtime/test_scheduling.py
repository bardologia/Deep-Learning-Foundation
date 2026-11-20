
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tools.training.scheduling import Scheduler, Warmup


def _warmup_config(enabled=True, fraction=0.1, start_factor=0.1, mode="linear", poly_power=2.0):
    return SimpleNamespace(
        warmup=SimpleNamespace(
            warmup_fraction     = fraction,
            warmup_start_factor = start_factor,
            warmup_enabled      = enabled,
            warmup_mode         = mode,
            warmup_poly_power   = poly_power,
        )
    )


def _warmup(logger, enabled=True, steps=10, start_factor=0.1, mode="linear", poly_power=2.0, total_steps=100):
    warmup = Warmup(_warmup_config(enabled=enabled, fraction=steps / total_steps, start_factor=start_factor, mode=mode, poly_power=poly_power), logger)
    warmup.resolve(total_steps)
    return warmup


def _scheduler_config(stype="cosine_annealing", epochs=10, eta_min=0.0, step_size=30, gamma=0.1, power=1.0):
    return SimpleNamespace(
        scheduler=SimpleNamespace(
            type      = stype,
            epochs    = epochs,
            eta_min   = eta_min,
            step_size = step_size,
            gamma     = gamma,
            power     = power,
        ),
        optimizer=SimpleNamespace(lr_scale=1.0),
    )


def test_warmup_disabled_returns_one(logger):
    warmup = _warmup(logger, enabled=False)
    assert warmup.factor() == 1.0
    assert warmup.step()   == 1.0
    assert warmup.is_finished() is True


def test_warmup_linear_factor_progression(logger):
    warmup = _warmup(logger, steps=10, start_factor=0.0, mode="linear")

    warmup.current_step = 0
    assert warmup.factor() == pytest.approx(0.0)

    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.5)

    warmup.current_step = 10
    assert warmup.factor() == pytest.approx(1.0)


def test_warmup_linear_respects_start_factor(logger):
    warmup              = _warmup(logger, steps=4, start_factor=0.2, mode="linear")
    warmup.current_step = 2
    assert warmup.factor() == pytest.approx(0.2 + 0.8 * 0.5)


def test_warmup_cosine_factor(logger):
    warmup              = _warmup(logger, steps=8, start_factor=0.1, mode="cosine")
    warmup.current_step = 4
    progress            = 0.5
    cos_factor          = (1.0 - math.cos(math.pi * progress)) / 2.0
    expected            = 0.1 + 0.9 * cos_factor
    assert warmup.factor() == pytest.approx(expected)


def test_warmup_polynomial_factor(logger):
    warmup              = _warmup(logger, steps=10, start_factor=0.0, mode="polynomial", poly_power=2.0)
    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.5 ** 2.0)


def test_warmup_exponential_factor(logger):
    warmup              = _warmup(logger, steps=10, start_factor=0.01, mode="exponential")
    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.01 ** 0.5)


def test_warmup_factor_one_after_completion(logger):
    warmup              = _warmup(logger, steps=10, mode="cosine")
    warmup.current_step = 11
    assert warmup.factor() == 1.0


def test_warmup_step_increments_and_finishes(logger):
    warmup = _warmup(logger, steps=3, start_factor=0.0, mode="linear")

    assert warmup.step() == pytest.approx(1.0 / 3.0)
    assert warmup.step() == pytest.approx(2.0 / 3.0)
    last = warmup.step()
    assert last == pytest.approx(1.0)
    assert warmup.warmup_finished is True
    assert warmup.step()          == 1.0


def test_warmup_resolve_converts_fraction_to_steps(logger):
    warmup = Warmup(_warmup_config(fraction=0.03), logger)
    warmup.resolve(1000)
    assert warmup.warmup_steps == 30


def test_warmup_unresolved_factor_is_start_factor(logger):
    warmup = Warmup(_warmup_config(fraction=0.1, start_factor=0.2), logger)
    assert warmup.factor() == pytest.approx(0.2)


def test_warmup_zero_fraction_is_inactive(logger):
    warmup = Warmup(_warmup_config(fraction=0.0), logger)
    assert warmup.factor()      == 1.0
    assert warmup.step()        == 1.0
    assert warmup.is_finished() is True


def test_warmup_tiny_fraction_finishes_at_resolution(logger):
    warmup = Warmup(_warmup_config(fraction=0.001), logger)
    warmup.resolve(10)
    assert warmup.warmup_steps  == 0
    assert warmup.is_finished() is True
    assert warmup.factor()      == 1.0


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


def test_scheduler_unknown_type_fails_at_the_lookup(logger):
    config = _scheduler_config("nonexistent", epochs=10)
    sched  = Scheduler([0.05], warmup=None, config=config, logger=logger)

    with pytest.raises(KeyError, match="nonexistent"):
        sched.step(0)


def test_effective_lrs_applies_warmup_factor(logger):
    warmup              = _warmup(logger, steps=10, start_factor=0.0, mode="linear")
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
