"""Tests for learning rate warmup and epoch-wise schedulers.

Covers the linear, cosine, polynomial and exponential warmup shapes, the
cosine, constant, linear, polynomial, exponential and step schedules, and
the combination of warmup factor with the current scheduled rates.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tools.training.scheduling import Scheduler, Warmup


def _warmup_config(enabled=True, fraction=0.1, start_factor=0.1, mode="linear", poly_power=2.0):
    """Returns a namespace holding a warmup section with the given settings."""
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
    """Returns a resolved Warmup whose ramp spans `steps` of `total_steps` optimizer steps."""
    warmup = Warmup(_warmup_config(enabled=enabled, fraction=steps / total_steps, start_factor=start_factor, mode=mode, poly_power=poly_power), logger)
    warmup.resolve(total_steps)
    return warmup


def _scheduler_config(stype="cosine_annealing", epochs=10, eta_min=0.0, step_size=30, gamma=0.1, power=1.0):
    """Returns a namespace holding a scheduler section with the given settings."""
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
    """Verifies that a disabled warmup reports a factor of one and is finished."""
    warmup = _warmup(logger, enabled=False)
    assert warmup.factor() == 1.0
    assert warmup.step()   == 1.0
    assert warmup.is_finished() is True


def test_warmup_linear_factor_progression(logger):
    """Verifies that the linear factor runs from 0 to 1 over the warmup steps."""
    warmup = _warmup(logger, steps=10, start_factor=0.0, mode="linear")

    warmup.current_step = 0
    assert warmup.factor() == pytest.approx(0.0)

    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.5)

    warmup.current_step = 10
    assert warmup.factor() == pytest.approx(1.0)


def test_warmup_linear_respects_start_factor(logger):
    """Verifies that the linear factor interpolates from the start factor to one."""
    warmup              = _warmup(logger, steps=4, start_factor=0.2, mode="linear")
    warmup.current_step = 2
    assert warmup.factor() == pytest.approx(0.2 + 0.8 * 0.5)


def test_warmup_cosine_factor(logger):
    """Verifies that the cosine factor matches the raised-cosine interpolation."""
    warmup              = _warmup(logger, steps=8, start_factor=0.1, mode="cosine")
    warmup.current_step = 4
    progress            = 0.5
    cos_factor          = (1.0 - math.cos(math.pi * progress)) / 2.0
    expected            = 0.1 + 0.9 * cos_factor
    assert warmup.factor() == pytest.approx(expected)


def test_warmup_polynomial_factor(logger):
    """Verifies that the polynomial factor is progress raised to the configured power."""
    warmup              = _warmup(logger, steps=10, start_factor=0.0, mode="polynomial", poly_power=2.0)
    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.5 ** 2.0)


def test_warmup_exponential_factor(logger):
    """Verifies that the exponential factor is the start factor raised to 1 - progress."""
    warmup              = _warmup(logger, steps=10, start_factor=0.01, mode="exponential")
    warmup.current_step = 5
    assert warmup.factor() == pytest.approx(0.01 ** 0.5)


def test_warmup_factor_one_after_completion(logger):
    """Verifies that the factor is one once the step count passes the warmup length."""
    warmup              = _warmup(logger, steps=10, mode="cosine")
    warmup.current_step = 11
    assert warmup.factor() == 1.0


def test_warmup_step_increments_and_finishes(logger):
    """Verifies that stepping advances the factor and latches the finished flag."""
    warmup = _warmup(logger, steps=3, start_factor=0.0, mode="linear")

    assert warmup.step() == pytest.approx(1.0 / 3.0)
    assert warmup.step() == pytest.approx(2.0 / 3.0)
    last = warmup.step()
    assert last == pytest.approx(1.0)
    assert warmup.warmup_finished is True
    assert warmup.step()          == 1.0


def test_warmup_reset_restores_initial_state(logger):
    """Verifies that reset clears the step count, finished flag and log latch."""
    warmup = _warmup(logger, steps=3, mode="linear")
    warmup.step()
    warmup.step()
    warmup.reset()

    assert warmup.current_step       == 0
    assert warmup.warmup_finished    is False
    assert warmup._logged_completion is False


def test_warmup_resolve_converts_fraction_to_steps(logger):
    """Verifies that resolve rounds the fraction of the total steps into the ramp length."""
    warmup = Warmup(_warmup_config(fraction=0.03), logger)
    warmup.resolve(1000)
    assert warmup.warmup_steps == 30


def test_warmup_step_raises_when_unresolved(logger):
    """Verifies that stepping an active but unresolved ramp raises."""
    warmup = Warmup(_warmup_config(fraction=0.1), logger)
    with pytest.raises(RuntimeError, match="resolve"):
        warmup.step()


def test_warmup_unresolved_factor_is_start_factor(logger):
    """Verifies that the factor before resolution equals the start-of-ramp factor."""
    warmup = Warmup(_warmup_config(fraction=0.1, start_factor=0.2), logger)
    assert warmup.factor() == pytest.approx(0.2)


def test_warmup_zero_fraction_is_inactive(logger):
    """Verifies that a zero fraction behaves as no warmup at all."""
    warmup = Warmup(_warmup_config(fraction=0.0), logger)
    assert warmup.factor()      == 1.0
    assert warmup.step()        == 1.0
    assert warmup.is_finished() is True


def test_warmup_tiny_fraction_finishes_at_resolution(logger):
    """Verifies that a fraction rounding to zero steps latches the finished flag on resolve."""
    warmup = Warmup(_warmup_config(fraction=0.001), logger)
    warmup.resolve(10)
    assert warmup.warmup_steps  == 0
    assert warmup.is_finished() is True
    assert warmup.factor()      == 1.0


def test_scheduler_cosine_endpoints(logger):
    """Verifies that cosine annealing starts at the base rate and ends at eta_min."""
    config = _scheduler_config("cosine_annealing", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.01], warmup=None, config=config, logger=logger)

    start = sched.step(0)
    assert start[0] == pytest.approx(0.01)

    end = sched.step(10)
    assert end[0] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_cosine_midpoint(logger):
    """Verifies that cosine annealing reaches half the base rate at the midpoint."""
    config = _scheduler_config("cosine_annealing", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.02], warmup=None, config=config, logger=logger)

    mid = sched.step(5)
    assert mid[0] == pytest.approx(0.01)


def test_scheduler_cosine_respects_eta_min(logger):
    """Verifies that cosine annealing floors at a non-zero eta_min."""
    base    = 0.1
    eta_min = 0.01
    config  = _scheduler_config("cosine_annealing", epochs=10, eta_min=eta_min)
    sched   = Scheduler([base], warmup=None, config=config, logger=logger)

    end = sched.step(10)
    assert end[0] == pytest.approx(eta_min)


def test_scheduler_constant_returns_base(logger):
    """Verifies that the constant schedule returns the base rates at every epoch."""
    config = _scheduler_config("constant", epochs=10)
    sched  = Scheduler([0.05, 0.1], warmup=None, config=config, logger=logger)

    for epoch in (0, 3, 9):
        lrs = sched.step(epoch)
        assert lrs == [0.05, 0.1]


def test_scheduler_linear_endpoints_and_midpoint(logger):
    """Verifies that the linear schedule halves at the midpoint and ends at zero."""
    config = _scheduler_config("linear", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(5)[0]  == pytest.approx(0.05)
    assert sched.step(10)[0] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_linear_respects_eta_min(logger):
    """Verifies that the linear schedule ends at a non-zero eta_min."""
    config = _scheduler_config("linear", epochs=10, eta_min=0.01)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)
    assert sched.step(10)[0] == pytest.approx(0.01)


def test_scheduler_polynomial_factor(logger):
    """Verifies that the polynomial schedule scales by progress to the given power."""
    config = _scheduler_config("polynomial", epochs=10, eta_min=0.0, power=2.0)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(5)[0]  == pytest.approx(0.1 * (0.5 ** 2.0))
    assert sched.step(10)[0] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_exponential_endpoints(logger):
    """Verifies that the exponential schedule decays geometrically to eta_min."""
    config = _scheduler_config("exponential", epochs=10, eta_min=0.001)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(5)[0]  == pytest.approx(0.1 * ((0.001 / 0.1) ** 0.5))
    assert sched.step(10)[0] == pytest.approx(0.001)


def test_scheduler_step_decay(logger):
    """Verifies that the step schedule multiplies by gamma every step_size epochs."""
    config = _scheduler_config("step", epochs=100, eta_min=0.0, step_size=10, gamma=0.5)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    assert sched.step(0)[0]  == pytest.approx(0.1)
    assert sched.step(9)[0]  == pytest.approx(0.1)
    assert sched.step(10)[0] == pytest.approx(0.05)
    assert sched.step(20)[0] == pytest.approx(0.025)


def test_scheduler_step_floors_at_eta_min(logger):
    """Verifies that step decay never falls below eta_min."""
    config = _scheduler_config("step", epochs=100, eta_min=0.05, step_size=10, gamma=0.5)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)
    assert sched.step(50)[0] == pytest.approx(0.05)


def test_scheduler_unknown_type_raises(logger):
    """Verifies that an unknown scheduler type raises ValueError."""
    config = _scheduler_config("nonexistent", epochs=10)
    sched  = Scheduler([0.05], warmup=None, config=config, logger=logger)

    with pytest.raises(ValueError):
        sched.step(0)


def test_scheduler_reset_applies_epoch_offset(logger):
    """Verifies that reset shifts the schedule so the offset epoch reads as epoch zero."""
    config = _scheduler_config("cosine_annealing", epochs=10, eta_min=0.0)
    sched  = Scheduler([0.02], warmup=None, config=config, logger=logger)

    sched.reset(epoch_offset=5)
    lrs = sched.step(5)
    assert lrs[0] == pytest.approx(0.02)


def test_effective_lrs_applies_warmup_factor(logger):
    """Verifies that effective rates multiply the scheduled rates by the warmup factor."""
    warmup              = _warmup(logger, steps=10, start_factor=0.0, mode="linear")
    warmup.current_step = 5
    config              = _scheduler_config("constant", epochs=10)
    sched               = Scheduler([0.1], warmup=warmup, config=config, logger=logger)

    sched.step(0)
    eff = sched.effective_lrs()
    assert eff[0] == pytest.approx(0.05)


def test_effective_lrs_no_warmup_returns_current(logger):
    """Verifies that effective rates equal the scheduled rates when no warmup is set."""
    config = _scheduler_config("constant", epochs=10)
    sched  = Scheduler([0.1], warmup=None, config=config, logger=logger)

    sched.step(0)
    assert sched.effective_lrs() == [0.1]
