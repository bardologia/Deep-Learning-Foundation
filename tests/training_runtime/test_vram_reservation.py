"""Tests for the VRAM reservation that parks free GPU memory before training.

Covers exposure of the reservation flags on every training flow config, the
inert paths on CPU or when disabled, and the CUDA-only behaviour of parking
memory that later allocations reuse.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from configuration.benchmark.general             import BenchmarkConfig
from configuration.cross_validation.general      import CrossValidationConfig
from configuration.training                      import BackboneEntryConfig, ImageAeEntryConfig, JepaEntryConfig, ProfileAeEntryConfig
from configuration.training.general.trainer      import _SHARED_SUBCONFIGS, SharedSubConfigInheritance
from configuration.tuning.general                import TuningEntryConfig
from tools.runtime.config_cli                    import ConfigCli
from tools.training.vram_reservation             import VramReservation

GB = 1024 ** 3
MB = 1024 ** 2

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_FLOW_CONFIGS = [BackboneEntryConfig, JepaEntryConfig, ProfileAeEntryConfig, ImageAeEntryConfig, BenchmarkConfig, CrossValidationConfig, TuningEntryConfig]


def _free_bytes() -> int:
    """Returns the number of free bytes reported by the CUDA allocator."""
    free, _total = torch.cuda.mem_get_info()
    return free


def _reservation(keep_free_gb: float, logger, enabled: bool = True) -> VramReservation:
    """Returns a CUDA VramReservation keeping `keep_free_gb` gigabytes unclaimed."""
    return VramReservation(enabled=enabled, keep_free_gb=keep_free_gb, device=torch.device("cuda"), logger=logger)


@pytest.mark.parametrize("flow_config", _FLOW_CONFIGS)
def test_reservation_exposed_and_disabled_on_every_training_flow(flow_config):
    """Verifies that every training flow exposes the reservation flags, disabled by default."""
    config = flow_config()
    paths  = dict(ConfigCli._leaves(config))

    assert "training.reserve_vram"      in paths
    assert "training.vram_keep_free_gb" in paths
    assert config.training.reserve_vram is False


def test_memory_is_shared_into_type_trainer_configs():
    """Verifies that the shared memory sub-config is inherited by type-specific configs."""
    class Carrier(SharedSubConfigInheritance):
        """Bare inheritance carrier used to observe the shared sub-configs."""
        pass

    base    = SimpleNamespace(**{name: object() for name in _SHARED_SUBCONFIGS})
    carrier = Carrier()

    carrier.inherit_shared_from(base)

    assert carrier.memory is base.memory


def test_disabled_flag_is_inert(logger):
    """Verifies that a disabled reservation stays disabled after fill."""
    reservation = VramReservation(enabled=False, keep_free_gb=1.0, device=torch.device("cuda"), logger=logger)

    reservation.fill()

    assert reservation.enabled is False


def test_cpu_device_disables_reservation(logger):
    """Verifies that a CPU device disables the reservation on fill."""
    reservation = VramReservation(enabled=True, keep_free_gb=1.0, device=torch.device("cpu"), logger=logger)

    reservation.fill()

    assert reservation.enabled is False


@cuda_only
def test_fill_parks_memory_and_training_allocations_reuse_it(logger):
    """Verifies that fill parks free VRAM and a later allocation reuses the cached blocks."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_before = _free_bytes()
    if free_before < 1 * GB:
        pytest.skip("not enough free VRAM for a safe reservation test")

    keep_free   = (free_before - 384 * MB) / GB
    reservation = _reservation(keep_free, logger)

    reserved_before = torch.cuda.memory_reserved()
    reservation.fill()

    assert _free_bytes() <= keep_free * GB + 64 * MB
    assert torch.cuda.memory_reserved() - reserved_before >= 256 * MB

    free_parked = _free_bytes()
    tensor      = torch.empty(128 * MB, dtype=torch.uint8, device="cuda")

    assert free_parked - _free_bytes() <= 32 * MB

    del tensor
    torch.cuda.empty_cache()


@cuda_only
def test_big_allocations_split_the_reservation_instead_of_dropping_it(logger):
    """Verifies that a large allocation splits the reservation instead of releasing it."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_before = _free_bytes()
    if free_before < 2 * GB:
        pytest.skip("not enough free VRAM for a big-allocation reservation test")

    reservation = _reservation(256 * MB / GB, logger)
    reservation.fill()

    reserved_after_fill = torch.cuda.memory_reserved()
    tensor              = torch.empty(768 * MB, dtype=torch.uint8, device="cuda")

    assert _free_bytes() <= 256 * MB + 64 * MB
    assert torch.cuda.memory_reserved() >= reserved_after_fill - 64 * MB

    del tensor
    torch.cuda.empty_cache()
