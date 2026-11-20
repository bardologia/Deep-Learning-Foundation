from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.normalization               import ChannelStats, NormalizationConfig, NormMethod
from pipelines.backbone.dataset.normalizer     import Normalizer
from pipelines.backbone.dataset.stats          import Stats
from pipelines.backbone.dataset.stats_computer import StatsComputer

from tests.conftest import SilentLogger


def _stats() -> Stats:
    return Stats(
        input_stats  = ChannelStats(names=["x0", "x1"], method=NormMethod.ZSCORE, loc=[1.0, -2.0], scale=[2.0, 0.5]),
        target_stats = ChannelStats(names=["y"], method=NormMethod.ROBUST_IQR, loc=[10.0], scale=[4.0]),
    )


def test_every_method_fits_its_location_and_scale():
    values = np.arange(1, 1001, dtype=np.float64)

    assert NormMethod.ZSCORE.fit(values)     == pytest.approx((values.mean(), values.std()))
    assert NormMethod.ROBUST_IQR.fit(values) == pytest.approx((500.5, 499.5))
    assert NormMethod.MIN_MAX.fit(values)    == pytest.approx((np.percentile(values, 0.1), np.percentile(values, 99.9) - np.percentile(values, 0.1)))


def test_a_constant_channel_keeps_a_floored_scale():
    assert NormMethod.ZSCORE.fit(np.full(10, 3.0)) == (3.0, 1e-8)


def test_the_normalizer_round_trips_numpy_and_torch():
    normalizer = Normalizer(_stats())
    inputs     = np.random.default_rng(0).normal(size=(2, 4, 4)).astype(np.float32)
    targets    = torch.randn(3, 1, 4, 4)

    normalized = normalizer.normalize_input(inputs)

    assert np.allclose(normalized[0], (inputs[0] - 1.0) / 2.0)
    assert torch.allclose(normalizer.denormalize_target(normalizer.normalize_target(targets)), targets, atol=1e-5)


def test_stats_save_and_load_round_trip(tmp_path):
    _stats().save(tmp_path)

    loaded = Stats.load(tmp_path, SilentLogger())

    assert loaded == _stats()


def test_the_computer_fits_each_channel_on_the_train_crops_only():
    rng   = np.random.default_rng(0)
    crops = [(rng.normal(3.0, 2.0, size=(2, 30, 30)), rng.normal(-1.0, 0.5, size=(1, 30, 30))) for _ in range(2)]

    stats = StatsComputer.compute(crops, ["a", "b"], ["y"], NormalizationConfig(), seed=0, logger=SilentLogger())

    assert stats.input_stats.names  == ["a", "b"]
    assert stats.target_stats.names == ["y"]
    assert stats.input_stats.loc[0]   == pytest.approx(3.0, abs=0.2)
    assert stats.input_stats.scale[0] == pytest.approx(2.0, abs=0.2)
    assert stats.target_stats.loc[0]  == pytest.approx(-1.0, abs=0.1)


def test_the_normalization_config_round_trips_its_payload():
    config = NormalizationConfig(inputs=NormMethod.MIN_MAX, targets=NormMethod.ROBUST_IQR)

    assert NormalizationConfig.from_dict(config.as_dict()) == config
