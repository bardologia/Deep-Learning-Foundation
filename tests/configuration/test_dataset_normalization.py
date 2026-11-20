from __future__ import annotations

from pathlib import Path

from configuration.dataset       import AugmentationConfig, DatasetConfig, PatchConfig
from configuration.normalization import ChannelStats, NormalizationConfig, NormMethod
from tests.configuration._helpers import make_split_regions


def test_patch_config_defaults():
    cfg = PatchConfig()
    assert cfg.size[0] > 0 and cfg.size[1] > 0
    assert cfg.stride[0] > 0 and cfg.stride[1] > 0
    assert isinstance(cfg.size, tuple)
    assert isinstance(cfg.stride, tuple)


def test_augmentation_config_probabilities_in_range():
    cfg = AugmentationConfig()
    for name in ("p_flip_h", "p_flip_v", "p_rot90", "p_noise"):
        assert 0.0 <= getattr(cfg, name) <= 1.0


def test_dataset_config_needs_only_the_dataset_and_its_splits():
    cfg = DatasetConfig(dataset_directory=Path("/tmp/run"), split_regions=make_split_regions())
    assert cfg.batch_size  > 0
    assert cfg.num_workers >= 0
    assert isinstance(cfg.patch, PatchConfig)
    assert isinstance(cfg.normalization, NormalizationConfig)


def test_norm_method_enum_values():
    assert {m.value for m in NormMethod} == {"zscore", "robust_iqr", "min_max"}


def test_channel_stats_round_trips():
    stats = ChannelStats(names=["a", "b"], method=NormMethod.MIN_MAX, loc=[0.5, -1.0], scale=[2.0, 3.0])

    assert ChannelStats.from_dict(stats.as_dict()) == stats
