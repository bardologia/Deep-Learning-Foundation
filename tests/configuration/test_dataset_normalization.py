from __future__ import annotations

import dataclasses

import pytest

from configuration.dataset               import (
    InputConfig,
    OutputConfig,
    PatchConfig,
    AugmentationConfig,
    DatasetConfig,
    ProfileAugmentationConfig,
    ProfileDatasetConfig,
)
from configuration.normalization         import (
    NormMethod,
    ChannelStrategy,
    Presets,
    ChannelStats,
    NormalizationConfig,
    OutputClampConfig,
)
from configuration.normalization.general import _SLOT_STRATEGIES
from configuration.training.general.loss import LossConfig
from tools.data.representation           import Representation
from tests.configuration._helpers        import make_split_regions


def test_input_config_defaults_channel_counts():
    cfg = InputConfig()
    assert cfg.primary_channels_per_pass        >= 0
    assert cfg.secondaries_channels_per_pass    == 0
    assert cfg.interferograms_channels_per_pass >= 0


def test_input_config_total_channels_consistent():
    cfg   = InputConfig(use_primary=True, use_secondaries=True, use_interferograms=True, use_dem=True)
    total = cfg.total_channels(n_secondaries=4, n_interferograms=4)
    keys  = cfg.channel_group_keys(n_secondaries=4, n_interferograms=4)
    assert total == len(keys)


def test_input_config_round_trips_through_dict():
    cfg     = InputConfig(use_secondaries=True, secondaries_representation=Representation.MAG_ANGLE)
    rebuilt = InputConfig.from_dict(cfg.as_dict())
    assert rebuilt == cfg


def test_input_config_dem_adds_one_channel():
    base = InputConfig(use_dem=False)
    dem  = InputConfig(use_dem=True)
    assert dem.total_channels(0, 0) == base.total_channels(0, 0) + 1


def test_output_config_role_names_and_params():
    cfg = OutputConfig()
    assert cfg.role_names          == ["a", "mu", "sig"]
    assert cfg.params_per_gaussian == 3
    assert cfg.total_channels(5)   == 15


def test_output_config_selected_indices():
    cfg = OutputConfig(use_mu=False)
    assert cfg.role_names          == ["a", "sig"]
    assert cfg.selected_indices(2) == [0, 2, 3, 5]


def test_output_config_round_trips_through_dict():
    cfg     = OutputConfig()
    rebuilt = OutputConfig.from_dict(cfg.as_dict())
    assert rebuilt.use_amplitude          == cfg.use_amplitude
    assert set(rebuilt.output_strategies) == set(cfg.output_strategies)


def test_output_config_strategy_for_known_key():
    cfg = OutputConfig()
    assert isinstance(cfg.strategy_for("out/amp"), ChannelStrategy)


def test_presets_by_name_resolves_and_rejects():
    assert Presets.by_name("zscore").norm_method           is NormMethod.ZSCORE
    assert Presets.by_name("robust_iqr_log1p").apply_log1p is True

    with pytest.raises(ValueError):
        Presets.by_name("not_a_preset")


def test_normalization_config_per_slot_matches_slot_defaults():
    cfg = NormalizationConfig()
    assert cfg.input_strategy                == "per_slot"
    assert cfg.output_strategy               == "per_slot"
    assert cfg.strategy("input", "pass/mag") == _SLOT_STRATEGIES["pass/mag"]
    assert cfg.strategy("output", "out/amp") == _SLOT_STRATEGIES["out/amp"]


def test_normalization_config_named_preset_overrides_every_slot():
    cfg = NormalizationConfig(input_strategy="zscore", output_strategy="zscore")
    assert cfg.strategy("input", "pass/mag").norm_method is NormMethod.ZSCORE
    assert cfg.strategy("output", "out/amp").norm_method is NormMethod.ZSCORE


def test_normalization_config_per_channel_override_wins_over_global():
    cfg = NormalizationConfig(output_strategy="per_slot", out_amp="zscore")

    assert cfg.strategy("output", "out/amp").norm_method is NormMethod.ZSCORE
    assert cfg.strategy("output", "out/sigma")           == _SLOT_STRATEGIES["out/sigma"]
    assert cfg.strategy("input", "pass/mag")             == _SLOT_STRATEGIES["pass/mag"]


def test_normalization_config_attaches_bounds_to_fixed_bounds_strategy():
    cfg   = NormalizationConfig(out_mu="fixed_bounds")
    strat = cfg.strategy("output", "out/mu")

    assert strat.norm_method is NormMethod.FIXED_BOUNDS
    assert strat.bounds == (tuple(cfg.fixed_out_bounds_min), tuple(cfg.fixed_out_bounds_max))

    plain = cfg.strategy("output", "out/amp")
    assert plain.bounds is None


def test_fixed_out_bounds_defaults_match_the_loss_legacy_bounds():
    norm = NormalizationConfig()
    loss = LossConfig()

    assert norm.fixed_out_bounds_min == loss.legacy_bounds_min
    assert norm.fixed_out_bounds_max == loss.legacy_bounds_max


def test_normalization_clamp_round_trips_through_dict():
    cfg   = NormalizationConfig(clamp_output=False, clamp_floor=0.0, clamp_ceil=50.0, clamp_leaky_slope=0.2, param_clamp_leaky_slope=0.4, amp_max=500.0)
    clamp = cfg.clamp()

    assert isinstance(clamp, OutputClampConfig)
    assert clamp.enabled           is False
    assert clamp.ceil              == 50.0
    assert clamp.leaky_slope       == 0.2
    assert clamp.param_leaky_slope == 0.4
    assert clamp.amp_max           == 500.0

    rebuilt = OutputClampConfig.from_dict(clamp.as_dict())
    assert rebuilt == clamp


def test_clamp_payload_missing_knobs_breaks_loudly():
    payload = {"enabled": True, "floor": 0.0, "ceil": 1000.0}

    with pytest.raises(KeyError):
        OutputClampConfig.from_dict(payload)


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


def test_dataset_config_requires_run_dir_and_split():
    cfg = DatasetConfig(preprocessing_run_directory="/tmp/run", split_regions=make_split_regions())
    assert cfg.batch_size  > 0
    assert cfg.num_workers >= 0
    assert isinstance(cfg.patch, PatchConfig)
    assert isinstance(cfg.input_config, InputConfig)
    assert isinstance(cfg.output_config, OutputConfig)
    assert cfg.n_gaussians > 0


def test_dataset_config_secondary_labels_default():
    cfg = DatasetConfig(preprocessing_run_directory="/tmp/run", split_regions=make_split_regions())
    assert cfg.secondary_labels == ("FL01_PS04", "FL01_PS06", "FL01_PS08", "FL01_PS26")


def test_profile_augmentation_defaults():
    cfg = ProfileAugmentationConfig()
    assert 0.0 <= cfg.p_amp_scale <= 1.0
    assert cfg.max_shift >= 0
    assert cfg.amp_scale_range[0] <= cfg.amp_scale_range[1]


def test_profile_dataset_config_defaults():
    cfg = ProfileDatasetConfig(preprocessing_run_directory="/tmp/run", split_regions=make_split_regions())
    assert cfg.n_gaussians > 0
    assert cfg.batch_size > 0
    assert 0.0 < cfg.pixel_subsample <= 1.0
    assert isinstance(cfg.augmentation, ProfileAugmentationConfig)


def test_norm_method_enum_values():
    assert {m.value for m in NormMethod} == {
        "min_max_p999", "robust_iqr", "fixed_div_pi", "zscore",
        "fixed_log1p", "fixed_angle_01", "fixed_bounds",
    }


@pytest.mark.parametrize("preset_name", [
    "MIN_MAX", "MIN_MAX_LOG1P", "ROBUST_IQR", "ROBUST_IQR_LOG1P",
    "FIXED_DIV_PI", "ZSCORE", "ZSCORE_LOG1P",
    "FIXED_LOG1P", "FIXED_ANGLE_01", "FIXED_BOUNDS",
])
def test_presets_are_channel_strategies(preset_name):
    preset = getattr(Presets, preset_name)
    assert isinstance(preset, ChannelStrategy)
    assert isinstance(preset.norm_method, NormMethod)


def test_channel_strategy_round_trips_through_dict():
    strat   = ChannelStrategy(NormMethod.ROBUST_IQR, apply_log1p=True)
    rebuilt = ChannelStrategy.from_dict(strat.as_dict())
    assert rebuilt == strat


def test_channel_strategy_from_slot_known_keys():
    for key in _SLOT_STRATEGIES:
        assert isinstance(ChannelStrategy.from_slot(key), ChannelStrategy)


def test_channel_stats_round_trips():
    stats = ChannelStats(
        loc        = [0.0, 1.0],
        scale      = [1.0, 2.0],
        names      = ["c0", "c1"],
        strategies = [Presets.ZSCORE, Presets.ROBUST_IQR],
        clampable  = [True, False],
    )
    rebuilt = ChannelStats.from_dict(stats.as_dict())
    assert rebuilt.n_channels == stats.n_channels
    assert rebuilt.loc        == stats.loc
    assert rebuilt.scale      == stats.scale
    assert rebuilt.names      == stats.names
    assert rebuilt.clampable  == stats.clampable


def test_channel_stats_serialization_requires_clampable():
    import pytest

    stats = ChannelStats(
        loc        = [0.0],
        scale      = [1.0],
        names      = ["c0"],
        strategies = [Presets.ZSCORE],
    )

    with pytest.raises(ValueError, match="clampable"):
        stats.as_dict()

    with pytest.raises(KeyError):
        ChannelStats.from_dict({"channels": [{"name": "c0", "loc": 0.0, "scale": 1.0, "norm_method": "zscore", "apply_log1p": False}]})


def test_factory_output_config_honors_out_slot_overrides():
    from configuration.benchmark import BenchmarkConfig
    from pipelines.shared.config.config_factory import ConfigFactory

    config = BenchmarkConfig()
    config.normalization.out_sigma = "zscore"

    output = ConfigFactory(config)._output_config()
    assert output.strategy_for("out/sigma") == Presets.ZSCORE
    assert output.strategy_for("out/amp")   == ChannelStrategy.from_slot("out/amp")
    assert output.strategy_for("out/mu")    == ChannelStrategy.from_slot("out/mu")


def test_factory_output_config_defaults_match_slots():
    from configuration.benchmark import BenchmarkConfig
    from pipelines.shared.config.config_factory import ConfigFactory

    output = ConfigFactory(BenchmarkConfig())._output_config()
    for key in ("out/amp", "out/mu", "out/sigma"):
        assert output.strategy_for(key) == ChannelStrategy.from_slot(key)
