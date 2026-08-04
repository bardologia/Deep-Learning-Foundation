from __future__ import annotations

import pytest

from pipelines.benchmark.sizing import DegeneracyAuditor, SizeMatcher, SizeMatchResult, WidthRule, WidthScaler
from configuration.benchmark    import BenchmarkConfig


LOCKED = ("embedding_dim", "embedding_dims")


@pytest.fixture
def scaler():
    return WidthScaler(locked=LOCKED)


def test_width_scaler_registers_a_rule_for_every_model(scaler):
    for model_name, rules in scaler.rules.items():
        assert rules


def test_width_scaler_rules_match_the_backbone_registry(scaler):
    from models import BACKBONE_MODEL_REGISTRY

    assert set(scaler.rules) == set(BACKBONE_MODEL_REGISTRY)


def test_width_scaler_resolves_rules_for_head_keys(scaler):
    assert scaler.rules_for("resunet-set_pred") == scaler.rules["resunet"]


def test_width_scaler_scaled_config_carries_the_head(scaler):
    config = scaler.scaled_config("unet-multihead", 0.5)

    assert config.head     == "multihead"
    assert config.features == [32, 64, 128, 256]


def test_width_scaler_rejects_rule_on_locked_parameter():
    with pytest.raises(ValueError):
        WidthScaler(locked=("features",))


def test_width_scaler_round_snaps_to_divisor_multiple(scaler):
    rule = WidthRule(attribute="features", divisor=8)

    assert scaler._round(64, 0.5, rule) == 32
    assert scaler._round(64, 1.0, rule) == 64
    assert scaler._round(64, 2.0, rule) == 128


def test_width_scaler_round_floors_at_divisor(scaler):
    rule = WidthRule(attribute="features", divisor=8)

    assert scaler._round(64, 0.001, rule) == 8


def test_width_scaler_round_preserves_float_for_float_rule(scaler):
    rule   = WidthRule(attribute="mlp_ratio", divisor=0.25, is_float=True)
    result = scaler._round(4.0, 0.5, rule)

    assert result == 2.0
    assert isinstance(result, float)


def test_width_scaler_round_returns_int_for_int_rule(scaler):
    rule   = WidthRule(attribute="features", divisor=8)
    result = scaler._round(64, 0.5, rule)

    assert isinstance(result, int)


def test_width_scaler_scales_list_attribute_elementwise(scaler):
    overrides = scaler.overrides("unet", 0.5)

    assert overrides["features"] == [32, 64, 128, 256]


def test_width_scaler_scale_one_is_identity(scaler):
    overrides = scaler.overrides("unet", 1.0)

    assert overrides["features"] == [64, 128, 256, 512]


def test_width_scaler_unknown_model_raises(scaler):
    with pytest.raises(ValueError):
        scaler.overrides("not_a_model", 1.0)


def test_width_scaler_scaled_config_applies_overrides(scaler):
    config = scaler.scaled_config("unet", 0.5)

    assert config.features == [32, 64, 128, 256]


def _config(dataset_path) -> BenchmarkConfig:
    config                       = BenchmarkConfig()
    config.paths.dataset_path    = dataset_path
    config.paths.parameters_path = dataset_path / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy"
    return config


def test_size_matcher_rejects_in_channels_drift(logger_stub):
    config = BenchmarkConfig()
    config.size_match.in_channels = 7

    with pytest.raises(SystemExit, match="capacity matching would count the wrong width"):
        SizeMatcher(config=config, logger=logger_stub)


@pytest.mark.real_data
def test_size_matcher_reference_count_matches_default_unet(test_data_dir, logger_stub):
    config  = _config(test_data_dir)
    matcher = SizeMatcher(config=config, logger=logger_stub)

    count = matcher.reference_count()

    assert count > 0
    assert count == matcher._count_at("unet", 1.0)


@pytest.mark.real_data
@pytest.mark.slow
def test_size_matcher_matches_resunet_within_tolerance(test_data_dir, logger_stub):
    config  = _config(test_data_dir)
    matcher = SizeMatcher(config=config, logger=logger_stub)
    target  = matcher.reference_count()

    result = matcher.match("resunet", target)

    assert isinstance(result, SizeMatchResult)
    assert result.model      == "resunet"
    assert result.target     == target
    assert result.iterations >= 1
    assert abs(result.deviation_pct) <= 100.0 * config.size_match.tolerance + 1e-6


@pytest.mark.real_data
@pytest.mark.slow
def test_size_matcher_history_records_each_iteration(test_data_dir, logger_stub):
    config  = _config(test_data_dir)
    matcher = SizeMatcher(config=config, logger=logger_stub)
    target  = matcher.reference_count()

    result = matcher.match("resunet", target)

    assert len(result.history) == result.iterations
    for entry in result.history:
        assert set(entry) == {"iteration", "scale", "parameters", "deviation_pct"}


def _fake_matcher(monkeypatch, count_fn):
    matcher         = SizeMatcher.__new__(SizeMatcher)
    matcher.config  = BenchmarkConfig()
    matcher.scaler  = WidthScaler(locked=LOCKED)
    matcher.auditor = DegeneracyAuditor(config=matcher.config.size_match, scaler=matcher.scaler)

    monkeypatch.setattr(matcher, "_count_at", count_fn)

    return matcher


def test_size_matcher_bisection_converges_with_injected_monotone_count(monkeypatch):
    def count_at(model_name, scale):
        return int(1_000_000 * scale)

    matcher = _fake_matcher(monkeypatch, count_at)
    target  = 1_000_000

    result = matcher.match("resunet", target)

    assert abs(result.deviation_pct) <= 100.0 * matcher.config.size_match.tolerance
    assert result.scale == pytest.approx(1.0, abs=0.1)


def test_size_matcher_bisection_walks_search_bounds(monkeypatch):
    seen = []

    def count_at(model_name, scale):
        seen.append(scale)
        return int(1_000_000 * scale)

    matcher = _fake_matcher(monkeypatch, count_at)
    matcher.match("resunet", 1_000_000)

    assert seen[0] == matcher.config.size_match.scale_high

    geometric_mean = (matcher.config.size_match.scale_low * matcher.config.size_match.scale_high) ** 0.5
    assert seen[1] == pytest.approx(geometric_mean)


def test_size_matcher_high_bound_doubles_until_target_reached(monkeypatch):
    high_probes = []

    def count_at(model_name, scale):
        if scale >= matcher.config.size_match.scale_high:
            high_probes.append(scale)
        return int(100 * scale)

    matcher = _fake_matcher(monkeypatch, count_at)
    matcher.match("resunet", 10_000)

    assert any(scale > matcher.config.size_match.scale_high for scale in high_probes)


def test_size_matcher_picks_best_seen_when_tolerance_unreachable(monkeypatch):
    def count_at(model_name, scale):
        return int(1_000_000 * scale) + 70_000

    matcher = _fake_matcher(monkeypatch, count_at)
    matcher.config.size_match.max_iterations = 5

    result = matcher.match("resunet", 1_000_000)

    assert result.iterations == 5
    for entry in result.history:
        assert abs(result.deviation_pct) <= abs(entry["deviation_pct"]) + 1e-9


def _result(scale=1.0, overrides=None, deviation_pct=0.0, iterations=1, parameters=1000):
    return SizeMatchResult(
        model         = "resunet",
        scale         = scale,
        overrides     = overrides if overrides is not None else {"features": [64, 128, 256, 512]},
        parameters    = parameters,
        target        = 1000,
        deviation_pct = deviation_pct,
        iterations    = iterations,
    )


def test_auditor_flags_lower_bound_convergence():
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(scale=config.scale_low))

    assert any("lower search bound" in flag for flag in flags)


def test_auditor_flags_upper_bound_convergence():
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(scale=config.scale_high))

    assert any("upper bound" in flag for flag in flags)


def test_auditor_flags_clamped_width():
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(overrides={"features": [8, 16, 32, 64]}))

    assert any("clamped" in flag for flag in flags)


def test_auditor_flags_excessive_deviation():
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(deviation_pct=50.0))

    assert any("tolerance" in flag for flag in flags)


def test_auditor_clean_result_has_no_flags():
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(scale=1.0, deviation_pct=0.0, overrides={"features": [64, 128, 256, 512]}))

    assert flags == []
