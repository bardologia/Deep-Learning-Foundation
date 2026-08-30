"""Tests for the benchmark capacity-matching machinery.

Covers WidthScaler rule coverage and rounding, the SizeMatcher bisection over
width scale, and the DegeneracyAuditor flags raised on suspect matches.
"""

from __future__ import annotations

import pytest

from pipelines.benchmark.sizing         import DegeneracyAuditor, SizeMatcher, SizeMatchResult, WidthRule, WidthScaler
from configuration.benchmark            import BenchmarkConfig
from configuration.training.general.run import DEFAULT_PARAMS_RUN


LOCKED = ("embedding_dim", "embedding_dims")


@pytest.fixture
def scaler():
    """Returns a WidthScaler with the embedding dimensions locked."""
    return WidthScaler(locked=LOCKED)


def test_width_scaler_registers_a_rule_for_every_model(scaler):
    """Verifies every registered model carries at least one width rule."""
    for model_name, rules in scaler.rules.items():
        assert rules


def test_width_scaler_rules_match_the_backbone_registry(scaler):
    """Verifies the rule keys are exactly the backbone model registry keys."""
    from models import BACKBONE_MODEL_REGISTRY

    assert set(scaler.rules) == set(BACKBONE_MODEL_REGISTRY)


def test_width_scaler_resolves_rules_for_head_keys(scaler):
    """Verifies a model key carrying a head resolves to the base model's rules."""
    assert scaler.rules_for("resunet-set_pred") == scaler.rules["resunet"]


def test_width_scaler_scaled_overrides_carry_the_head(scaler):
    """Verifies a config built from scaled overrides keeps the requested head alongside the scaled widths."""
    config = scaler.config_from_overrides("unet-multihead", scaler.overrides("unet-multihead", 0.5))

    assert config.head     == "multihead"
    assert config.features == [32, 64, 128, 256]


def test_width_scaler_rejects_rule_on_locked_parameter():
    """Verifies locking a parameter that a rule scales is rejected."""
    with pytest.raises(ValueError):
        WidthScaler(locked=("features",))


def test_width_scaler_round_snaps_to_divisor_multiple(scaler):
    """Verifies scaled widths snap to a multiple of the rule divisor."""
    rule = WidthRule(attribute="features", divisor=8)

    assert scaler._round(64, 0.5, rule) == 32
    assert scaler._round(64, 1.0, rule) == 64
    assert scaler._round(64, 2.0, rule) == 128


def test_width_scaler_round_floors_at_divisor(scaler):
    """Verifies a vanishing scale floors the width at the rule divisor."""
    rule = WidthRule(attribute="features", divisor=8)

    assert scaler._round(64, 0.001, rule) == 8


def test_width_scaler_round_preserves_float_for_float_rule(scaler):
    """Verifies a float rule returns a float value."""
    rule   = WidthRule(attribute="mlp_ratio", divisor=0.25, is_float=True)
    result = scaler._round(4.0, 0.5, rule)

    assert result == 2.0
    assert isinstance(result, float)


def test_width_scaler_round_returns_int_for_int_rule(scaler):
    """Verifies an integer rule returns an int value."""
    rule   = WidthRule(attribute="features", divisor=8)
    result = scaler._round(64, 0.5, rule)

    assert isinstance(result, int)


def test_width_scaler_scales_list_attribute_elementwise(scaler):
    """Verifies a list-valued width attribute is scaled element by element."""
    overrides = scaler.overrides("unet", 0.5)

    assert overrides["features"] == [32, 64, 128, 256]


def test_width_scaler_scale_one_is_identity(scaler):
    """Verifies a scale of one reproduces the default widths."""
    overrides = scaler.overrides("unet", 1.0)

    assert overrides["features"] == [64, 128, 256, 512]


def test_width_scaler_unknown_model_raises(scaler):
    """Verifies overrides for an unregistered model are rejected."""
    with pytest.raises(ValueError):
        scaler.overrides("not_a_model", 1.0)


def test_width_scaler_config_from_overrides_applies_overrides(scaler):
    """Verifies a config built from scaled overrides carries the scaled widths."""
    config = scaler.config_from_overrides("unet", scaler.overrides("unet", 0.5))

    assert config.features == [32, 64, 128, 256]


def test_width_scaler_config_from_overrides_rejects_an_attribute_the_config_lost(scaler):
    """Verifies an override naming a field the config no longer has is rejected."""
    with pytest.raises(AttributeError, match="Unknown model override 'renamed_features'"):
        scaler.config_from_overrides("unet", {"renamed_features": [8]})


def _config(dataset_path) -> BenchmarkConfig:
    """Returns a BenchmarkConfig pointed at the dataset and its extracted parameters."""
    config                       = BenchmarkConfig()
    config.paths.dataset_path    = dataset_path
    config.paths.parameters_path = dataset_path / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    return config


def test_size_matcher_rejects_in_channels_drift(logger_stub):
    """Verifies a size-match input-channel count differing from the dataset stack is rejected."""
    config = BenchmarkConfig()
    config.size_match.in_channels = 7

    with pytest.raises(SystemExit, match="capacity matching would count the wrong width"):
        SizeMatcher(config=config, logger=logger_stub)


@pytest.mark.dataset
def test_size_matcher_reference_count_matches_default_unet(test_data_dir, logger_stub):
    """Verifies the reference parameter count equals the unscaled reference model count."""
    config  = _config(test_data_dir)
    matcher = SizeMatcher(config=config, logger=logger_stub)

    count = matcher.reference_count()

    assert count > 0
    assert count == matcher._count_at("unet", 1.0)


@pytest.mark.dataset
@pytest.mark.slow
def test_size_matcher_matches_resunet_within_tolerance(test_data_dir, logger_stub):
    """Verifies matching resunet to the reference count lands inside the configured tolerance."""
    config  = _config(test_data_dir)
    matcher = SizeMatcher(config=config, logger=logger_stub)
    target  = matcher.reference_count()

    result = matcher.match("resunet", target)

    assert isinstance(result, SizeMatchResult)
    assert result.model      == "resunet"
    assert result.target     == target
    assert result.iterations >= 1
    assert abs(result.deviation_pct) <= 100.0 * config.size_match.tolerance + 1e-6


@pytest.mark.dataset
@pytest.mark.slow
def test_size_matcher_history_records_each_iteration(test_data_dir, logger_stub):
    """Verifies the match history holds one fully populated entry per bisection iteration."""
    config  = _config(test_data_dir)
    matcher = SizeMatcher(config=config, logger=logger_stub)
    target  = matcher.reference_count()

    result = matcher.match("resunet", target)

    assert len(result.history) == result.iterations
    for entry in result.history:
        assert set(entry) == {"iteration", "scale", "parameters", "deviation_pct"}


def _fake_matcher(monkeypatch, count_fn):
    """Returns a SizeMatcher whose parameter counting is replaced by count_fn."""
    matcher         = SizeMatcher.__new__(SizeMatcher)
    matcher.config  = BenchmarkConfig()
    matcher.scaler  = WidthScaler(locked=LOCKED)
    matcher.auditor = DegeneracyAuditor(config=matcher.config.size_match, scaler=matcher.scaler)

    monkeypatch.setattr(matcher, "_count_at", count_fn)

    return matcher


def test_size_matcher_counts_each_distinct_width_once(monkeypatch):
    """Verifies parameter counting is cached per distinct width configuration, not per scale."""
    matcher             = SizeMatcher.__new__(SizeMatcher)
    matcher.config      = BenchmarkConfig()
    matcher.scaler      = WidthScaler(locked=LOCKED)
    matcher.count_cache = {}

    built = []

    def count(model_key, model_config):
        """Records the widths built and returns a constant parameter count."""
        built.append(list(model_config.features))
        return 42

    monkeypatch.setattr(matcher, "_count", count)

    assert matcher._count_at("unet", 1.0)   == 42
    assert matcher._count_at("unet", 1.001) == 42
    assert matcher._count_at("unet", 0.5)   == 42

    assert len(built)    == 2
    assert built[0]      != built[1]


def test_size_matcher_bisection_converges_with_injected_monotone_count(monkeypatch):
    """Verifies bisection over a monotone count converges to unit scale within tolerance."""
    def count_at(model_name, scale):
        """Returns a parameter count linear in the width scale."""
        return int(1_000_000 * scale)

    matcher = _fake_matcher(monkeypatch, count_at)
    target  = 1_000_000

    result = matcher.match("resunet", target)

    assert abs(result.deviation_pct) <= 100.0 * matcher.config.size_match.tolerance
    assert result.scale == pytest.approx(1.0, abs=0.1)


def test_size_matcher_bisection_walks_search_bounds(monkeypatch):
    """Verifies the search probes the high bound, the low bound, then their geometric mean."""
    seen = []

    def count_at(model_name, scale):
        """Records the probed scale and returns a count linear in it."""
        seen.append(scale)
        return int(1_000_000 * scale)

    matcher = _fake_matcher(monkeypatch, count_at)
    matcher.match("resunet", 1_000_000)

    assert seen[0] == matcher.config.size_match.scale_high
    assert seen[1] == matcher.config.size_match.scale_low

    geometric_mean = (matcher.config.size_match.scale_low * matcher.config.size_match.scale_high) ** 0.5
    assert seen[2] == pytest.approx(geometric_mean)


def test_size_matcher_high_bound_doubles_until_target_reached(monkeypatch):
    """Verifies the upper search bound is doubled while the target stays out of reach."""
    high_probes = []

    def count_at(model_name, scale):
        """Records probes at or above the configured high bound and returns a count linear in the scale."""
        if scale >= matcher.config.size_match.scale_high:
            high_probes.append(scale)
        return int(100 * scale)

    matcher = _fake_matcher(monkeypatch, count_at)
    matcher.match("resunet", 1_000)

    assert any(scale > matcher.config.size_match.scale_high for scale in high_probes)


def test_size_matcher_rejects_target_above_the_scale_ceiling(monkeypatch):
    """Verifies a target unreachable by widening exits with a matching error."""
    def count_at(model_name, scale):
        """Returns a small parameter count linear in the width scale."""
        return int(100 * scale)

    matcher = _fake_matcher(monkeypatch, count_at)

    with pytest.raises(SystemExit, match="cannot match the target"):
        matcher.match("resunet", 1_000_000)


def test_size_matcher_rejects_target_below_the_lower_bound(monkeypatch):
    """Verifies a target below the smallest reachable count exits with a matching error."""
    def count_at(model_name, scale):
        """Returns an offset parameter count that never drops below the offset."""
        return int(1_000_000 * scale) + 500_000

    matcher = _fake_matcher(monkeypatch, count_at)

    with pytest.raises(SystemExit, match="cannot be shrunk to the target"):
        matcher.match("resunet", 100_000)


def test_size_matcher_picks_best_seen_when_tolerance_unreachable(monkeypatch):
    """Verifies the closest probe seen is returned once the iteration budget is spent."""
    def count_at(model_name, scale):
        """Returns an offset parameter count that cannot reach the target within tolerance."""
        return int(1_000_000 * scale) + 70_000

    matcher = _fake_matcher(monkeypatch, count_at)
    matcher.config.size_match.max_iterations = 5

    result = matcher.match("resunet", 1_000_000)

    assert result.iterations == 5
    for entry in result.history:
        assert abs(result.deviation_pct) <= abs(entry["deviation_pct"]) + 1e-9


def _result(scale=1.0, overrides=None, deviation_pct=0.0, iterations=1, parameters=1000):
    """Returns a SizeMatchResult for resunet with the given scale, overrides and deviation."""
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
    """Verifies convergence at the lower search bound is flagged."""
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(scale=config.scale_low))

    assert any("lower search bound" in flag for flag in flags)


def test_auditor_flags_upper_bound_convergence():
    """Verifies convergence at the upper search bound is flagged."""
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(scale=config.scale_high))

    assert any("upper bound" in flag for flag in flags)


def test_auditor_flags_clamped_width():
    """Verifies widths sitting at the clamp floor are flagged."""
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(overrides={"features": [8, 16, 32, 64]}))

    assert any("clamped" in flag for flag in flags)


def test_auditor_flags_excessive_deviation():
    """Verifies a parameter deviation beyond tolerance is flagged."""
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(deviation_pct=50.0))

    assert any("tolerance" in flag for flag in flags)


def test_auditor_clean_result_has_no_flags():
    """Verifies a clean match at unit scale raises no flags."""
    config  = BenchmarkConfig().size_match
    auditor = DegeneracyAuditor(config=config, scaler=WidthScaler(locked=LOCKED))

    flags = auditor.audit(_result(scale=1.0, deviation_pct=0.0, overrides={"features": [64, 128, 256, 512]}))

    assert flags == []
