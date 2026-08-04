from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from configuration.param_extraction import (
    FitMode,
    FitConfig,
    FitSettings,
    ExtractionConfig,
    ExtractParamsEntryConfig,
)


def test_fit_config_defaults_sane():
    cfg = FitConfig()
    assert cfg.k_max > 0
    assert cfg.sigma_init_divisor > 0
    assert cfg.lambda_k > 0
    assert 0.0 < cfg.threshold_factor < 1.0


def test_fit_config_asdict_round_trips():
    cfg     = FitConfig()
    payload = dataclasses.asdict(cfg)
    assert FitConfig(**payload) == cfg


def test_fit_config_default_toggles():
    cfg = FitConfig()
    assert cfg.fit_sigma     is True
    assert cfg.fit_amplitude is False
    assert cfg.fit_mean      is False


def test_free_parameters_default_is_sigma():
    cfg = FitSettings()
    assert cfg.free_parameters == ("sigma",)
    assert cfg.fitting_method  == "sigma_adam"


def test_free_parameters_all_free():
    cfg = FitSettings(fit_config=FitConfig(fit_amplitude=True, fit_mean=True))
    assert cfg.free_parameters == ("sigma", "amp", "mu")
    assert cfg.fitting_method  == "sigma_amp_mu_adam"


def test_free_parameters_sigma_amplitude():
    cfg = FitSettings(fit_config=FitConfig(fit_amplitude=True))
    assert cfg.free_parameters == ("sigma", "amp")
    assert cfg.fitting_method  == "sigma_amp_adam"


def test_free_parameters_sigma_mean():
    cfg = FitSettings(fit_config=FitConfig(fit_mean=True))
    assert cfg.free_parameters == ("sigma", "mu")
    assert cfg.fitting_method  == "sigma_mu_adam"


def test_free_parameters_without_sigma():
    amp_only = FitSettings(fit_config=FitConfig(fit_sigma=False, fit_amplitude=True))
    mu_only  = FitSettings(fit_config=FitConfig(fit_sigma=False, fit_mean=True))
    amp_mu   = FitSettings(fit_config=FitConfig(fit_sigma=False, fit_amplitude=True, fit_mean=True))

    assert amp_only.free_parameters == ("amp",)
    assert amp_only.fitting_method  == "amp_adam"
    assert mu_only.free_parameters  == ("mu",)
    assert mu_only.fitting_method   == "mu_adam"
    assert amp_mu.free_parameters   == ("amp", "mu")
    assert amp_mu.fitting_method    == "amp_mu_adam"


def test_free_parameters_none_free_raises():
    cfg = FitSettings(fit_config=FitConfig(fit_sigma=False))
    with pytest.raises(ValueError):
        cfg.free_parameters
    with pytest.raises(ValueError):
        cfg.fitting_method


def test_entry_config_fit_modes_default_sweep():
    cfg = ExtractParamsEntryConfig()
    assert cfg.fit_modes == ["sigma", "sigma_amp", "sigma_amp_mu"]
    for mode in cfg.fit_modes:
        FitMode.free_flags(mode)


def test_fit_mode_free_flags_all_compositions():
    assert FitMode.free_flags("sigma")        == (True,  False, False)
    assert FitMode.free_flags("amp")          == (False, True,  False)
    assert FitMode.free_flags("mu")           == (False, False, True)
    assert FitMode.free_flags("sigma_amp")    == (True,  True,  False)
    assert FitMode.free_flags("sigma_mu")     == (True,  False, True)
    assert FitMode.free_flags("amp_mu")       == (False, True,  True)
    assert FitMode.free_flags("sigma_amp_mu") == (True,  True,  True)


def test_fit_mode_free_flags_order_insensitive():
    assert FitMode.free_flags("mu_amp")       == FitMode.free_flags("amp_mu")
    assert FitMode.free_flags("mu_amp_sigma") == FitMode.free_flags("sigma_amp_mu")


def test_fit_mode_free_flags_rejects_invalid():
    for mode in ("quartic", "", "sigma_sigma", "sigma_quartic", "sigma+amp"):
        with pytest.raises(ValueError):
            FitMode.free_flags(mode)


def test_fit_settings_default_factory_independent():
    a = FitSettings()
    b = FitSettings()
    assert a.fit_config is not b.fit_config


def test_extraction_config_paths_derived():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"))
    assert cfg.data_directory            == Path("/tmp/run/data")
    assert cfg.metadata_directory        == Path("/tmp/run/meta")
    assert cfg.parameters_npy_path.name  == "parameters.npy"
    assert cfg.diagnostics_npz_path.name == "fit_diagnostics.npz"


def test_extraction_config_default_suffix_encodes_fit():
    cfg    = ExtractionConfig(processed_data_path=Path("/tmp/run"))
    suffix = cfg.output_suffix_value
    assert suffix.startswith(f"k{cfg.fit_settings.fit_config.k_max}_lam")
    assert "_sig" in suffix
    assert suffix.endswith("_sigma")


def test_extraction_config_suffix_encodes_free_parameters():
    sigma_amp    = ExtractionConfig(processed_data_path=Path("/tmp/run"), fit_settings=FitSettings(fit_config=FitConfig(k_max=2, lambda_k=1e-2, sigma_init_divisor=4.0, fit_amplitude=True)))
    sigma_amp_mu = ExtractionConfig(processed_data_path=Path("/tmp/run"), fit_settings=FitSettings(fit_config=FitConfig(k_max=2, lambda_k=1e-2, sigma_init_divisor=4.0, fit_amplitude=True, fit_mean=True)))
    amp_mu       = ExtractionConfig(processed_data_path=Path("/tmp/run"), fit_settings=FitSettings(fit_config=FitConfig(k_max=2, lambda_k=1e-2, sigma_init_divisor=4.0, fit_sigma=False, fit_amplitude=True, fit_mean=True)))

    assert sigma_amp.output_subdir_name    == "params_k2_lam0.01_sig4_sigma_amp"
    assert sigma_amp_mu.output_subdir_name == "params_k2_lam0.01_sig4_sigma_amp_mu"
    assert amp_mu.output_subdir_name       == "params_k2_lam0.01_sig4_amp_mu"


def test_extraction_config_explicit_suffix_used():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"), output_suffix="custom")
    assert cfg.output_suffix_value == "custom"
    assert cfg.output_subdir_name  == "params_custom"


def test_extraction_config_output_directory_layout():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"))
    assert cfg.output_directory.parent.name == "params"
    assert cfg.output_directory.name        == cfg.output_subdir_name


@pytest.mark.real_data
def test_extraction_config_matches_param_extraction_meta_suffix(param_extraction_meta, params_dir):
    fit_config = FitConfig(
        k_max              = param_extraction_meta["k_max"],
        lambda_k           = param_extraction_meta["lambda_k"],
        sigma_init_divisor = param_extraction_meta["sigma_init_divisor"],
        fit_sigma          = param_extraction_meta["fit_sigma"],
        fit_amplitude      = param_extraction_meta["fit_amplitude"],
        fit_mean           = param_extraction_meta["fit_mean"],
    )
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"), fit_settings=FitSettings(fit_config=fit_config))

    assert cfg.output_subdir_name == params_dir.name


def test_extraction_discover_height_range_from_state(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "config_state.json").write_text(json.dumps({"tomogram_config": {"height_range": [-20.0, 80.0]}}))

    cfg = ExtractionConfig(processed_data_path=tmp_path)
    assert cfg.discover_height_range() == (-20.0, 80.0)


def test_extraction_discover_height_range_override():
    cfg = ExtractionConfig(processed_data_path=Path("/tmp/run"), height_range=(0.0, 50.0))
    assert cfg.discover_height_range() == (0.0, 50.0)


def test_extraction_discover_height_range_missing_raises(tmp_path):
    cfg = ExtractionConfig(processed_data_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        cfg.discover_height_range()


def test_extraction_discover_tomogram_missing_raises(tmp_path):
    (tmp_path / "data").mkdir()
    cfg = ExtractionConfig(processed_data_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        cfg.discover_tomogram_path()


def test_extract_params_entry_defaults():
    cfg = ExtractParamsEntryConfig()
    assert cfg.fit_k_values      == [5]
    assert cfg.fit_lambda_values == [1e-2]
    assert cfg.parameter_workers > 0
    assert cfg.range_batch_size > 0
    assert isinstance(cfg.gpu_device_ids, list)
    assert isinstance(cfg.dataset_filter, list)


def test_extract_params_entry_fit_constants_match_fit_config_defaults():
    entry = ExtractParamsEntryConfig()
    fit   = FitConfig()
    ext   = ExtractionConfig(processed_data_path=Path("/tmp/run"))

    assert entry.fit_threshold_factor   == fit.threshold_factor
    assert entry.fit_truncation_index   == fit.truncation_index
    assert entry.fit_prominence_frac    == fit.prominence_frac
    assert entry.fit_activity_threshold == fit.activity_threshold
    assert entry.fit_sigma_init_divisor == fit.sigma_init_divisor
    assert entry.adam_steps             == ext.adam_steps
    assert entry.adam_lr                == ext.adam_lr
    assert entry.gpu_pixel_batch_size   == ext.gpu_pixel_batch_size
    assert entry.range_batch_size       == ext.range_batch_size


def test_extract_params_entry_default_lists_independent():
    a = ExtractParamsEntryConfig()
    b = ExtractParamsEntryConfig()
    a.gpu_device_ids.append(99)
    assert b.gpu_device_ids == [0, 1, 2, 3]
