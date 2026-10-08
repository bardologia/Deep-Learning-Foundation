"""Tests for the parameter-extraction trial comparison.

Covers the refusal of requested run tags that are not completed fit runs and
the sections and per-trial completion lines the measurement and plotting
stages log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.comparison                         import ParamExtractionComparisonConfig
from pipelines.comparison.param_extraction_comparison import ParamExtractionComparisonPipeline, ParamMetrics, ParamTrialCollector

from tests.conftest import RecordingLogger


def _write_fit_run(run_dir: Path, k_max: int = 2, parameters: bool = True) -> Path:
    """Writes the extraction metadata of a fitted parameter run and, by default, its parameter stack.

    Args:
        run_dir    : Parameter run directory to create.
        k_max      : Gaussian order recorded in the metadata.
        parameters : Whether the parameter stack is written, False for a run that stopped before it.

    Returns:
        Path of the run directory.
    """
    run_dir.mkdir(parents=True)

    meta = {"k_max": k_max, "lambda_k": 0.1, "sigma_init_divisor": 4.0, "fit_sigma": True, "fit_amplitude": True, "fit_mean": True}
    (run_dir / "param_extraction_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    if parameters:
        (run_dir / "parameters.npy").write_bytes(b"\x00" * 64)

    return run_dir


def test_requested_tags_that_are_not_fit_runs_raise_naming_the_missing_files(tmp_path):
    """Verifies a half-written run and a missing run abort the comparison together, each named with what it lacks."""
    _write_fit_run(tmp_path / "ds" / "params" / "fit_ok")
    _write_fit_run(tmp_path / "ds" / "params" / "fit_half", parameters=False)

    collector = ParamTrialCollector(tmp_path, ["ds/params/fit_ok", "ds/params/fit_half", "ds/params/gone"], RecordingLogger())

    with pytest.raises(FileNotFoundError) as excinfo:
        collector.collect()

    message = str(excinfo.value)

    assert "ds/params/fit_half: no parameters.npy" in message
    assert "ds/params/gone: no param_extraction_meta.json or parameters.npy" in message
    assert "fit_ok" not in message


def test_auto_discovery_offers_only_completed_fit_runs(tmp_path):
    """Verifies discovery leaves a run without its parameter stack out instead of aborting."""
    _write_fit_run(tmp_path / "ds" / "params" / "fit_ok")
    _write_fit_run(tmp_path / "ds" / "params" / "fit_half", parameters=False)

    trials = ParamTrialCollector(tmp_path, [], RecordingLogger()).collect()

    assert [trial.name for trial in trials] == ["ds/params/fit_ok"]


def _pipeline(tmp_path: Path, logger: RecordingLogger, **overrides) -> ParamExtractionComparisonPipeline:
    """Returns a pipeline over an empty parameters root with the given config overrides."""
    config = ParamExtractionComparisonConfig(params_dir=tmp_path, run_tags=[], make_plots=False, output_dir=tmp_path / "out", **overrides)
    return ParamExtractionComparisonPipeline(config=config, out_dir=tmp_path / "out", logger=logger)


def test_measurement_opens_a_section_and_logs_each_trial_with_its_duration(tmp_path, monkeypatch):
    """Verifies the measurement stage runs under its own section and reports every trial with a counter and its duration."""
    monkeypatch.setattr(ParamMetrics, "compute", lambda self, trial: {"r2_median": 0.5})

    _write_fit_run(tmp_path / "ds" / "params" / "fit_a")
    _write_fit_run(tmp_path / "ds" / "params" / "fit_b")

    logger = RecordingLogger()
    trials = ParamTrialCollector(tmp_path, [], RecordingLogger()).collect()

    _pipeline(tmp_path, logger)._measure(trials)

    sections = [message for level, message in logger.messages if level == "section"]
    oks      = [message for level, message in logger.messages if level == "ok"]

    assert sections == ["Measuring trials"]
    assert [message[:5] for message in oks] == ["[1/2]", "[2/2]"]
    assert all("measured in" in message for message in oks)
    assert all(trial.metrics == {"r2_median": 0.5} for trial in trials)
