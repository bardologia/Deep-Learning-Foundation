"""Tests for collecting inference figures of several runs into a paper figure pack.

The pack copies matched figures under stable ``run__group__name`` filenames, writes a
manifest, takes the most recent inference stamp of each run, and refuses to run when any
requested glob pattern matches nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration.diagnostics                           import PaperFigurePackConfig
from pipelines.backbone.inference.analysis.paper_figures import PaperFigurePack
from tools.runtime.completion                            import CompletionMarker

from tests.conftest import RecordingLogger, SilentLogger


def _stamp(run: Path, stamp: str, split: str | None = "test") -> Path:
    """Writes an inference stamp with a metrics file and a report under a run, completed unless split is None.

    Args:
        run   : Run directory receiving the inference/<stamp> subtree.
        stamp : Timestamp of the inference output directory.
        split : Split recorded in the completion marker, or None to leave the stamp unfinished.

    Returns:
        Path of the stamp's figures directory.
    """
    figures = run / "inference" / stamp / "figures"
    figures.mkdir(parents=True)

    (run / "inference" / stamp / "metrics.json").write_text("{}", encoding="utf-8")
    (run / "inference" / stamp / "report.md").write_text("# r", encoding="utf-8")

    if split is not None:
        CompletionMarker.stamp(run / "inference" / stamp, {"stage": "inference", "split": split})

    return figures


def _run_with_figures(base: Path, name: str, stamp: str = "20260101_000000") -> Path:
    """Creates a run directory holding one completed inference stamp with pixel-map and profile figures.

    Args:
        base  : Runs directory the run is created under.
        name  : Name of the run directory.
        stamp : Timestamp of the inference output directory.

    Returns:
        Path of the created run directory.
    """
    run     = base / name
    figures = _stamp(run, stamp)

    (figures / "pixel_maps").mkdir()
    (figures / "profiles").mkdir()

    (figures / "pixel_maps" / "mse.png").write_bytes(b"png-mse")
    (figures / "pixel_maps" / "r2.png").write_bytes(b"png-r2")
    (figures / "profiles" / "best_01.png").write_bytes(b"png-prof")

    return run


def _config(base: Path, out: Path, runs: list[str]) -> PaperFigurePackConfig:
    """Returns a figure-pack configuration over the given runs directory, run filter and output directory."""
    return PaperFigurePackConfig(
        runs_dir   = base,
        run_filter = runs,
        patterns   = ["pixel_maps/*.png", "profiles/*.png"],
        output_dir = out,
    )


def test_pack_copies_stable_names_and_writes_a_manifest(tmp_path, monkeypatch):
    """Figures of two runs are copied under run-prefixed names and listed in the manifest."""
    _run_with_figures(tmp_path / "runs", "run_a")
    _run_with_figures(tmp_path / "runs", "run_b")
    out = tmp_path / "figures"

    payload = PaperFigurePack(_config(tmp_path / "runs", out, ["run_a", "run_b"]), SilentLogger(), {}).run()

    assert (out / "run_a__pixel_maps__mse.png").read_bytes() == b"png-mse"
    assert (out / "run_b__profiles__best_01.png").is_file()
    assert len(payload["figures"]) == 6

    manifest = json.loads((out / "figure_manifest.json").read_text(encoding="utf-8"))
    assert {entry["run"] for entry in manifest["figures"]} == {"run_a", "run_b"}
    assert {entry["split"] for entry in manifest["figures"]} == {"test"}
    assert all(entry["inference"].endswith("20260101_000000") and entry["completed_at"] for entry in manifest["figures"])


def test_pack_takes_the_vector_figures_of_the_paper_style(tmp_path):
    """An extension-agnostic pattern also picks up the PDF variants of a figure."""
    run = _run_with_figures(tmp_path / "runs", "run_a")
    (run / "inference" / "20260101_000000" / "figures" / "profiles" / "best_02.pdf").write_bytes(b"pdf-prof")

    out    = tmp_path / "figures"
    config = _config(tmp_path / "runs", out, ["run_a"])

    config.patterns = ["pixel_maps/*", "profiles/*"]

    payload = PaperFigurePack(config, SilentLogger(), {}).run()

    assert (out / "run_a__profiles__best_02.pdf").read_bytes() == b"pdf-prof"
    assert len(payload["figures"]) == 4


def test_pack_raises_when_one_pattern_matches_nothing(tmp_path):
    """A pattern group with no matches raises even when the other patterns matched."""
    _run_with_figures(tmp_path / "runs", "run_a")
    config          = _config(tmp_path / "runs", tmp_path / "figures", ["run_a"])
    config.patterns = ["pixel_maps/*", "stratified/*"]

    with pytest.raises(FileNotFoundError):
        PaperFigurePack(config, SilentLogger(), {}).run()


def test_pack_without_matches_raises(tmp_path):
    """A pack whose only pattern matches nothing raises FileNotFoundError."""
    _run_with_figures(tmp_path / "runs", "run_a")
    config          = _config(tmp_path / "runs", tmp_path / "figures", ["run_a"])
    config.patterns = ["does_not_exist/*.png"]

    with pytest.raises(FileNotFoundError):
        PaperFigurePack(config, SilentLogger(), {}).run()


def test_pack_uses_latest_completed_inference_stamp(tmp_path):
    """Figures are taken from the most recent stamp completed on the split, skipping newer unfinished and other-split stamps."""
    run = _run_with_figures(tmp_path / "runs", "run_a", stamp="20260101_000000")

    late = _stamp(run, "20260601_000000")
    (late / "pixel_maps").mkdir()
    (late / "pixel_maps" / "mse.png").write_bytes(b"png-late")

    crashed = _stamp(run, "20260701_000000", split=None)
    (crashed / "pixel_maps").mkdir()
    (crashed / "pixel_maps" / "mse.png").write_bytes(b"png-crashed")

    val = _stamp(run, "20260801_000000", split="val")
    (val / "pixel_maps").mkdir()
    (val / "pixel_maps" / "mse.png").write_bytes(b"png-val")

    out     = tmp_path / "figures"
    config  = _config(tmp_path / "runs", out, ["run_a"])
    config.patterns = ["pixel_maps/*.png"]

    logger = RecordingLogger()
    PaperFigurePack(config, logger, {}).run()

    assert (out / "run_a__pixel_maps__mse.png").read_bytes() == b"png-late"
    assert any(level == "warning" and "20260701_000000" in message for level, message in logger.messages)
    assert any(level == "info" and "20260601_000000" in message and "split 'test'" in message for level, message in logger.messages)
