"""Tests for collecting inference figures of several runs into a paper figure pack.

The pack copies matched figures under stable ``run__group__name`` filenames, writes a
manifest, takes the most recent inference stamp of each run, and refuses to run when any
requested glob pattern matches nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types   import SimpleNamespace

import pytest

from configuration.diagnostics                           import PaperFigurePackConfig
from pipelines.backbone.inference.analysis.paper_figures import PaperFigurePack

from tests.conftest import SilentLogger


def _run_with_figures(base: Path, name: str, stamp: str = "20260101_000000") -> Path:
    """Creates a run directory holding one inference stamp with pixel-map and profile figures.

    Args:
        base  : Runs directory the run is created under.
        name  : Name of the run directory.
        stamp : Timestamp of the inference output directory.

    Returns:
        Path of the created run directory.
    """
    run     = base / name
    figures = run / "inference" / stamp / "figures"

    (figures / "pixel_maps").mkdir(parents=True)
    (figures / "profiles").mkdir(parents=True)
    (run / "inference" / stamp / "metrics.json").write_text("{}", encoding="utf-8")
    (run / "inference" / stamp / "report.md").write_text("# r", encoding="utf-8")

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

    payload = PaperFigurePack(_config(tmp_path / "runs", out, ["run_a", "run_b"]), SilentLogger()).run()

    assert (out / "run_a__pixel_maps__mse.png").read_bytes() == b"png-mse"
    assert (out / "run_b__profiles__best_01.png").is_file()
    assert len(payload["figures"]) == 6

    manifest = json.loads((out / "figure_manifest.json").read_text(encoding="utf-8"))
    assert {entry["run"] for entry in manifest["figures"]} == {"run_a", "run_b"}


def test_pack_takes_the_vector_figures_of_the_paper_style(tmp_path):
    """An extension-agnostic pattern also picks up the PDF variants of a figure."""
    run = _run_with_figures(tmp_path / "runs", "run_a")
    (run / "inference" / "20260101_000000" / "figures" / "profiles" / "best_02.pdf").write_bytes(b"pdf-prof")

    out    = tmp_path / "figures"
    config = _config(tmp_path / "runs", out, ["run_a"])

    config.patterns = ["pixel_maps/*", "profiles/*"]

    payload = PaperFigurePack(config, SilentLogger()).run()

    assert (out / "run_a__profiles__best_02.pdf").read_bytes() == b"pdf-prof"
    assert len(payload["figures"]) == 4


def test_pack_raises_when_one_pattern_matches_nothing(tmp_path):
    """A pattern group with no matches raises even when the other patterns matched."""
    _run_with_figures(tmp_path / "runs", "run_a")
    config          = _config(tmp_path / "runs", tmp_path / "figures", ["run_a"])
    config.patterns = ["pixel_maps/*", "stratified/*"]

    with pytest.raises(FileNotFoundError):
        PaperFigurePack(config, SilentLogger()).run()


def test_pack_without_matches_raises(tmp_path):
    """A pack whose only pattern matches nothing raises FileNotFoundError."""
    _run_with_figures(tmp_path / "runs", "run_a")
    config          = _config(tmp_path / "runs", tmp_path / "figures", ["run_a"])
    config.patterns = ["does_not_exist/*.png"]

    with pytest.raises(FileNotFoundError):
        PaperFigurePack(config, SilentLogger()).run()


def test_pack_uses_latest_inference_stamp(tmp_path):
    """Figures are taken from the most recent inference stamp of the run."""
    run = _run_with_figures(tmp_path / "runs", "run_a", stamp="20260101_000000")

    late = run / "inference" / "20260601_000000"
    (late / "figures" / "pixel_maps").mkdir(parents=True)
    (late / "metrics.json").write_text("{}", encoding="utf-8")
    (late / "report.md").write_text("# r", encoding="utf-8")
    (late / "figures" / "pixel_maps" / "mse.png").write_bytes(b"png-late")

    out     = tmp_path / "figures"
    config  = _config(tmp_path / "runs", out, ["run_a"])
    config.patterns = ["pixel_maps/*.png"]

    PaperFigurePack(config, SilentLogger()).run()

    assert (out / "run_a__pixel_maps__mse.png").read_bytes() == b"png-late"
