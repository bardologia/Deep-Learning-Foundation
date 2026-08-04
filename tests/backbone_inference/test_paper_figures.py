from __future__ import annotations

import json
from pathlib import Path
from types   import SimpleNamespace

import pytest

from configuration.diagnostics                           import PaperFigurePackConfig
from pipelines.backbone.inference.analysis.paper_figures import PaperFigurePack

from tests.conftest import SilentLogger


def _run_with_figures(base: Path, name: str, stamp: str = "20260101_000000") -> Path:
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
    return PaperFigurePackConfig(
        runs_dir   = base,
        run_filter = runs,
        patterns   = ["pixel_maps/*.png", "profiles/*.png"],
        output_dir = out,
    )


def test_pack_copies_stable_names_and_writes_a_manifest(tmp_path, monkeypatch):
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
    run = _run_with_figures(tmp_path / "runs", "run_a")
    (run / "inference" / "20260101_000000" / "figures" / "profiles" / "best_02.pdf").write_bytes(b"pdf-prof")

    out    = tmp_path / "figures"
    config = _config(tmp_path / "runs", out, ["run_a"])

    config.patterns = ["pixel_maps/*", "profiles/*"]

    payload = PaperFigurePack(config, SilentLogger()).run()

    assert (out / "run_a__profiles__best_02.pdf").read_bytes() == b"pdf-prof"
    assert len(payload["figures"]) == 4


def test_pack_raises_when_one_pattern_matches_nothing(tmp_path):
    _run_with_figures(tmp_path / "runs", "run_a")
    config          = _config(tmp_path / "runs", tmp_path / "figures", ["run_a"])
    config.patterns = ["pixel_maps/*", "stratified/*"]

    with pytest.raises(FileNotFoundError):
        PaperFigurePack(config, SilentLogger()).run()


def test_pack_without_matches_raises(tmp_path):
    _run_with_figures(tmp_path / "runs", "run_a")
    config          = _config(tmp_path / "runs", tmp_path / "figures", ["run_a"])
    config.patterns = ["does_not_exist/*.png"]

    with pytest.raises(FileNotFoundError):
        PaperFigurePack(config, SilentLogger()).run()


def test_pack_uses_latest_inference_stamp(tmp_path):
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
