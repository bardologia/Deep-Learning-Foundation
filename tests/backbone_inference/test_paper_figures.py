
from __future__ import annotations

import json
from pathlib import Path

from configuration.diagnostics                           import PaperFigurePackConfig
from pipelines.backbone.inference.analysis.paper_figures import PaperFigurePack
from tools.runtime.completion                            import CompletionMarker

from tests.conftest import RecordingLogger, SilentLogger


def _stamp(run: Path, stamp: str, split: str | None = "test") -> Path:
    figures = run / "inference" / stamp / "figures"
    figures.mkdir(parents=True)

    (run / "inference" / stamp / "metrics.json").write_text("{}", encoding="utf-8")
    (run / "inference" / stamp / "report.md").write_text("# r", encoding="utf-8")

    if split is not None:
        CompletionMarker.stamp(run / "inference" / stamp, {"stage": "inference", "split": split})

    return figures


def _run_with_figures(base: Path, name: str, stamp: str = "20260101_000000") -> Path:
    run     = base / name
    figures = _stamp(run, stamp)

    (figures / "pixel_maps").mkdir()
    (figures / "profiles").mkdir()

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

    payload = PaperFigurePack(_config(tmp_path / "runs", out, ["run_a", "run_b"]), SilentLogger(), {}).run()

    assert (out / "run_a__pixel_maps__mse.png").read_bytes() == b"png-mse"
    assert (out / "run_b__profiles__best_01.png").is_file()
    assert len(payload["figures"]) == 6

    manifest = json.loads((out / "figure_manifest.json").read_text(encoding="utf-8"))
    assert {entry["run"] for entry in manifest["figures"]} == {"run_a", "run_b"}
    assert {entry["split"] for entry in manifest["figures"]} == {"test"}
    assert all(entry["inference"].endswith("20260101_000000") and entry["completed_at"] for entry in manifest["figures"])


def test_pack_takes_the_vector_figures_of_the_paper_style(tmp_path):
    run = _run_with_figures(tmp_path / "runs", "run_a")
    (run / "inference" / "20260101_000000" / "figures" / "profiles" / "best_02.pdf").write_bytes(b"pdf-prof")

    out    = tmp_path / "figures"
    config = _config(tmp_path / "runs", out, ["run_a"])

    config.patterns = ["pixel_maps/*", "profiles/*"]

    payload = PaperFigurePack(config, SilentLogger(), {}).run()

    assert (out / "run_a__profiles__best_02.pdf").read_bytes() == b"pdf-prof"
    assert len(payload["figures"]) == 4


def test_pack_uses_latest_completed_inference_stamp(tmp_path):
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


def test_repacking_warns_about_overwrites_and_keeps_earlier_entries(tmp_path):
    _run_with_figures(tmp_path / "runs", "run_a")
    _run_with_figures(tmp_path / "runs", "run_b")
    out = tmp_path / "figures"

    PaperFigurePack(_config(tmp_path / "runs", out, ["run_a", "run_b"]), SilentLogger(), {}).run()
    (out / "stray.png").write_bytes(b"stray")

    logger  = RecordingLogger()
    payload = PaperFigurePack(_config(tmp_path / "runs", out, ["run_a"]), logger, {}).run()

    warnings = [message for level, message in logger.messages if level == "warning"]
    manifest = json.loads((out / "figure_manifest.json").read_text(encoding="utf-8"))

    assert len(payload["figures"]) == 6 and len(manifest["figures"]) == 6
    assert {entry["run"] for entry in manifest["figures"]} == {"run_a", "run_b"}
    assert all(entry["packed_at"] for entry in manifest["figures"])
    assert any(message.startswith("3 figure(s) overwritten in") and "run_a__pixel_maps__mse.png" in message for message in warnings)
    assert any("1 file(s) in" in message and "stray.png" in message for message in warnings)
