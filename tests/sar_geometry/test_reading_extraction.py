from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.baselines.containers import TrackBaselines, TrackProfiles
from tools.baselines.extraction import BaselineExtractor
from tools.baselines.reading    import TrackFileResolver, TrackReader


def _fake_track(n_azimuth: int = 50, h: float = 5.0, v: float = 3700.0) -> np.ndarray:
    rows = np.zeros((4, n_azimuth), dtype=float)
    rows[BaselineExtractor.HORIZONTAL_ROW] = h + np.linspace(-0.5, 0.5, n_azimuth)
    rows[BaselineExtractor.VERTICAL_ROW]   = v + np.linspace(-0.2, 0.2, n_azimuth)
    return rows


def test_track_reader_passes_custom_reader_through():
    reader = TrackReader(lambda path: _fake_track())
    data   = reader.read("anything.rat")

    assert data.shape == (4, 50)


def test_track_reader_rejects_too_few_rows():
    reader = TrackReader(lambda path: np.zeros((2, 10)))

    with pytest.raises(ValueError):
        reader.read("x.rat")


def test_track_reader_rejects_non_2d():
    reader = TrackReader(lambda path: np.zeros((4, 4, 4)))

    with pytest.raises(ValueError):
        reader.read("x.rat")


def test_resolver_label_from_pass_directory():
    resolver = TrackFileResolver()

    assert resolver.label("/data/FL01/PS02") == "FL01_PS02"


def test_resolver_label_strips_track_subdir():
    resolver = TrackFileResolver()

    assert resolver.label("/data/FL01/PS02/T01L") == "FL01_PS02"


def test_resolver_resolve_finds_track_file(tmp_path):
    track_dir = tmp_path / "PS02" / "INF" / "INF-TRACK"
    track_dir.mkdir(parents=True)
    (track_dir / "track_sar_resa_x.rat").write_text("stub")

    resolved = TrackFileResolver().resolve(tmp_path / "PS02")

    assert resolved.name == "track_sar_resa_x.rat"


def test_resolver_prefers_resa_pattern(tmp_path):
    track_dir = tmp_path / "PS02" / "INF" / "INF-TRACK"
    track_dir.mkdir(parents=True)
    (track_dir / "track_other.rat").write_text("stub")
    (track_dir / "track_sar_resa_y.rat").write_text("stub")

    resolved = TrackFileResolver().resolve(tmp_path / "PS02")

    assert "resa" in resolved.name


def test_resolver_raises_when_missing(tmp_path):
    (tmp_path / "PS02" / "INF" / "INF-TRACK").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        TrackFileResolver().resolve(tmp_path / "PS02")


def test_resolve_passes_builds_label_mapping(tmp_path):
    for pass_name in ("PS02", "PS04"):
        d = tmp_path / "FL01" / pass_name / "INF" / "INF-TRACK"
        d.mkdir(parents=True)
        (d / "track_sar_resa_z.rat").write_text("stub")

    mapping = TrackFileResolver().resolve_passes([tmp_path / "FL01" / "PS02", tmp_path / "FL01" / "PS04"])

    assert set(mapping.keys()) == {"FL01_PS02", "FL01_PS04"}


def test_resolve_passes_rejects_duplicate_labels(tmp_path):
    d = tmp_path / "FL01" / "PS02" / "INF" / "INF-TRACK"
    d.mkdir(parents=True)
    (d / "track_sar_resa_z.rat").write_text("stub")

    with pytest.raises(ValueError, match="unique label"):
        TrackFileResolver().resolve_passes([tmp_path / "FL01" / "PS02", tmp_path / "FL01" / "PS02"])


def test_extractor_reference_baselines_are_zero():
    paths = {
        "FL01_PS02": Path("a.rat"),
        "FL01_PS04": Path("b.rat"),
    }
    tracks = {
        "a.rat": _fake_track(h=5.0, v=3700.0),
        "b.rat": _fake_track(h=9.0, v=3699.0),
    }

    extractor = BaselineExtractor(paths, reader=lambda p: tracks[Path(p).name])
    table     = extractor.extract()

    assert table.vertical[0] == pytest.approx(0.0, abs=1e-9)
    assert table.horizontal[0] == pytest.approx(0.0, abs=1e-9)


def test_extractor_relative_equals_absolute_minus_reference():
    paths  = {"FL01_PS02": Path("a.rat"), "FL01_PS04": Path("b.rat")}
    tracks = {
        "a.rat": _fake_track(h=5.0, v=3700.0),
        "b.rat": _fake_track(h=9.0, v=3699.0),
    }

    table = BaselineExtractor(paths, reader=lambda p: tracks[Path(p).name]).extract()

    assert table.horizontal[1] == pytest.approx(table.horizontal_absolute[1] - table.horizontal_absolute[0])
    assert table.vertical[1] == pytest.approx(table.vertical_absolute[1] - table.vertical_absolute[0])


def test_extractor_with_profiles_returns_aligned_arrays():
    paths  = {"FL01_PS02": Path("a.rat"), "FL01_PS04": Path("b.rat")}
    tracks = {
        "a.rat": _fake_track(n_azimuth=40, h=5.0, v=3700.0),
        "b.rat": _fake_track(n_azimuth=50, h=9.0, v=3699.0),
    }

    table, profiles = BaselineExtractor(paths, reader=lambda p: tracks[Path(p).name]).extract_with_profiles()

    assert isinstance(table, TrackBaselines)
    assert isinstance(profiles, TrackProfiles)
    assert profiles.horizontal.shape == (2, 40)
    assert profiles.vertical.shape == (2, 40)


def test_extractor_azimuth_window_slices():
    paths  = {"FL01_PS02": Path("a.rat")}
    tracks = {"a.rat": _fake_track(n_azimuth=100)}

    table, profiles = BaselineExtractor(paths, azimuth_window=(20, 60), reader=lambda p: tracks[Path(p).name]).extract_with_profiles()

    assert profiles.n_samples == 40
    assert profiles.azimuth_start == 20
    assert table.azimuth_window == (20, 60)


def test_extractor_window_start_beyond_length_raises():
    paths  = {"FL01_PS02": Path("a.rat")}
    tracks = {"a.rat": _fake_track(n_azimuth=30)}

    with pytest.raises(ValueError):
        BaselineExtractor(paths, azimuth_window=(40, 60), reader=lambda p: tracks[Path(p).name]).extract()


def test_extractor_window_end_beyond_length_raises():
    paths  = {"FL01_PS02": Path("a.rat")}
    tracks = {"a.rat": _fake_track(n_azimuth=50)}

    with pytest.raises(ValueError, match="not covered"):
        BaselineExtractor(paths, azimuth_window=(20, 60), reader=lambda p: tracks[Path(p).name]).extract()


def test_extractor_short_secondary_track_raises():
    paths  = {"FL01_PS02": Path("a.rat"), "FL01_PS04": Path("b.rat")}
    tracks = {"a.rat": _fake_track(n_azimuth=100), "b.rat": _fake_track(n_azimuth=50)}

    with pytest.raises(ValueError, match="not covered"):
        BaselineExtractor(paths, azimuth_window=(20, 60), reader=lambda p: tracks[Path(p).name]).extract_with_profiles()
