from __future__ import annotations

import math

import numpy as np
import pytest

from tools.baselines.containers import SecondarySelection, TrackBaselines, TrackProfiles


def _table() -> TrackBaselines:
    return TrackBaselines(
        labels         = ["FL01_PS02", "FL01_PS04", "FL01_PS06", "FL01_PS08"],
        vertical       = [0.0, -1.0, 0.5, 2.0],
        horizontal     = [0.0, 9.0, 14.0, 20.0],
        vertical_std   = [0.06, 0.04, 0.04, 0.05],
        horizontal_std = [0.01, 0.01, 0.02, 0.02],
    )


def test_secondary_selection_returns_secondary_indices():
    labels = ["P0", "P1", "P2", "P3"]

    assert SecondarySelection.indices(labels, ["P1", "P3"]) == [0, 2]


def test_secondary_selection_honors_requested_order():
    labels = ["P0", "P1", "P2", "P3"]

    assert SecondarySelection.indices(labels, ["P3", "P1"]) == [2, 0]


def test_secondary_selection_rejects_duplicates():
    labels = ["P0", "P1", "P2"]

    with pytest.raises(ValueError):
        SecondarySelection.indices(labels, ["P1", "P1"])


def test_secondary_selection_rejects_primary():
    labels = ["P0", "P1", "P2"]

    with pytest.raises(ValueError):
        SecondarySelection.indices(labels, ["P0"])


def test_secondary_selection_rejects_unknown():
    labels = ["P0", "P1", "P2"]

    with pytest.raises(ValueError):
        SecondarySelection.indices(labels, ["PX"])


def test_reference_is_first_label():
    assert _table().reference == "FL01_PS02"


def test_n_tracks_counts_labels():
    assert _table().n_tracks == 4


def test_baselines_vertical_component():
    table = _table()

    assert table.baselines("vertical") == (0.0, -1.0, 0.5, 2.0)


def test_baselines_horizontal_component():
    table = _table()

    assert table.baselines("horizontal") == (0.0, 9.0, 14.0, 20.0)


def test_baselines_magnitude_is_hypot():
    table    = _table()
    expected = tuple(math.hypot(v, h) for v, h in zip(table.vertical, table.horizontal))

    assert table.baselines("magnitude") == pytest.approx(expected)


def test_baselines_perpendicular_hand_computed():
    table = _table()
    theta = math.radians(45.0)

    expected = tuple(h * math.cos(theta) + v * math.sin(theta) for v, h in zip(table.vertical, table.horizontal))

    assert table.baselines("perpendicular", look_angle_deg=45.0) == pytest.approx(expected)


def test_baselines_perpendicular_requires_look_angle():
    with pytest.raises(ValueError):
        _table().baselines("perpendicular")


def test_baselines_unknown_component_raises():
    with pytest.raises(ValueError):
        _table().baselines("diagonal")


def test_subset_keeps_reference_and_selected():
    table  = _table()
    subset = table.subset(["FL01_PS06"])

    assert subset.labels == ["FL01_PS02", "FL01_PS06"]
    assert subset.vertical == [0.0, 0.5]
    assert subset.horizontal == [0.0, 14.0]


def test_subset_none_returns_self():
    table = _table()

    assert table.subset(None) is table


def test_payload_round_trip_preserves_values():
    table     = _table()
    recovered = TrackBaselines.from_payload(table.to_payload())

    assert recovered.labels == table.labels
    assert recovered.vertical == table.vertical
    assert recovered.horizontal == table.horizontal
    assert recovered.reference == table.reference


def test_save_load_round_trip(tmp_path):
    table = _table()
    path  = table.save(tmp_path / "baselines.json")

    loaded = TrackBaselines.load(path)

    assert loaded.to_payload() == table.to_payload()


def test_describe_includes_reference_and_track_count():
    info = _table().describe()

    assert info["Tracks"] == 4
    assert info["Reference"] == "FL01_PS02"
    assert info["Azimuth window"] == "full track"


def test_describe_window_formatted_when_present():
    table = _table()
    table.azimuth_window = (1000, 2000)

    assert table.describe()["Azimuth window"] == "[1000, 2000)"


def _profiles() -> TrackProfiles:
    horizontal = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float)
    vertical   = np.array([[10.0, 12.0, 14.0], [20.0, 22.0, 24.0]], dtype=float)
    return TrackProfiles(labels=["P0", "P1"], horizontal=horizontal, vertical=vertical, azimuth_start=100)


def test_profiles_n_tracks_and_samples():
    prof = _profiles()

    assert prof.n_tracks == 2
    assert prof.n_samples == 3


def test_profiles_azimuth_axis():
    prof = _profiles()

    assert prof.azimuth_axis.tolist() == [100, 101, 102]


def test_profiles_relative_to_reference_subtracts_first():
    prof = _profiles()
    rel  = prof.relative_to_reference("vertical")

    assert rel[0].tolist() == [0.0, 0.0, 0.0]
    assert rel[1].tolist() == [10.0, 10.0, 10.0]


def test_profiles_planar_deviation_is_centered_radius():
    prof = _profiles()
    dev  = prof.planar_deviation()

    h_c = prof.horizontal - prof.horizontal.mean(axis=1, keepdims=True)
    v_c = prof.vertical   - prof.vertical.mean(axis=1, keepdims=True)

    assert np.allclose(dev, np.sqrt(h_c ** 2 + v_c ** 2))


def test_profiles_deviation_radii_shape():
    prof = _profiles()

    assert prof.deviation_radii().shape == (2,)


def test_profiles_position_summary_keys():
    summary = _profiles().position_summary()

    for key in ("labels", "horizontal_mean", "vertical_mean", "deviation_rms", "azimuth_start", "n_samples"):
        assert key in summary

    assert summary["azimuth_start"] == 100
    assert summary["n_samples"] == 3


def test_profiles_subset_keeps_reference():
    prof   = TrackProfiles(
        labels        = ["P0", "P1", "P2"],
        horizontal    = np.arange(9, dtype=float).reshape(3, 3),
        vertical      = np.arange(9, 18, dtype=float).reshape(3, 3),
        azimuth_start = 0,
    )
    subset = prof.subset(["P2"])

    assert subset.labels == ["P0", "P2"]
    assert subset.horizontal.shape == (2, 3)
    assert subset.vertical[1].tolist() == prof.vertical[2].tolist()


def test_profiles_save_load_round_trip(tmp_path):
    prof = _profiles()
    path = prof.save(tmp_path / "track_profiles.npz")

    loaded = TrackProfiles.load(path)

    assert loaded.labels == prof.labels
    assert loaded.azimuth_start == prof.azimuth_start
    assert np.allclose(loaded.horizontal, prof.horizontal)
    assert np.allclose(loaded.vertical, prof.vertical)


def test_profiles_file_path_helper(tmp_path):
    assert TrackProfiles.profiles_file(tmp_path) == tmp_path / "data" / "track_profiles.npz"
