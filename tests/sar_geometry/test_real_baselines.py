from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from configuration.sar.geometry_config import GeometryConfig
from tools.baselines.containers        import TrackBaselines, TrackProfiles
from tools.sar.tomo_geometry           import TomoGeometry


@pytest.mark.real_data
def test_baselines_json_lengths_match_labels(baselines_json):
    n = len(baselines_json["labels"])

    assert n == 29
    for key in ("vertical", "horizontal", "vertical_std", "horizontal_std", "vertical_absolute", "horizontal_absolute", "track_files"):
        assert len(baselines_json[key]) == n


@pytest.mark.real_data
def test_baselines_json_reference_is_first_label(baselines_json):
    assert baselines_json["reference"] == baselines_json["labels"][0]
    assert baselines_json["reference"] == "FL01_PS02"


@pytest.mark.real_data
def test_real_reference_relative_baselines_are_zero(baselines_json):
    assert baselines_json["vertical"][0] == pytest.approx(0.0, abs=1e-9)
    assert baselines_json["horizontal"][0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.real_data
def test_real_relative_equals_absolute_minus_reference(baselines_json):
    v_abs = baselines_json["vertical_absolute"]
    h_abs = baselines_json["horizontal_absolute"]

    for i in range(len(v_abs)):
        assert baselines_json["vertical"][i]   == pytest.approx(v_abs[i] - v_abs[0], abs=1e-6)
        assert baselines_json["horizontal"][i] == pytest.approx(h_abs[i] - h_abs[0], abs=1e-6)


@pytest.mark.real_data
def test_real_stds_are_non_negative(baselines_json):
    assert all(s >= 0.0 for s in baselines_json["vertical_std"])
    assert all(s >= 0.0 for s in baselines_json["horizontal_std"])


@pytest.mark.real_data
def test_real_baselines_round_trip_through_container(baselines_json):
    table     = TrackBaselines.from_payload(baselines_json)
    recovered = TrackBaselines.from_payload(table.to_payload())

    assert recovered.labels == table.labels
    assert recovered.vertical == table.vertical
    assert recovered.horizontal == table.horizontal
    assert recovered.n_tracks == 29


@pytest.mark.real_data
def test_real_container_save_load(baselines_json, tmp_path):
    table = TrackBaselines.from_payload(baselines_json)
    path  = table.save(tmp_path / "baselines.json")

    assert TrackBaselines.load(path).to_payload() == table.to_payload()


@pytest.mark.real_data
def test_real_perpendicular_baseline_formula(baselines_json):
    table = TrackBaselines.from_payload(baselines_json)
    perp  = table.baselines("perpendicular", look_angle_deg=45.0)
    theta = math.radians(45.0)

    expected = tuple(h * math.cos(theta) + v * math.sin(theta) for v, h in zip(table.vertical, table.horizontal))

    assert perp == pytest.approx(expected)
    assert perp[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.real_data
def test_real_magnitude_baseline_dominated_by_horizontal(baselines_json):
    table = TrackBaselines.from_payload(baselines_json)
    mag   = table.baselines("magnitude")

    assert mag[0] == pytest.approx(0.0, abs=1e-9)
    assert all(m >= abs(v) - 1e-9 for m, v in zip(mag, table.vertical))


@pytest.mark.real_data
def test_real_subset_preset_secondaries(baselines_json):
    table  = TrackBaselines.from_payload(baselines_json)
    subset = table.subset(["FL01_PS04", "FL01_PS06", "FL01_PS08"])

    assert subset.labels == ["FL01_PS02", "FL01_PS04", "FL01_PS06", "FL01_PS08"]
    assert subset.n_tracks == 4
    assert subset.vertical[0] == 0.0


@pytest.mark.real_data
def test_real_kz_from_perpendicular_baselines_sign(baselines_json):
    table = TrackBaselines.from_payload(baselines_json)
    perp  = table.baselines("perpendicular", look_angle_deg=45.0)

    cfg   = GeometryConfig(wavelength=0.23, slant_range=5000.0, look_angle_deg=45.0, baselines=tuple(perp))
    geom  = TomoGeometry(cfg, torch.linspace(-20.0, 80.0, 150))

    scale = 4.0 * math.pi / (0.23 * 5000.0 * math.sin(math.radians(45.0)))

    assert geom.n_tracks == 29
    assert float(geom.kz[0]) == 0.0
    assert geom.kz.tolist() == pytest.approx([scale * b for b in perp], rel=1e-6)


@pytest.mark.real_data
def test_geometry_config_resolved_loads_real_baselines(baselines_json, meta_dir):
    cfg      = GeometryConfig(baselines_source="dataset", baseline_component="perpendicular", look_angle_deg=45.0)
    dataset  = meta_dir.parent
    resolved = cfg.resolved(dataset)

    table = TrackBaselines.from_payload(baselines_json)
    perp  = table.baselines("perpendicular", look_angle_deg=45.0)

    assert len(resolved.baselines) == 29
    assert resolved.baselines == pytest.approx(perp)
    assert resolved.baselines_origin.endswith("baselines.json")


@pytest.mark.real_data
def test_real_track_profiles_shapes(track_profiles):
    assert len(track_profiles["labels"]) == 29
    assert track_profiles["horizontal"].shape == (29, 1000)
    assert track_profiles["vertical"].shape == (29, 1000)
    assert int(track_profiles["azimuth_start"]) == 1000


@pytest.mark.real_data
def test_real_track_profiles_container_round_trip(track_profiles, tmp_path):
    prof = TrackProfiles(
        labels        = [str(x) for x in track_profiles["labels"]],
        horizontal    = np.asarray(track_profiles["horizontal"], dtype=float),
        vertical      = np.asarray(track_profiles["vertical"],   dtype=float),
        azimuth_start = int(track_profiles["azimuth_start"]),
        track_files   = [str(x) for x in track_profiles["track_files"]],
    )

    path   = prof.save(tmp_path / "track_profiles.npz")
    loaded = TrackProfiles.load(path)

    assert loaded.labels == prof.labels
    assert loaded.azimuth_start == 1000
    assert np.allclose(loaded.horizontal, prof.horizontal)
    assert np.allclose(loaded.vertical, prof.vertical)


@pytest.mark.real_data
def test_real_profiles_azimuth_axis_matches_window(track_profiles):
    prof = TrackProfiles(
        labels        = [str(x) for x in track_profiles["labels"]],
        horizontal    = np.asarray(track_profiles["horizontal"], dtype=float),
        vertical      = np.asarray(track_profiles["vertical"],   dtype=float),
        azimuth_start = int(track_profiles["azimuth_start"]),
    )

    axis = prof.azimuth_axis

    assert axis[0] == 1000
    assert axis[-1] == 1999
    assert prof.n_samples == 1000


@pytest.mark.real_data
def test_real_profiles_means_match_relative_baselines(track_profiles, baselines_json):
    prof = TrackProfiles(
        labels        = [str(x) for x in track_profiles["labels"]],
        horizontal    = np.asarray(track_profiles["horizontal"], dtype=float),
        vertical      = np.asarray(track_profiles["vertical"],   dtype=float),
        azimuth_start = int(track_profiles["azimuth_start"]),
    )

    h_mean = np.nanmean(prof.horizontal, axis=1)
    v_mean = np.nanmean(prof.vertical,   axis=1)

    h_rel = h_mean - h_mean[0]
    v_rel = v_mean - v_mean[0]

    assert np.allclose(h_rel, baselines_json["horizontal"], atol=1e-3)
    assert np.allclose(v_rel, baselines_json["vertical"],   atol=1e-3)


@pytest.mark.real_data
def test_real_profiles_relative_reference_is_zero(track_profiles):
    prof = TrackProfiles(
        labels        = [str(x) for x in track_profiles["labels"]],
        horizontal    = np.asarray(track_profiles["horizontal"], dtype=float),
        vertical      = np.asarray(track_profiles["vertical"],   dtype=float),
        azimuth_start = int(track_profiles["azimuth_start"]),
    )

    rel = prof.relative_to_reference("vertical")

    assert np.allclose(rel[0], 0.0)


@pytest.mark.real_data
def test_real_profiles_deviation_radii_positive(track_profiles):
    prof = TrackProfiles(
        labels        = [str(x) for x in track_profiles["labels"]],
        horizontal    = np.asarray(track_profiles["horizontal"], dtype=float),
        vertical      = np.asarray(track_profiles["vertical"],   dtype=float),
        azimuth_start = int(track_profiles["azimuth_start"]),
    )

    radii = prof.deviation_radii()

    assert radii.shape == (29,)
    assert np.all(radii >= 0.0)


@pytest.mark.real_data
def test_real_profiles_subset_consistent_with_baselines_subset(track_profiles, baselines_json):
    prof = TrackProfiles(
        labels        = [str(x) for x in track_profiles["labels"]],
        horizontal    = np.asarray(track_profiles["horizontal"], dtype=float),
        vertical      = np.asarray(track_profiles["vertical"],   dtype=float),
        azimuth_start = int(track_profiles["azimuth_start"]),
    )
    table = TrackBaselines.from_payload(baselines_json)

    secondaries = ["FL01_PS04", "FL01_PS08"]

    assert prof.subset(secondaries).labels == table.subset(secondaries).labels
