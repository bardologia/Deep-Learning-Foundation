from __future__ import annotations

import math

import numpy as np
import pytest

from tools.sar.geometry_field import GeometryField


PYRAT_BEAMFORMING_SIGN = -1.0


def _geometry_field(meta_dir) -> GeometryField:
    path = meta_dir / GeometryField.FILENAME
    if not path.exists():
        pytest.skip(f"{path} not present; re-run preprocessing for this dataset")

    return GeometryField.load(path)


def _height_axis(n_elevation: int) -> np.ndarray:
    return np.linspace(-20.0, 80.0, n_elevation)


def _clean_pixels(tomogram: np.ndarray, z: np.ndarray, az: np.ndarray, rg: np.ndarray):
    profile = np.abs(np.asarray(tomogram[:, az][:, :, rg])).astype(np.float64)
    n_elev  = profile.shape[0]

    argmax = profile.argmax(0)
    total  = profile.sum(0)
    peak_z = z[argmax]

    window        = 6
    concentration = np.empty(argmax.shape)
    for i in range(argmax.shape[0]):
        for j in range(argmax.shape[1]):
            a = argmax[i, j]
            concentration[i, j] = profile[max(0, a - window):min(n_elev, a + window + 1), i, j].sum() / max(total[i, j], 1e-12)

    clean = (concentration > 0.55) & (total > np.percentile(total, 40))
    ys, xs = np.where(clean)

    return ys, xs, peak_z


def _beamform_peaks(interferograms: np.ndarray, kz_secondary: np.ndarray, z: np.ndarray, sign: float) -> np.ndarray:
    phase = sign * kz_secondary[:, :, None] * z[None, None, :]
    steer = np.exp(1j * phase)

    spectrum = np.abs((steer * interferograms[:, :, None]).sum(0)) ** 2

    return z[spectrum.argmax(1)]


def _rayleigh(field: GeometryField, convention: str) -> float:
    return 2.0 * math.pi / float(np.ptp(field.kz(convention)[:, field.n_azimuth // 2, field.n_range // 2]))


def _sampled(meta_dir, tomogram_full, interferograms, convention: str, sign: float = PYRAT_BEAMFORMING_SIGN, max_pixels: int = 2000):
    field  = _geometry_field(meta_dir)
    n_elev = tomogram_full.shape[0]
    z      = _height_axis(n_elev)

    az      = np.arange(0, tomogram_full.shape[1], 8)
    rg      = np.arange(0, tomogram_full.shape[2], 8)

    ys, xs, peak_z = _clean_pixels(tomogram_full, z, az, rg)

    take    = np.linspace(0, len(ys) - 1, min(max_pixels, len(ys))).astype(int)
    ys, xs  = ys[take], xs[take]

    reference = peak_z[ys, xs]

    kz     = field.kz(convention)[:, az][:, :, rg]
    kz_sec = kz[1:, ys, xs]
    ifg    = np.asarray(interferograms[:, az][:, :, rg])[:, ys, xs]

    peaks = _beamform_peaks(ifg, kz_sec, z, sign)

    return peaks, reference, field, z


@pytest.mark.real_data
@pytest.mark.slow
def test_geometry_field_kz_matches_hand_formula(meta_dir):
    field = _geometry_field(meta_dir)

    scale    = 4.0 * math.pi / field.wavelength
    cos      = np.cos(field.look_angle).reshape(1, 1, -1)
    sin      = np.sin(field.look_angle).reshape(1, 1, -1)
    bperp    = field.baseline_h[:, :, None] * cos + field.baseline_v[:, :, None] * sin
    expected = scale * bperp / (field.slant_range.reshape(1, 1, -1) * sin)

    assert np.allclose(field.kz("height"), expected)


@pytest.mark.real_data
@pytest.mark.slow
def test_reference_track_kz_is_zero(meta_dir):
    field = _geometry_field(meta_dir)

    assert np.allclose(field.kz("height")[0], 0.0)
    assert np.allclose(field.kz("slant")[0],  0.0)


@pytest.mark.real_data
@pytest.mark.slow
def test_beamformed_peaks_match_pyrat_tomogram_axis(meta_dir, tomogram_full, interferograms):
    peaks, reference, field, z = _sampled(meta_dir, tomogram_full, interferograms, "height")

    rayleigh = _rayleigh(field, "height")

    median_error = float(np.median(np.abs(peaks - reference)))
    within_band  = float((np.abs(peaks - reference) < 3.0).mean())

    assert median_error < rayleigh
    assert within_band  > 0.7


@pytest.mark.real_data
@pytest.mark.slow
def test_flipped_beamforming_sign_misses_the_pyrat_tomogram_axis(meta_dir, tomogram_full, interferograms):
    aligned, reference, field, _ = _sampled(meta_dir, tomogram_full, interferograms, "height")
    flipped, _,         _,     _ = _sampled(meta_dir, tomogram_full, interferograms, "height", sign=-PYRAT_BEAMFORMING_SIGN)

    rayleigh = _rayleigh(field, "height")

    aligned_error = float(np.median(np.abs(aligned - reference)))
    flipped_error = float(np.median(np.abs(flipped - reference)))

    assert aligned_error < rayleigh
    assert flipped_error > rayleigh
    assert flipped_error > 3.0 * aligned_error


@pytest.mark.real_data
@pytest.mark.slow
def test_height_convention_fits_at_least_as_well_as_slant(meta_dir, tomogram_full, interferograms):
    height_peaks, reference, _, _ = _sampled(meta_dir, tomogram_full, interferograms, "height")
    slant_peaks,  _,         _, _ = _sampled(meta_dir, tomogram_full, interferograms, "slant")

    height_error = float(np.median(np.abs(height_peaks - reference)))
    slant_error  = float(np.median(np.abs(slant_peaks  - reference)))

    assert height_error <= slant_error + 0.5


@pytest.mark.real_data
@pytest.mark.slow
def test_tomogram_peaks_reference_terrain_surface(tomogram_full):
    n_elev  = tomogram_full.shape[0]
    z       = _height_axis(n_elev)

    az      = np.arange(0, tomogram_full.shape[1], 8)
    rg      = np.arange(0, tomogram_full.shape[2], 8)

    profile = np.abs(np.asarray(tomogram_full[:, az][:, :, rg])).astype(np.float64)
    peak_z  = z[profile.argmax(0)]

    assert -20.0 <= float(np.median(peak_z)) <= 20.0
    assert float(np.percentile(np.abs(peak_z), 90)) < 40.0
