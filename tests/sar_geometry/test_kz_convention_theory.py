from __future__ import annotations

import math

import numpy as np
import pytest

from tools.sar.geometry_field import GeometryField


def _field(look_deg=35.0, wavelength=0.2262, slant=3700.0, n_az=4, n_rg=3) -> GeometryField:
    look  = np.full(n_rg, math.radians(look_deg), dtype=np.float64)
    slant = np.linspace(slant, slant + 200.0, n_rg, dtype=np.float64)

    horizontal = np.array([0.0, 10.0, 25.0, 40.0], dtype=np.float64)[:, None] * np.ones((1, n_az))
    vertical   = np.array([0.0, -1.5, 2.0, -3.0],  dtype=np.float64)[:, None] * np.ones((1, n_az))

    return GeometryField(
        labels        = [f"T{i}" for i in range(4)],
        reference     = "T0",
        wavelength    = wavelength,
        azimuth_start = 0,
        range_start   = 0,
        look_angle    = look,
        slant_range   = slant,
        baseline_h    = horizontal,
        baseline_v    = vertical,
    )


def test_perpendicular_baseline_matches_projection_formula():
    field = _field(look_deg=35.0)
    theta = field.look_angle[0]

    bperp    = field.perpendicular_baseline()
    expected = field.baseline_h[:, :, None] * math.cos(theta) + field.baseline_v[:, :, None] * math.sin(theta)

    assert np.allclose(bperp, expected)


def test_slant_kz_equals_four_pi_bperp_over_lambda_r():
    field   = _field()
    kz      = field.kz("slant")

    scale    = 4.0 * math.pi / field.wavelength
    bperp    = field.perpendicular_baseline()
    expected = scale * bperp / field.slant_range.reshape(1, 1, -1)

    assert np.allclose(kz, expected)


def test_height_kz_is_slant_kz_divided_by_sin_theta():
    field = _field()
    sin   = np.sin(field.look_angle).reshape(1, 1, -1)

    assert np.allclose(field.kz("height"), field.kz("slant") / sin)


def test_interferometric_phase_invariant_across_conventions():
    field = _field(look_deg=35.0)
    sin   = np.sin(field.look_angle).reshape(1, 1, -1)

    elevation = 17.0
    height    = elevation * sin

    phase_slant  = field.kz("slant")  * elevation
    phase_height = field.kz("height") * height

    assert np.allclose(phase_slant, phase_height)


def test_reference_track_has_zero_kz_in_both_conventions():
    field = _field()

    assert np.allclose(field.kz("slant")[0],  0.0)
    assert np.allclose(field.kz("height")[0], 0.0)


def test_kz_grows_with_baseline_magnitude():
    field = _field()
    kz    = field.kz("slant")

    magnitude = np.abs(kz[:, 0, 0])

    assert np.all(np.diff(magnitude) > 0.0)


def test_unknown_convention_raises():
    field = _field()

    with pytest.raises(ValueError):
        field.kz("baseline")


def test_kz_inversely_proportional_to_slant_range():
    near = _field(slant=3000.0)
    far  = _field(slant=6000.0)

    ratio = far.kz("slant")[1, 0, 0] / near.kz("slant")[1, 0, 0]

    assert abs(ratio - 0.5) < 1e-9
