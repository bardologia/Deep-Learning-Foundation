from __future__ import annotations

import numpy as np
import pytest

from tools.data.representation import Representation


def test_channels_per_pass_values():
    assert Representation.REAL_IMAG.channels_per_pass     == 2
    assert Representation.MAG_REAL_IMAG.channels_per_pass == 3
    assert Representation.MAG_ANGLE.channels_per_pass     == 2
    assert Representation.MAG_RI_ANGLE.channels_per_pass  == 4
    assert Representation.ANGLE_ONLY.channels_per_pass    == 1
    assert Representation.MAG_ONLY.channels_per_pass      == 1


def test_slot_kinds_length_matches_channels():
    for rep in Representation:
        assert len(rep.slot_kinds) == rep.channels_per_pass


def _convert_single_pass(rep, data):
    cpp = rep.channels_per_pass
    out = np.zeros((cpp,) + data.shape, dtype=np.float32)
    rep.convert_into(out, data)
    return out


def test_real_imag_channels():
    data = np.array([3.0 + 4.0j, 1.0 - 2.0j], dtype=np.complex64)
    out  = _convert_single_pass(Representation.REAL_IMAG, data)

    assert np.allclose(out[0], data.real)
    assert np.allclose(out[1], data.imag)


def test_mag_only_channel():
    data = np.array([3.0 + 4.0j], dtype=np.complex64)
    out  = _convert_single_pass(Representation.MAG_ONLY, data)

    assert np.isclose(out[0, 0], 5.0)


def test_angle_only_channel():
    data = np.array([1.0j], dtype=np.complex64)
    out  = _convert_single_pass(Representation.ANGLE_ONLY, data)

    assert np.isclose(out[0, 0], np.pi / 2.0)


def test_mag_angle_channels():
    data = np.array([0.0 + 2.0j], dtype=np.complex64)
    out  = _convert_single_pass(Representation.MAG_ANGLE, data)

    assert np.isclose(out[0, 0], 2.0)
    assert np.isclose(out[1, 0], np.pi / 2.0)


def test_mag_real_imag_unit_normalized():
    data = np.array([3.0 + 4.0j], dtype=np.complex64)
    out  = _convert_single_pass(Representation.MAG_REAL_IMAG, data)

    assert np.isclose(out[0, 0], 5.0)
    assert np.isclose(out[1, 0], 3.0 / 5.0)
    assert np.isclose(out[2, 0], 4.0 / 5.0)
    assert np.isclose(out[1, 0] ** 2 + out[2, 0] ** 2, 1.0)


def test_mag_real_imag_zero_safe():
    data = np.array([0.0 + 0.0j], dtype=np.complex64)
    out  = _convert_single_pass(Representation.MAG_REAL_IMAG, data)

    assert np.all(np.isfinite(out))
    assert out[0, 0] == 0.0


def test_mag_ri_angle_four_channels():
    data = np.array([3.0 + 4.0j], dtype=np.complex64)
    out  = _convert_single_pass(Representation.MAG_RI_ANGLE, data)

    assert out.shape[0] == 4
    assert np.isclose(out[0, 0], 5.0)
    assert np.isclose(out[1, 0], 0.6)
    assert np.isclose(out[2, 0], 0.8)
    assert np.isclose(out[3, 0], np.angle(3.0 + 4.0j))


def test_convert_into_interleaves_multiple_passes():
    rep    = Representation.REAL_IMAG
    passes = 3
    data   = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], dtype=np.complex64)
    out    = np.zeros((rep.channels_per_pass * passes,), dtype=np.float32)

    rep.convert_into(out, data)

    assert np.allclose(out[0::2], data.real)
    assert np.allclose(out[1::2], data.imag)


@pytest.mark.real_data
def test_real_imag_reconstructs_complex(secondaries):
    data = np.asarray(secondaries[0, :16, :16]).astype(np.complex64)
    out  = _convert_single_pass(Representation.REAL_IMAG, data)

    reconstructed = out[0] + 1j * out[1]
    assert np.allclose(reconstructed, data, atol=1e-4)


@pytest.mark.real_data
def test_mag_only_matches_abs_on_real_data(interferograms):
    data = np.asarray(interferograms[0, :16, :16]).astype(np.complex64)
    out  = _convert_single_pass(Representation.MAG_ONLY, data)

    assert np.allclose(out[0], np.abs(data), atol=1e-4)


@pytest.mark.real_data
def test_mag_real_imag_norm_on_real_data(interferograms):
    data = np.asarray(interferograms[0, :16, :16]).astype(np.complex64)
    out  = _convert_single_pass(Representation.MAG_REAL_IMAG, data)

    norm    = out[1] ** 2 + out[2] ** 2
    nonzero = np.abs(data) > 0
    assert np.allclose(norm[nonzero], 1.0, atol=1e-3)
