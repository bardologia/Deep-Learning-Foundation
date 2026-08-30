"""Tests for complex-to-real channel conversion in the Representation enum.

Covers channel counts, slot-kind labelling, per-mode channel semantics
(magnitude, unit real/imag, phase angle), zero-magnitude safety, pass
interleaving, and agreement with numpy on real SAR windows.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.data.representation import Representation


def test_channels_per_source_values():
    """Each representation reports its documented channel count per source."""
    assert Representation.REAL_IMAG.channels_per_source     == 2
    assert Representation.MAG_REAL_IMAG.channels_per_source == 3
    assert Representation.MAG_ANGLE.channels_per_source     == 2
    assert Representation.MAG_RI_ANGLE.channels_per_source  == 4
    assert Representation.ANGLE_ONLY.channels_per_source    == 1
    assert Representation.MAG_ONLY.channels_per_source      == 1


def test_slot_kinds_length_matches_channels():
    """Every representation labels exactly as many slot kinds as it emits channels."""
    for rep in Representation:
        assert len(rep.slot_kinds) == rep.channels_per_source


def _convert_single_channel(rep, data):
    """Returns the converted channels of shape (channels_per_source, *data.shape) for one chan."""
    cpp = rep.channels_per_source
    out = np.zeros((cpp,) + data.shape, dtype=np.float32)
    rep.convert_into(out, data)
    return out


def test_real_imag_channels():
    """REAL_IMAG writes the real part to channel 0 and the imaginary part to channel 1."""
    data = np.array([3.0 + 4.0j, 1.0 - 2.0j], dtype=np.complex64)
    out  = _convert_single_channel(Representation.REAL_IMAG, data)

    assert np.allclose(out[0], data.real)
    assert np.allclose(out[1], data.imag)


def test_mag_only_channel():
    """MAG_ONLY emits the complex modulus in its single channel."""
    data = np.array([3.0 + 4.0j], dtype=np.complex64)
    out  = _convert_single_channel(Representation.MAG_ONLY, data)

    assert np.isclose(out[0, 0], 5.0)


def test_angle_only_channel():
    """ANGLE_ONLY emits the phase in radians in its single channel."""
    data = np.array([1.0j], dtype=np.complex64)
    out  = _convert_single_channel(Representation.ANGLE_ONLY, data)

    assert np.isclose(out[0, 0], np.pi / 2.0)


def test_mag_angle_channels():
    """MAG_ANGLE emits modulus then phase in radians."""
    data = np.array([0.0 + 2.0j], dtype=np.complex64)
    out  = _convert_single_channel(Representation.MAG_ANGLE, data)

    assert np.isclose(out[0, 0], 2.0)
    assert np.isclose(out[1, 0], np.pi / 2.0)


def test_mag_real_imag_unit_normalized():
    """MAG_REAL_IMAG emits modulus plus a unit-norm real/imaginary direction."""
    data = np.array([3.0 + 4.0j], dtype=np.complex64)
    out  = _convert_single_channel(Representation.MAG_REAL_IMAG, data)

    assert np.isclose(out[0, 0], 5.0)
    assert np.isclose(out[1, 0], 3.0 / 5.0)
    assert np.isclose(out[2, 0], 4.0 / 5.0)
    assert np.isclose(out[1, 0] ** 2 + out[2, 0] ** 2, 1.0)


def test_mag_real_imag_zero_safe():
    """A zero sample yields finite channels with zero magnitude instead of NaN."""
    data = np.array([0.0 + 0.0j], dtype=np.complex64)
    out  = _convert_single_channel(Representation.MAG_REAL_IMAG, data)

    assert np.all(np.isfinite(out))
    assert out[0, 0] == 0.0


def test_mag_ri_angle_four_channels():
    """MAG_RI_ANGLE emits modulus, unit real, unit imaginary and phase in radians."""
    data = np.array([3.0 + 4.0j], dtype=np.complex64)
    out  = _convert_single_channel(Representation.MAG_RI_ANGLE, data)

    assert out.shape[0] == 4
    assert np.isclose(out[0, 0], 5.0)
    assert np.isclose(out[1, 0], 0.6)
    assert np.isclose(out[2, 0], 0.8)
    assert np.isclose(out[3, 0], np.angle(3.0 + 4.0j))


def test_convert_into_interleaves_multiple_sources():
    """Multi-source output interleaves channels source by source rather than blocking them."""
    rep    = Representation.REAL_IMAG
    sources = 3
    data   = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], dtype=np.complex64)
    out    = np.zeros((rep.channels_per_source * sources,), dtype=np.float32)

    rep.convert_into(out, data)

    assert np.allclose(out[0::2], data.real)
    assert np.allclose(out[1::2], data.imag)


@pytest.mark.dataset
def test_real_imag_reconstructs_complex(secondaries):
    """REAL_IMAG channels recombine into the original complex secondary window."""
    data = np.asarray(secondaries[0, :16, :16]).astype(np.complex64)
    out  = _convert_single_channel(Representation.REAL_IMAG, data)

    reconstructed = out[0] + 1j * out[1]
    assert np.allclose(reconstructed, data, atol=1e-4)


@pytest.mark.dataset
def test_mag_only_matches_abs_on_scene_channels(derived):
    """MAG_ONLY on a scene derived-channel window equals numpy's absolute value."""
    data = np.asarray(derived[0, :16, :16]).astype(np.complex64)
    out  = _convert_single_channel(Representation.MAG_ONLY, data)

    assert np.allclose(out[0], np.abs(data), atol=1e-4)


@pytest.mark.dataset
def test_mag_real_imag_norm_on_scene_channels(derived):
    """Unit real/imaginary channels stay on the unit circle for nonzero real pixels."""
    data = np.asarray(derived[0, :16, :16]).astype(np.complex64)
    out  = _convert_single_channel(Representation.MAG_REAL_IMAG, data)

    norm    = out[1] ** 2 + out[2] ** 2
    nonzero = np.abs(data) > 0
    assert np.allclose(norm[nonzero], 1.0, atol=1e-3)
