"""Tests for the quantised volume the cube explorer packs for the WebGL volume view.

Covers the eight-float header carrying the strided extents, the diverging flag
and the display range, the byte mapping of the voxels with NaN at zero, the
striding cap and its clipping, the refusal of unknown sources and cubes, and
the /api/cubes/volume route.
"""

from __future__ import annotations

from urllib.parse import urlencode

import numpy as np

from routers.cube_routers import CubeRouter
from routers.dispatch     import HttpExchange

from tests.webui.conftest import N_COLS, N_ROWS, N_Z, FakeHandler, load_cube, loaded_cube, make_cube_run, make_preproc, make_stamp, open_explorer


def _unpack(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Splits a volume blob into its float32 header and its voxels shaped by the header extents."""
    header = np.frombuffer(blob[:32], dtype=np.float32)
    voxels = np.frombuffer(blob[32:], dtype=np.uint8).reshape(tuple(int(v) for v in header[:3]))
    return header, voxels


def _expected_bytes(block: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Quantises a float block onto bytes over [vmin, vmax] with NaN at the low end."""
    norm = np.clip((np.nan_to_num(block, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.round(norm * 255.0).astype(np.uint8)


def test_volume_bin_quantises_and_strides(tmp_path):
    """Checks the header matches the full extents of a small cube and the voxels span the byte range."""
    explorer, cube_id = loaded_cube(tmp_path)

    header, voxels = _unpack(explorer.volume_bin(cube_id, "pred"))

    assert tuple(int(v) for v in header[:3]) == (N_Z, N_ROWS, N_COLS)
    assert header[3] == 0.0 and header[5] > header[4]
    assert voxels.size == N_Z * N_ROWS * N_COLS

    small_header, _ = _unpack(explorer.volume_bin(cube_id, "pred", max_dim=4))
    assert all(int(v) <= 4 for v in small_header[:3])

    assert explorer.volume_bin(cube_id, "banana") is None
    assert explorer.volume_bin(cube_id, "predb") is None
    assert explorer.volume_bin("wrong", "pred") is None


def test_volume_bin_maps_the_display_range_onto_bytes_with_nan_at_zero(tmp_path):
    """Checks every voxel is the source value scaled over its colour limits and a NaN voxel packs as zero."""
    stamp = make_cube_run(tmp_path)
    cube  = np.load(stamp / "cubes" / "pred_curves.npy")
    cube[1, 2, 3] = np.nan
    np.save(stamp / "cubes" / "pred_curves.npy", cube)

    explorer, cube_ids = open_explorer(tmp_path, expected=1)
    load_cube(explorer, cube_ids[0])

    entry          = explorer._entry(cube_ids[0], "pred")
    header, voxels = _unpack(explorer.volume_bin(cube_ids[0], "pred"))

    assert header[4] == np.float32(entry["vmin"]) and header[5] == np.float32(entry["vmax"])
    assert np.array_equal(voxels, _expected_bytes(cube, float(entry["vmin"]), float(entry["vmax"])))
    assert voxels[1, 2, 3] == 0


def test_volume_bin_strides_every_axis_under_the_cap(tmp_path):
    """Checks each axis keeps every n-th voxel with the smallest stride that fits the cap, and the cap is clipped to at least two."""
    explorer, cube_id = loaded_cube(tmp_path)
    entry             = explorer._entry(cube_id, "pred")

    header, voxels = _unpack(explorer.volume_bin(cube_id, "pred", max_dim=4))
    block          = np.asarray(entry["cube"][::2, ::2, ::2], dtype=np.float32)

    assert tuple(int(v) for v in header[:3]) == (3, 4, 3)
    assert np.array_equal(voxels, _expected_bytes(block, float(entry["vmin"]), float(entry["vmax"])))

    floor_header, _ = _unpack(explorer.volume_bin(cube_id, "pred", max_dim=1))
    assert tuple(int(v) for v in floor_header[:3]) == (2, 2, 2)


def test_volume_bin_of_the_difference_is_symmetric_about_zero(tmp_path):
    """Checks the A minus B source is flagged diverging and quantised over a range centred on zero."""
    preproc = make_preproc(tmp_path)
    stamp_a = make_stamp(tmp_path / "run_a", preproc, np.random.default_rng(1))
    stamp_b = make_stamp(tmp_path / "run_b", preproc, np.random.default_rng(2))

    explorer, _ = open_explorer(tmp_path, expected=2)
    load_cube(explorer, str(stamp_a))
    assert explorer.attach_second(str(stamp_a), str(stamp_b))["ok"]

    entry          = explorer._entry(str(stamp_a), "diff")
    header, voxels = _unpack(explorer.volume_bin(str(stamp_a), "diff"))
    peak           = max(abs(float(entry["vmin"])), abs(float(entry["vmax"])))

    assert header[3] == 1.0
    assert header[4] == np.float32(-peak) and header[5] == np.float32(peak)
    assert np.array_equal(voxels, _expected_bytes(np.asarray(entry["cube"], dtype=np.float32), -peak, peak))


def test_volume_route_answers_the_packed_blob(tmp_path):
    """Checks /api/cubes/volume answers the explorer's blob for the requested cap and 404 for an unknown source."""
    explorer, cube_id = loaded_cube(tmp_path)

    def ask(query: dict) -> FakeHandler:
        handler = FakeHandler("GET", f"/api/cubes/volume?{urlencode(query)}")
        assert CubeRouter(explorer).handle(HttpExchange.of(handler))
        return handler

    answered = ask({"id": cube_id, "source": "gt", "max": "4"})
    assert answered.status == 200
    assert answered.wfile.getvalue() == explorer.volume_bin(cube_id, "gt", max_dim=4)

    assert ask({"id": cube_id, "source": "banana"}).status == 404
