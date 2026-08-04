from __future__ import annotations

from pathlib import Path

import numpy as np

from cube_explorer import CubeExplorer

from tests.webui.conftest import N_ROWS, N_Z, N_COLS, load_cube, make_preproc, make_stamp, open_explorer


def _make_run(base: Path, preproc: Path, name: str, seed: int, x_min: float = -10.0) -> Path:
    return make_stamp(base / name, preproc, np.random.default_rng(seed), x_min=x_min)


def _loaded(base: Path) -> tuple[CubeExplorer, str, str]:
    preproc = make_preproc(base)
    stamp_a = _make_run(base, preproc, "run_a", seed=1)
    stamp_b = _make_run(base, preproc, "run_b", seed=2)

    explorer, _ = open_explorer(base, expected=2)
    cube_a      = str(stamp_a)

    load_cube(explorer, cube_a)

    return explorer, cube_a, str(stamp_b)


def test_attach_adds_predb_and_diff(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)

    result = explorer.attach_second(cube_a, cube_b)
    assert result["ok"], result

    meta = result["cube"]
    assert meta["sources"] == ["pred", "predb", "diff", "gt"]
    assert meta["attached"]["id"] == cube_b
    assert meta["attached"]["run"] == "run_b"

    pred_a = np.load(Path(cube_a) / "cubes" / "pred_curves.npy")
    pred_b = np.load(Path(cube_b) / "cubes" / "pred_curves.npy")

    profiles = explorer.profiles(cube_a, row=2, col=3)
    assert set(profiles["sources"]) == {"pred", "predb", "diff", "gt"}

    expected_diff = (pred_a - pred_b)[:, 2, 3]
    order         = np.argsort(np.linspace(-10.0, 30.0, N_Z))
    assert np.allclose(profiles["sources"]["diff"]["values"], expected_diff[order])

    assert explorer.slice_png(cube_a, "predb", "col", row=0, col=0)[:4] == b"\x89PNG"
    assert explorer.slice_png(cube_a, "diff",  "col", row=0, col=0)[:4] == b"\x89PNG"
    assert explorer.plane_png(cube_a, "diff", frac=0.5, space="normalized")[:4] == b"\x89PNG"


def test_diff_range_is_symmetric(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    entry = explorer._entry(cube_a, "diff")
    assert entry["diverging"] is True
    assert entry["vmin"] == -entry["vmax"]
    assert entry["vmax"] > 0


def test_detach_removes_comparison(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.detach_second(cube_a)
    assert result["ok"]
    assert result["cube"]["sources"] == ["pred", "gt"]
    assert result["cube"]["attached"] is None
    assert explorer.slice_png(cube_a, "predb", "col", row=0, col=0) is None


def test_attach_rejects_self_and_unknown(tmp_path):
    explorer, cube_a, _ = _loaded(tmp_path)

    assert not explorer.attach_second(cube_a, cube_a)["ok"]
    assert not explorer.attach_second(cube_a, str(tmp_path / "nowhere"))["ok"]
    assert not explorer.attach_second(str(tmp_path / "other"), cube_a)["ok"]


def test_attach_rejects_mismatched_axis(tmp_path):
    preproc = make_preproc(tmp_path)
    stamp_a = _make_run(tmp_path, preproc, "run_a", seed=1)
    stamp_c = _make_run(tmp_path, preproc, "run_c", seed=3, x_min=-99.0)

    explorer, _ = open_explorer(tmp_path, expected=2)
    load_cube(explorer, str(stamp_a))

    result = explorer.attach_second(str(stamp_a), str(stamp_c))
    assert not result["ok"] and "z axis" in result["error"]


def test_ssim_covers_predb(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.slice_ssim(cube_a, row=2, col=2)
    assert result["ok"]
    assert "predb" in result["col"]
    assert "diff" not in result["col"]


def test_transect_png_samples_line(tmp_path):
    explorer, cube_a, _ = _loaded(tmp_path)

    png = explorer.transect_png(cube_a, "pred", row0=0, col0=0, row1=N_ROWS - 1, col1=N_COLS - 1)
    assert png and png[:4] == b"\x89PNG"

    normalized = explorer.transect_png(cube_a, "pred", row0=2, col0=1, row1=2, col1=4, space="normalized")
    assert normalized and normalized[:4] == b"\x89PNG"

    assert explorer.transect_png(cube_a, "banana", 0, 0, 1, 1) is None
    assert explorer.transect_png("wrong", "pred", 0, 0, 1, 1) is None


def test_transect_cut_geometry(tmp_path):
    explorer, cube_a, _ = _loaded(tmp_path)

    entry = explorer._entry(cube_a, "pred")
    data, z_values, vmin, vmax = explorer._transect_cut(entry, 0, 0, N_ROWS - 1, N_COLS - 1, "physical")

    assert data.shape == (N_Z, max(N_ROWS, N_COLS))
    assert z_values[0] == -10.0 and z_values[-1] == 30.0
    assert vmin == entry["vmin"] and vmax == entry["vmax"]

    cube = entry["cube"]
    assert np.allclose(data[:, 0], cube[:, 0, 0])


def test_save_transect_writes_figures(tmp_path):
    explorer, cube_a, _ = _loaded(tmp_path)

    result = explorer.save_transect(cube_a, row0=0, col0=0, row1=4, col1=5)
    assert result["ok"], result

    out_dir = Path(result["dir"])
    assert result["rel"] == "figures/cube_transects/row0000_col0000_to_row0004_col0005"

    expected = {f"transect_{source}_physical.png" for source in ("pred", "gt")}
    assert set(result["files"]) == expected
    assert all((out_dir / name).stat().st_size > 0 for name in expected)


def test_save_transect_rejects_bad_space_and_unloaded(tmp_path):
    explorer, cube_a, _ = _loaded(tmp_path)

    assert not explorer.save_transect(cube_a, 0, 0, 1, 1, space="banana")["ok"]
    assert not explorer.save_transect(str(tmp_path / "nowhere"), 0, 0, 1, 1)["ok"]


def test_save_slices_includes_comparison_sources(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    result = explorer.save_slices(cube_a, row=1, col=1)
    assert result["ok"]

    expected = {f"{axis}_{source}_physical.png" for source in ("pred", "predb", "diff", "gt") for axis in ("col", "row")}
    assert set(result["files"]) == expected


def test_cmap_selection_and_diverging_override(tmp_path):
    explorer, cube_a, cube_b = _loaded(tmp_path)
    explorer.attach_second(cube_a, cube_b)

    pred = explorer._entry(cube_a, "pred")
    diff = explorer._entry(cube_a, "diff")

    assert explorer._entry_cmap(pred, "viridis") == "viridis"
    assert explorer._entry_cmap(pred, "banana") == "jet"
    assert explorer._entry_cmap(diff, "viridis") == "coolwarm"

    assert explorer.slice_png(cube_a, "pred", "col", row=0, col=0, cmap="viridis")[:4] == b"\x89PNG"
    assert explorer.plane_png(cube_a, "pred", frac=0.5, cmap="gray")[:4] == b"\x89PNG"
    assert explorer.transect_png(cube_a, "pred", 0, 0, 2, 2, cmap="inferno")[:4] == b"\x89PNG"
