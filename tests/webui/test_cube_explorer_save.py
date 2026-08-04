from __future__ import annotations

from pathlib import Path

from cube_explorer import CubeExplorer
from web_logger    import WebLogger

from tests.webui.conftest import N_AZ, loaded_cube, make_cube_run

SOURCES = ("pred", "gt", "reduced")


def _loaded_explorer(base: Path) -> tuple[CubeExplorer, str]:
    return loaded_cube(base, sources=SOURCES)


def test_save_slices_writes_paper_figures(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    result = explorer.save_slices(cube_id, az=3, rg=2, space="physical")
    assert result["ok"], result

    out_dir = Path(result["dir"])
    assert out_dir == Path(cube_id).parent.parent / "figures" / "cube_slices" / "az0003_rg0002"
    assert result["rel"] == "figures/cube_slices/az0003_rg0002"

    expected = {f"{axis}_{source}_physical.png" for source in SOURCES for axis in ("range", "azimuth")}
    assert set(result["files"]) == expected

    for name in expected:
        target = out_dir / name
        assert target.is_file() and target.stat().st_size > 0


def test_save_slices_normalized_space_clips_indices(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    result = explorer.save_slices(cube_id, az=999, rg=-5, space="normalized")
    assert result["ok"], result
    assert result["az"] == N_AZ - 1 and result["rg"] == 0
    assert all(name.endswith("_normalized.png") for name in result["files"])
    assert all((Path(result["dir"]) / name).is_file() for name in result["files"])


def test_save_slices_restores_figure_style(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    from tools.reporting.plotting import PlotBase

    assert explorer.save_slices(cube_id, az=1, rg=1)["ok"]
    assert PlotBase.style == "report"


def test_save_slices_rejects_unknown_space(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    result = explorer.save_slices(cube_id, az=0, rg=0, space="banana")
    assert not result["ok"]


def test_save_slices_requires_loaded_cube(tmp_path):
    stamp    = make_cube_run(tmp_path, sources=SOURCES)
    explorer = CubeExplorer(WebLogger())

    result = explorer.save_slices(str(stamp), az=0, rg=0)
    assert not result["ok"]


def test_slice_png_still_serves_after_cut_refactor(tmp_path):
    explorer, cube_id = _loaded_explorer(tmp_path)

    png = explorer.slice_png(cube_id, "pred", "range", az=0, rg=2)
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
