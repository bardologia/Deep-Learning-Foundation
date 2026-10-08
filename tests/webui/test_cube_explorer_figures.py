"""Tests for the cube explorer's saved-figure browser and stamp timeline.

Covers listing the figure and animation trees of a loaded stamp, path-guarded
serving of individual files, the two routes that expose both, and the timeline
of sibling inference stamps that cover the same region and z axis.
"""

from __future__ import annotations

import json
from pathlib      import Path
from urllib.parse import quote

import numpy as np

from cube_explorer        import CubeExplorer
from routers.cube_routers import CubeRouter
from routers.dispatch     import HttpExchange

from tests.webui.conftest import N_COLS, N_Z, FakeHandler, load_cube, loaded_cube, make_cube_run, make_preproc, make_stamp, open_explorer


def _with_figures(base: Path) -> tuple[CubeExplorer, str, Path]:
    """Builds a run with a small figure tree and returns the loaded explorer, its cube id and the stamp directory."""
    stamp = make_cube_run(base)

    (stamp / "figures" / "pixel_maps").mkdir(parents=True)
    (stamp / "figures" / "pixel_maps" / "mse.png").write_bytes(b"\x89PNG_fake")
    (stamp / "figures" / "notes.txt").write_text("not a figure")
    (stamp / "animations").mkdir()
    (stamp / "animations" / "walk_z.gif").write_bytes(b"GIF89a_fake")

    explorer, cube_ids = open_explorer(base, expected=1)
    load_cube(explorer, cube_ids[0])
    return explorer, cube_ids[0], stamp


def test_list_figures_groups_by_directory(tmp_path):
    """Figures group by folder and files that are not figures are skipped."""
    explorer, cube_id, _ = _with_figures(tmp_path)

    listing = explorer.list_figures(cube_id)
    assert listing["ok"]

    groups = {group["dir"]: group["files"] for group in listing["groups"]}
    assert groups == {"figures/pixel_maps": ["mse.png"], "animations": ["walk_z.gif"]}


def test_figure_file_is_path_guarded(tmp_path):
    """Serving resolves only figure files inside the stamp's figure trees."""
    explorer, cube_id, stamp = _with_figures(tmp_path)

    assert explorer.figure_file(cube_id, "figures/pixel_maps/mse.png") == (stamp / "figures" / "pixel_maps" / "mse.png").resolve()
    assert explorer.figure_file(cube_id, "figures/../metrics.json") is None
    assert explorer.figure_file(cube_id, "figures/notes.txt") is None
    assert explorer.figure_file(cube_id, "cubes/pred_curves.npy") is None
    assert explorer.figure_file(cube_id, str(stamp / "figures" / "pixel_maps" / ".." / ".." / "metrics.json")) is None
    assert explorer.figure_file("wrong", "figures/pixel_maps/mse.png") is None


def test_figure_file_refuses_a_link_out_of_the_figure_tree(tmp_path):
    """A figure-named symlink inside the tree that points outside it is listed by name but never served."""
    explorer, cube_id, stamp = _with_figures(tmp_path)

    outside = tmp_path / "secret.png"
    outside.write_bytes(b"\x89PNG_secret")
    (stamp / "figures" / "leak.png").symlink_to(outside)

    assert explorer.figure_file(cube_id, "figures/leak.png") is None


def test_list_figures_empty_stamp(tmp_path):
    """A stamp without figure trees lists zero groups."""
    explorer, cube_id = loaded_cube(tmp_path)

    listing = explorer.list_figures(cube_id)
    assert listing["ok"] and listing["groups"] == []


def test_list_figures_refuses_a_cube_that_is_not_loaded(tmp_path):
    """Listing for any id other than the loaded cube's reports that the cube is not loaded."""
    explorer, _, _ = _with_figures(tmp_path)

    assert explorer.list_figures("wrong") == {"ok": False, "error": "cube not loaded"}


def test_figure_routes_list_and_serve_through_the_cube_router(tmp_path):
    """The figures route answers the grouped listing, and the figure route serves a file's bytes or 404 outside the trees."""
    explorer, cube_id, _ = _with_figures(tmp_path)
    router               = CubeRouter(explorer)

    listing = FakeHandler("GET", f"/api/cubes/figures?id={quote(cube_id)}")
    served  = FakeHandler("GET", f"/api/cubes/figure?id={quote(cube_id)}&rel={quote('animations/walk_z.gif')}")
    escaped = FakeHandler("GET", f"/api/cubes/figure?id={quote(cube_id)}&rel={quote('figures/../metrics.json')}")

    for handler in (listing, served, escaped):
        assert router.handle(HttpExchange.of(handler))

    assert listing.status == 200 and len(listing.payload()["groups"]) == 2
    assert served.status == 200 and served.wfile.getvalue() == b"GIF89a_fake"
    assert served.sent["Content-Type"] == "image/gif" and served.sent["Cache-Control"] == "max-age=300"
    assert escaped.status == 404


def _timeline_run(base: Path) -> Path:
    """Writes a run whose inference directory holds two matching stamps, one over another region, one over other z bounds and a non-stamp folder, and returns the second matching stamp."""
    rng     = np.random.default_rng(0)
    preproc = make_preproc(base, rng)
    run     = base / "group" / "run_a"

    make_stamp(run, preproc, rng, name="stamp_1")
    second = make_stamp(run, preproc, rng, name="stamp_2")
    make_stamp(run, preproc, rng, name="stamp_other", region=[0, 4, 0, N_COLS], shape=(N_Z, 4, N_COLS))
    make_stamp(run, preproc, rng, name="stamp_shifted", x_min=-20.0)
    (run / "inference" / "seed_comparison").mkdir()

    payload = json.loads((second / "metrics.json").read_text())
    payload["curve_mse_gt"] = 0.125
    (second / "metrics.json").write_text(json.dumps(payload))

    return second


def test_timeline_lists_matching_sibling_stamps(tmp_path):
    """Sibling stamps over the same region and z bounds are listed in order with the current index and their recorded curve MSE."""
    second = _timeline_run(tmp_path)

    explorer, _ = open_explorer(tmp_path, expected=4)
    result      = explorer.timeline(str(second))

    assert result["ok"]
    assert [stamp["stamp"] for stamp in result["stamps"]] == ["stamp_1", "stamp_2"]
    assert result["current"] == 1
    assert result["stamps"][0]["mse"] is None
    assert result["stamps"][1]["mse"] == 0.125
    assert result["stamps"][1]["id"] == str(second.resolve())

    assert explorer.timeline("wrong")["ok"] is False


def test_timeline_route_answers_through_the_cube_router(tmp_path):
    """The /api/cubes/timeline route answers a stamp with its siblings and an unknown id with an error status."""
    second = _timeline_run(tmp_path)

    explorer, _ = open_explorer(tmp_path, expected=4)
    router      = CubeRouter(explorer)

    known   = FakeHandler("GET", f"/api/cubes/timeline?id={quote(str(second))}")
    unknown = FakeHandler("GET", "/api/cubes/timeline?id=wrong")

    assert router.handle(HttpExchange.of(known)) and router.handle(HttpExchange.of(unknown))
    assert known.status == 200 and known.payload()["current"] == 1
    assert unknown.status == 400
