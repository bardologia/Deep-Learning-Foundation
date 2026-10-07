"""Tests for the slice collector that exports z-profile slices across several runs.

Covers cube inspection (dimensions, available sources, intensity ranges), the primary
map, single-slice PNG rendering with and without a colour-limit override, the basket collect
operation (directory layout, per-item styling, run label disambiguation, shared colour limits,
input validation, restoration of the global plot style) and mailing a basket as zip
attachments through a stand-in SMTP server.
"""

from __future__ import annotations

import io
import json
import zipfile
from email   import message_from_bytes, policy
from pathlib import Path

import numpy as np
from PIL import Image

import tools.reporting.report_mailer as report_mailer
from cube_explorer import SliceCollectionMailer, SliceCollector
from web_logger    import WebLogger

from tests.webui.conftest import N_COLS, N_ROWS, N_Z, make_cube_run, open_explorer


def _make_run(base: Path, group: str, name: str, stamp: str = "stamp_1", sources: tuple = ("pred", "gt"), seed: int = 0) -> Path:
    """Writes one inference stamp without a dataset run behind it.

    Args:
        base    : Runs root the run is written under.
        group   : Group folder between the root and the run, or empty.
        name    : Run directory name.
        stamp   : Stamp directory name.
        sources : Curve cubes the stamp carries.
        seed    : Seed of the random cube values.

    Returns:
        The stamp directory.
    """
    rng       = np.random.default_rng(seed)
    run       = base / group / name if group else base / name
    stamp_dir = run / "inference" / stamp
    (stamp_dir / "cubes").mkdir(parents=True)

    (stamp_dir / "metrics.json").write_text(json.dumps({"x_axis_min": -10.0, "x_axis_max": 30.0}))
    for source in sources:
        np.save(stamp_dir / "cubes" / f"{source}_curves.npy", rng.random((N_Z, N_ROWS, N_COLS)).astype(np.float32))

    return stamp_dir


def _collector(base: Path) -> tuple[SliceCollector, list]:
    """Returns a collector over every stamp under ``base`` and the catalogued cube ids."""
    cubes, cube_ids = open_explorer(base)

    return SliceCollector(cubes, WebLogger()), cube_ids


def _basket(ids: list, points: list, sources: tuple = ("pred",), axes: tuple = ("col",), space: str = "physical", cmap: str = "jet") -> list:
    """Returns the basket entries a map click adds: every run, source and axis at every point.

    Args:
        ids     : Cube ids of the runs.
        points  : (row, column) pixel pairs.
        sources : Cube sources to cut.
        axes    : Cut axes.
        space   : Value space of every entry.
        cmap    : Colormap of every entry.

    Returns:
        List of basket entry dicts in point, source, axis, run order.
    """
    return [
        {"id": cube_id, "source": source, "axis": axis, "row": row, "col": col, "space": space, "cmap": cmap}
        for row, col in points
        for source in sources
        for axis in axes
        for cube_id in ids
    ]


class FakeSmtp:
    """Stand-in for smtplib.SMTP_SSL recording the login and every submitted message.

    Attributes:
        sent   : Messages submitted across every instance, as bytes.
        logins : (user, password) pairs used across every instance.
        fail   : Whether sending raises an SMTP error instead of recording.
    """

    sent   = []
    logins = []
    fail   = False

    def __init__(self, host: str, port: int, **kwargs) -> None:
        """Records nothing on connection; the host, port and any timeout are accepted as given."""
        self.host = host
        self.port = port

    def __enter__(self):
        """Returns the connection itself."""
        return self

    def __exit__(self, *exc) -> None:
        """Closes nothing."""

    def login(self, user: str, password: str) -> None:
        """Records the login pair."""
        FakeSmtp.logins.append((user, password))

    def send_message(self, message) -> None:
        """Records the message, or raises when the class is set to fail."""
        if FakeSmtp.fail:
            raise report_mailer.smtplib.SMTPServerDisconnected("connection dropped")
        FakeSmtp.sent.append(message.as_bytes())


def _mailer(base: Path, monkeypatch, password: str = "app-password", fail: bool = False) -> tuple[SliceCollectionMailer, list]:
    """Returns a basket mailer over ``base`` whose SMTP connection is a FakeSmtp, and the cube ids.

    Args:
        base        : Root the runs were written under.
        monkeypatch : Pytest fixture used to swap the SMTP class and the password variable.
        password    : Value of the password variable; empty unsets it.
        fail        : Whether the fake SMTP server refuses to send.

    Returns:
        The mailer and the catalogued cube ids.
    """
    FakeSmtp.sent, FakeSmtp.logins, FakeSmtp.fail = [], [], fail
    monkeypatch.setattr(report_mailer.smtplib, "SMTP_SSL", FakeSmtp)

    collector, ids = _collector(base)
    mailer         = SliceCollectionMailer(collector, WebLogger())
    if password:
        monkeypatch.setenv(mailer.defaults.password_env, password)
    else:
        monkeypatch.delenv(mailer.defaults.password_env, raising=False)

    return mailer, ids


def test_info_reports_dims_sources_and_intensity(tmp_path):
    """Cube info reports row and column extents, the available sources and a non-degenerate intensity range each."""
    _make_run(tmp_path, "group", "run_a")
    collector, ids = _collector(tmp_path)

    info = collector.info(ids[0])
    assert info["ok"], info
    assert info["n_rows"] == N_ROWS and info["n_cols"] == N_COLS
    assert info["sources"] == ["pred", "gt"]
    assert set(info["intensity"]) == {"pred", "gt"}
    assert all(pair[1] > pair[0] for pair in info["intensity"].values())


def test_info_rejects_ids_outside_catalogued_roots(tmp_path):
    """A stamp outside every catalogued runs root is refused."""
    _make_run(tmp_path / "runs", "group", "run_a")
    foreign = _make_run(tmp_path / "foreign", "group", "run_b")

    collector, _ = _collector(tmp_path / "runs")
    assert not collector.info(str(foreign))["ok"]


def test_slice_png_serves_with_and_without_clim_override(tmp_path):
    """A slice renders as a PNG with its own or an overriding colour scale, and a missing source or unknown axis renders nothing."""
    _make_run(tmp_path, "group", "run_a")
    collector, ids = _collector(tmp_path)

    plain  = collector.slice_png(ids[0], "pred", "col", row=3, col=2)
    scaled = collector.slice_png(ids[0], "pred", "row", row=3, col=2, vmin=0.0, vmax=2.0)
    assert plain is not None and plain[:8] == b"\x89PNG\r\n\x1a\n"
    assert scaled is not None and scaled[:8] == b"\x89PNG\r\n\x1a\n"

    assert collector.slice_png(ids[0], "full", "col", row=0, col=0) is None
    assert collector.slice_png(ids[0], "pred", "diagonal", row=0, col=0) is None


def test_primary_map_matches_the_cube_grid_and_is_cached(tmp_path):
    """The primary map is one image pixel per cube pixel, rows down and columns across, and repeat requests reuse the rendered bytes."""
    make_cube_run(tmp_path)
    collector, ids = _collector(tmp_path)

    png = collector.primary_png(ids[0])
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(png)).size == (N_COLS, N_ROWS)
    assert collector.primary_png(ids[0]) is png


def test_primary_map_without_a_dataset_run_explains_itself(tmp_path, capsys):
    """A stamp whose run names no dataset run serves no map and logs why."""
    _make_run(tmp_path, "group", "run_a")
    collector, ids = _collector(tmp_path)
    capsys.readouterr()

    assert collector.primary_png(ids[0]) is None
    assert "primary map for" in capsys.readouterr().out


def test_a_refused_slice_explains_itself_in_the_log(tmp_path, capsys):
    """An unknown cube id and a missing source each log the reason the slice was refused."""
    _make_run(tmp_path / "runs", "group", "run_a", sources=("pred",))
    foreign = _make_run(tmp_path / "foreign", "group", "run_b")

    collector, ids = _collector(tmp_path / "runs")
    capsys.readouterr()

    assert collector.slice_png(str(foreign), "pred", "col", row=0, col=0) is None
    assert collector.slice_png(ids[0], "gt", "col", row=0, col=0) is None

    logged = capsys.readouterr().out
    assert "unknown cube id" in logged
    assert "carries no 'gt' cube" in logged


def test_collect_refuses_an_existing_collection_name(tmp_path):
    """A second collection under a taken name is refused instead of overwriting the first."""
    _make_run(tmp_path, "group", "run_a", seed=0)
    collector, ids = _collector(tmp_path)

    assert collector.collect(_basket(ids, [(0, 0)]), name="twice")["ok"]

    again = collector.collect(_basket(ids, [(1, 1)]), name="twice")
    assert again["ok"] is False
    assert "already exists" in again["error"]


def test_collect_writes_figures_grouped_by_cut(tmp_path):
    """Figures land under point/axis/source/space/colormap folders, one file per run, and the manifest mirrors them."""
    _make_run(tmp_path, "group", "run_a", seed=0)
    _make_run(tmp_path, "group", "run_b", seed=1)
    collector, ids = _collector(tmp_path)

    result = collector.collect(_basket(ids, [(3, 2)], ("pred", "gt"), ("col", "row")), name="my test")
    assert result["ok"], result
    assert result["runs"] == 2 and not result["missing"]

    out_dir = Path(result["dir"])
    assert out_dir == tmp_path / "slice_collections" / "my_test"

    expected = {
        f"row0003_col0002/{axis}_{source}_physical_jet/{run}.png"
        for axis in ("col", "row")
        for source in ("pred", "gt")
        for run in ("run_a", "run_b")
    }
    assert set(result["files"]) == expected
    assert all((out_dir / rel).is_file() and (out_dir / rel).stat().st_size > 0 for rel in expected)

    manifest = json.loads((out_dir / "collection.json").read_text())
    assert {r["label"] for r in manifest["runs"]} == {"run_a", "run_b"}
    assert manifest["points"] == [{"row": 3, "col": 2}]
    assert set(manifest["files"]) == expected
    assert {item["file"] for item in manifest["items"]} == expected


def test_collect_keeps_each_items_own_cut_space_and_colormap(tmp_path):
    """Basket entries of one run at different points, spaces and colormaps each land in their own folder, and repeated entries render once."""
    _make_run(tmp_path, "group", "run_a")
    collector, ids = _collector(tmp_path)

    items  = _basket(ids, [(1, 1)]) + _basket(ids, [(4, 2)], space="normalized", cmap="viridis") + _basket(ids, [(1, 1)])
    result = collector.collect(items, name="mixed")
    assert result["ok"], result
    assert set(result["files"]) == {"row0001_col0001/col_pred_physical_jet/run_a.png", "row0004_col0002/col_pred_normalized_viridis/run_a.png"}


def test_collect_shared_clim_spans_all_runs(tmp_path):
    """Shared colour limits span the intensity ranges of every basket run, and are absent when per-run scaling is asked for."""
    _make_run(tmp_path, "group", "run_a", seed=0)
    _make_run(tmp_path, "group", "run_b", seed=1)
    collector, ids = _collector(tmp_path)

    intensities = [collector.info(i)["intensity"]["pred"] for i in ids]
    result      = collector.collect(_basket(ids, [(0, 0)]), shared=True, name="clim")
    assert result["ok"], result

    manifest = json.loads((Path(result["dir"]) / "collection.json").read_text())
    assert manifest["clims"]["pred"] == [min(p[0] for p in intensities), max(p[1] for p in intensities)]

    per_run = collector.collect(_basket(ids, [(0, 0)]), shared=False, name="perrun")
    assert json.loads((Path(per_run["dir"]) / "collection.json").read_text())["clims"] == {}


def test_collect_disambiguates_duplicate_run_names(tmp_path):
    """Runs sharing a name in different groups are labelled with their group."""
    _make_run(tmp_path, "group_a", "run", seed=0)
    _make_run(tmp_path, "group_b", "run", seed=1)
    collector, ids = _collector(tmp_path)

    result = collector.collect(_basket(ids, [(0, 0)]), name="dupes")
    assert result["ok"], result

    labels = {r["label"] for r in json.loads((Path(result["dir"]) / "collection.json").read_text())["runs"]}
    assert labels == {"group_a__run", "group_b__run"}


def test_collect_disambiguates_stamps_of_the_same_run(tmp_path):
    """Two stamps of one run are labelled with their stamp names."""
    _make_run(tmp_path, "group", "run", stamp="stamp_1", seed=0)
    _make_run(tmp_path, "group", "run", stamp="stamp_2", seed=1)
    collector, ids = _collector(tmp_path)

    result = collector.collect(_basket(ids, [(0, 0)]), name="stamps")
    assert result["ok"], result

    labels = {r["label"] for r in json.loads((Path(result["dir"]) / "collection.json").read_text())["runs"]}
    assert labels == {"group__run__stamp_1", "group__run__stamp_2"}


def test_collect_skips_runs_missing_a_source(tmp_path):
    """Entries whose run lacks the source are reported as missing, and a basket nothing can render is refused."""
    _make_run(tmp_path, "group", "run_a", sources=("pred", "gt"), seed=0)
    _make_run(tmp_path, "group", "run_b", sources=("pred",), seed=1)
    collector, ids = _collector(tmp_path)

    result = collector.collect(_basket(ids, [(0, 0)], ("gt",)), name="partial")
    assert result["ok"], result
    assert result["files"] == ["row0000_col0000/col_gt_physical_jet/run_a.png"]
    assert result["missing"] == [{"label": "run_b", "source": "gt"}]

    empty = collector.collect(_basket(ids, [(0, 0)], ("full",)), name="empty")
    assert not empty["ok"]


def test_collect_rejects_bad_inputs(tmp_path):
    """An empty basket, entries lacking a key, unknown sources, axes, spaces or colormaps, negative indices and unknown runs are all refused."""
    _make_run(tmp_path, "group", "run_a")
    collector, ids = _collector(tmp_path)

    good = _basket(ids, [(0, 0)])[0]

    assert not collector.collect([])["ok"]
    assert "lacks" in collector.collect([{k: v for k, v in good.items() if k != "cmap"}])["error"]
    assert not collector.collect([{**good, "source": "banana"}])["ok"]
    assert not collector.collect([{**good, "axis": "diagonal"}])["ok"]
    assert not collector.collect([{**good, "space": "banana"}])["ok"]
    assert not collector.collect([{**good, "cmap": "rainbow"}])["ok"]
    assert not collector.collect([{**good, "row": -1}])["ok"]
    assert not collector.collect([{**good, "id": str(tmp_path / "nowhere")}])["ok"]


def test_collect_restores_figure_style(tmp_path):
    """Rendering a collection in the paper style leaves the process-wide style as it was."""
    _make_run(tmp_path, "group", "run_a")
    collector, ids = _collector(tmp_path)

    from tools.reporting.plotting import PlotBase

    assert collector.collect(_basket(ids, [(0, 0)]), name="style")["ok"]
    assert PlotBase.style == "report"


def test_mail_settings_report_the_default_recipient_and_password_state(tmp_path, monkeypatch):
    """The form defaults carry the configured recipient and whether the password variable is set."""
    _make_run(tmp_path, "group", "run_a")

    mailer, _ = _mailer(tmp_path, monkeypatch)
    settings  = mailer.settings()
    assert settings["recipient"] == mailer.defaults.recipient
    assert settings["ready"] is True

    monkeypatch.delenv(mailer.defaults.password_env)
    assert mailer.settings()["ready"] is False


def test_mail_sends_the_rendered_collection_as_one_zip(tmp_path, monkeypatch):
    """Mailing renders the basket to disk, zips it under the collection name and sends one message to every recipient."""
    _make_run(tmp_path, "group", "run_a", seed=0)
    _make_run(tmp_path, "group", "run_b", seed=1)
    mailer, ids = _mailer(tmp_path, monkeypatch)

    result = mailer.mail(_basket(ids, [(3, 2)], ("pred", "gt")), name="for review", recipient="a@example.org; b@example.org", note="Cuts through the bright edge.")
    assert result["ok"], result
    assert result["recipient"] == "a@example.org, b@example.org"
    assert result["volumes"] == ["for_review.zip"]

    assert FakeSmtp.logins == [(mailer.defaults.sender, "app-password")]
    assert len(FakeSmtp.sent) == 1

    message = message_from_bytes(FakeSmtp.sent[0], policy=policy.default)
    assert message["To"] == "a@example.org, b@example.org"
    assert message["Subject"] == f"{mailer.defaults.subject_prefix}: slice collection for_review"

    parts = [part for part in message.walk() if part.get_filename()]
    body  = next(part for part in message.walk() if part.get_content_type() == "text/plain").get_content()
    assert body.startswith("Cuts through the bright edge.")
    assert "4 figures from 2 runs" in body and "row 3 col 2" in body

    archive = zipfile.ZipFile(io.BytesIO(parts[0].get_payload(decode=True)))
    assert set(archive.namelist()) == {"for_review/collection.json"} | {f"for_review/{rel}" for rel in result["files"]}
    assert (Path(result["dir"]) / "mail" / "for_review.zip").is_file()


def test_mail_refuses_before_rendering_without_recipient_or_password(tmp_path, monkeypatch):
    """A malformed recipient or an unset password variable is refused before any figure is written."""
    _make_run(tmp_path, "group", "run_a")

    mailer, ids = _mailer(tmp_path, monkeypatch)
    assert "malformed" in mailer.mail(_basket(ids, [(0, 0)]), recipient="not-an-address")["error"]
    assert "at least one" in mailer.mail(_basket(ids, [(0, 0)]), recipient=" , ")["error"]

    mailer, ids = _mailer(tmp_path, monkeypatch, password="")
    assert "SMTP password not found" in mailer.mail(_basket(ids, [(0, 0)]), recipient="a@example.org")["error"]

    assert not (tmp_path / "slice_collections").exists()
    assert FakeSmtp.sent == []


def test_mail_failure_after_rendering_keeps_the_collection(tmp_path, monkeypatch):
    """When the SMTP server drops the message, the error names the saved folder, which stays on disk."""
    _make_run(tmp_path, "group", "run_a")
    mailer, ids = _mailer(tmp_path, monkeypatch, fail=True)

    result = mailer.mail(_basket(ids, [(0, 0)]), name="dropped", recipient="a@example.org")
    assert result["ok"] is False
    assert "mailing failed" in result["error"]
    assert (Path(result["dir"]) / "collection.json").is_file()


def test_mail_reuses_a_basket_saved_under_the_same_name(tmp_path, monkeypatch):
    """Mailing a basket already saved under its name sends that folder without rendering it again, while another basket under the name is refused."""
    _make_run(tmp_path, "group", "run_a")
    mailer, ids = _mailer(tmp_path, monkeypatch)

    basket = _basket(ids, [(1, 1)], ("pred", "gt"))
    saved  = mailer.collector.collect(basket, name="meeting")
    assert saved["ok"] and saved["reused"] is False

    figure = Path(saved["dir"]) / saved["files"][0]
    stamp  = figure.stat().st_mtime_ns

    mailed = mailer.mail(basket, name="meeting", recipient="a@example.org")
    assert mailed["ok"], mailed
    assert mailed["reused"] is True
    assert mailed["files"] == saved["files"]
    assert figure.stat().st_mtime_ns == stamp
    assert len(FakeSmtp.sent) == 1

    other = mailer.mail(_basket(ids, [(2, 2)]), name="meeting", recipient="a@example.org")
    assert other["ok"] is False
    assert "for a different basket" in other["error"]
    assert len(FakeSmtp.sent) == 1
