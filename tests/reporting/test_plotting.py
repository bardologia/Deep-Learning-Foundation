
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy             as np
import pytest
from PIL import Image

from tools.reporting.plotting import PlotBase


@pytest.fixture
def plotter():
    return PlotBase()


@pytest.fixture
def small_field():
    rng = np.random.default_rng(0)
    return rng.normal(size=(24, 24)).astype(np.float32)


def test_apply_style_sets_dpi(plotter):
    plotter._apply_style()
    assert plt.rcParams["figure.dpi"]  == PlotBase.fig_dpi
    assert plt.rcParams["savefig.dpi"] == PlotBase.save_dpi


def test_apply_style_sets_font_family(plotter):
    plotter._apply_style()
    assert "serif" in plt.rcParams["font.family"]


def test_save_creates_file(plotter, tmp_path):
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    out = plotter._save(fig, tmp_path / "deep" / "fig.png")

    assert out.exists()
    assert out.stat().st_size > 0


def test_save_returns_path(plotter, tmp_path):
    fig = plt.figure()
    out = plotter._save(fig, tmp_path / "x.png")
    assert out == tmp_path / "x.png"


def test_save_closes_figure(plotter, tmp_path):
    fig = plt.figure()
    num = fig.number
    plotter._save(fig, tmp_path / "y.png")
    assert not plt.fignum_exists(num)


def test_shared_clim_basic():
    arr = np.linspace(0.0, 100.0, 1000).astype(np.float32)
    lo, hi = PlotBase._shared_clim(arr, q_low=1.0, q_high=99.0)
    assert lo < hi
    assert lo == pytest.approx(np.percentile(arr, 1.0))
    assert hi == pytest.approx(np.percentile(arr, 99.0))


def test_shared_clim_ignores_nan():
    arr = np.array([1.0, 2.0, 3.0, np.nan, 4.0], dtype=np.float32)
    lo, hi = PlotBase._shared_clim(arr, q_low=0.0, q_high=100.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(4.0)


def test_shared_clim_multiple_arrays():
    a = np.array([0.0, 1.0])
    b = np.array([5.0, 10.0])
    lo, hi = PlotBase._shared_clim(a, b, q_low=0.0, q_high=100.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(10.0)


def test_cmap_with_bad_returns_colormap():
    cmap = PlotBase._cmap_with_bad("viridis")
    assert isinstance(cmap, mcolors.Colormap)


def test_cmap_with_bad_sets_bad_color():
    cmap = PlotBase._cmap_with_bad("viridis", bad_color="red")
    bad  = cmap.get_bad()
    assert tuple(bad) == mcolors.to_rgba("red")


def test_imshow_figure_returns_figure(plotter, small_field):
    fig = plotter._imshow_figure(
        small_field,
        x_label = "col",
        y_label = "row",
        title   = "field",
        cmap    = "viridis",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_imshow_figure_axis_labels(plotter, small_field):
    fig = plotter._imshow_figure(
        small_field,
        x_label = "col",
        y_label = "row",
        title   = "Field Title",
        cmap    = "viridis",
    )
    ax = fig.axes[0]
    assert ax.get_xlabel() == "col"
    assert ax.get_ylabel() == "row"
    assert ax.get_title()  == "Field Title"
    plt.close(fig)


def test_imshow_figure_has_colorbar(plotter, small_field):
    fig = plotter._imshow_figure(
        small_field,
        x_label = "x",
        y_label = "y",
        title   = "t",
        cmap    = "viridis",
    )
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_imshow_figure_text_overlay(plotter, small_field):
    fig = plotter._imshow_figure(
        small_field,
        x_label      = "x",
        y_label      = "y",
        title        = "t",
        cmap         = "viridis",
        text_overlay = "RMSE=0.1",
    )
    ax    = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "RMSE=0.1" in texts
    plt.close(fig)


def test_imshow_figure_discrete_levels(plotter):
    label = np.array([[0, 1, 2], [2, 1, 0], [1, 1, 2]])
    fig   = plotter._imshow_figure(
        label.astype(float),
        x_label  = "x",
        y_label  = "y",
        title    = "labels",
        cmap     = "tab10",
        discrete = True,
        levels   = [0, 1, 2],
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_imshow_figure_saves_to_path(plotter, small_field, tmp_path):
    out = plotter._imshow_figure(
        small_field,
        x_label = "x",
        y_label = "y",
        title   = "t",
        cmap    = "viridis",
        path    = tmp_path / "img.png",
    )
    assert out == tmp_path / "img.png"
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.slow
def test_imshow_figure_saved_dpi(plotter, small_field, tmp_path):
    out = plotter._imshow_figure(
        small_field,
        x_label = "x",
        y_label = "y",
        title   = "t",
        cmap    = "viridis",
        figsize = (6.0, 4.0),
        path    = tmp_path / "dpi.png",
    )
    with Image.open(out) as img:
        dpi = img.info.get("dpi")
    assert dpi is not None
    assert round(dpi[0]) == PlotBase.save_dpi


@pytest.mark.dataset
@pytest.mark.slow
def test_imshow_input_window(plotter, inputs, tmp_path):
    window = np.asarray(inputs[0, :32, :32], dtype=np.float32)
    out    = plotter._imshow_figure(
        window,
        x_label        = "col",
        y_label        = "row",
        title          = "Input window",
        cmap           = plotter._cmap_with_bad("terrain"),
        colorbar_label = "x0",
        path           = tmp_path / "input.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.dataset
@pytest.mark.slow
def test_imshow_target_window_with_clim(plotter, targets, tmp_path):
    window = np.asarray(targets[0, :32, :32], dtype=np.float32)
    lo, hi = plotter._shared_clim(window, q_low=2.0, q_high=98.0)

    out = plotter._imshow_figure(
        window,
        x_label = "col",
        y_label = "row",
        title   = "pointwise",
        cmap    = "viridis",
        vmin    = lo,
        vmax    = hi,
        path    = tmp_path / "target.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.dataset
def test_shared_clim_target(targets):
    layer = np.asarray(targets[1, :32, :32])
    lo, hi = PlotBase._shared_clim(layer, q_low=5.0, q_high=95.0)
    assert lo <= hi
    assert np.isfinite(lo) and np.isfinite(hi)


@pytest.fixture
def paper_style():
    PlotBase.use_style("paper")
    yield
    PlotBase.use_style("report")


def test_default_style_is_report():
    assert PlotBase.style == "report"


def test_paper_style_sets_print_font_sizes(plotter, paper_style):
    plotter._apply_style()
    assert plt.rcParams["font.size"]       == 9
    assert plt.rcParams["axes.labelsize"]  == 9
    assert plt.rcParams["xtick.labelsize"] == 8
    assert plt.rcParams["legend.fontsize"] == 8


def test_paper_style_sets_stix_mathtext(plotter, paper_style):
    plotter._apply_style()
    assert plt.rcParams["mathtext.fontset"] == "stix"


def test_paper_style_sets_okabe_ito_cycle(plotter, paper_style):
    plotter._apply_style()
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == PlotBase.OKABE_ITO


def test_paper_style_overrides_instance_dpi(plotter, paper_style):
    plotter.fig_dpi  = 150
    plotter.save_dpi = 150
    plotter._apply_style()
    assert plt.rcParams["figure.dpi"]  == PlotBase.PAPER_DPI
    assert plt.rcParams["savefig.dpi"] == PlotBase.PAPER_DPI


def test_paper_style_keeps_scientific_base(plotter, paper_style):
    plotter._apply_style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert "serif" in plt.rcParams["font.family"]


def test_report_style_restores_after_paper(plotter):
    PlotBase.use_style("paper")
    plotter._apply_style()
    PlotBase.use_style("report")
    plotter._apply_style()

    assert plt.rcParams["font.size"]        == 11
    assert plt.rcParams["mathtext.fontset"] == "dejavuserif"
    assert plt.rcParams["figure.dpi"]       == plotter.fig_dpi
    assert plt.rcParams["axes.prop_cycle"]  == plt.rcParamsDefault["axes.prop_cycle"]


def test_serif_fonts_fall_back_to_stix_before_dejavu(plotter):
    plotter._apply_style()
    assert plt.rcParams["font.serif"][:3] == ["Times New Roman", "STIXGeneral", "DejaVu Serif"]


def test_figsize_full_width():
    w, h = PlotBase.figsize(PlotBase.FULL_WIDTH)
    assert w == pytest.approx(5.5)
    assert h == pytest.approx(5.5 * 0.62)


def test_figsize_custom_aspect():
    w, h = PlotBase.figsize(PlotBase.FULL_WIDTH, aspect=1.0)
    assert w == pytest.approx(5.5)
    assert h == pytest.approx(5.5)


def test_paper_save_upgrades_line_plot_to_pdf(plotter, paper_style, tmp_path):
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    out = plotter._save(fig, tmp_path / "line.png")

    assert out == tmp_path / "line.pdf"
    assert out.read_bytes()[:5] == b"%PDF-"


def test_paper_save_keeps_png_for_images(plotter, paper_style, small_field, tmp_path):
    fig = plt.figure()
    fig.add_subplot(111).imshow(small_field)
    out = plotter._save(fig, tmp_path / "map.png")

    assert out == tmp_path / "map.png"
    with Image.open(out) as img:
        assert img.format == "PNG"


def test_paper_save_respects_explicit_pdf(plotter, paper_style, tmp_path):
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    out = plotter._save(fig, tmp_path / "fig.pdf")
    assert out.read_bytes()[:5] == b"%PDF-"


def test_report_save_keeps_png(plotter, tmp_path):
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    out = plotter._save(fig, tmp_path / "line.png")

    assert out == tmp_path / "line.png"
    with Image.open(out) as img:
        assert img.format == "PNG"
