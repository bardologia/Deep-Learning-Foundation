"""Animated walks through predicted and ground-truth volumetric cubes.

A cube pair is sliced along z, col or row and each slice is rendered in a
worker process as a side-by-side panel of ground truth, prediction and
absolute error, then assembled into a GIF. Optionally every cube
(prediction, ground truth, raw cube) is also written as its own
single-panel GIF whose figure follows the slice aspect ratio, so the image
is not stretched into a square box.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses        import dataclass
from io                 import BytesIO
from pathlib            import Path
from typing             import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np
from PIL import Image

from pipelines.backbone.inference.plots import Plotter
from tools.monitoring.logger            import Logger
from tools.reporting.plotting           import PlotBase


@dataclass
class FrameSpec:
    """Everything one worker process needs to render a single GIF frame.

    Attributes:
        frame_order : Position of this frame in the animation.
        n_frames    : Total number of frames, used for the progress bar.
        gt          : Ground-truth slice of shape (rows, columns).
        pred        : Predicted slice of the same shape.
        vmin        : Lower intensity limit shared by the GT and prediction panels.
        vmax        : Upper intensity limit shared by the GT and prediction panels.
        emax_gt     : Upper limit of the absolute-error panel.
        extent      : Axis extent of the slice in scene coordinates.
        x_label     : Horizontal axis label.
        y_label     : Vertical axis label.
        cmap        : Colormap of the GT and prediction panels.
        err_cmap    : Colormap of the error panel.
        dpi         : Rendering resolution.
        origin      : Image origin, "upper" or "lower".
        title       : Figure title carrying the slice coordinate.
        full        : Raw cube slice of the same shape, or None when not shown.
    """

    frame_order : int
    n_frames    : int
    gt          : np.ndarray
    pred        : np.ndarray
    vmin        : float
    vmax        : float
    emax_gt     : float
    extent      : list
    x_label     : str
    y_label     : str
    cmap        : str
    err_cmap    : str
    dpi         : int
    origin      : str
    title       : str
    full        : np.ndarray | None = None


@dataclass
class PanelFrameSpec:
    """Everything one worker process needs to render a single-cube GIF frame.

    Attributes:
        frame_order : Position of this frame in the animation.
        n_frames    : Total number of frames, used for the progress bar.
        data        : Cube slice of shape (rows, columns).
        vmin        : Lower intensity limit of the panel.
        vmax        : Upper intensity limit of the panel.
        extent      : Axis extent of the slice in scene coordinates.
        x_label     : Horizontal axis label.
        y_label     : Vertical axis label.
        cmap        : Colormap of the panel.
        dpi         : Rendering resolution.
        origin      : Image origin, "upper" or "lower".
        title       : Figure title carrying the slice coordinate.
        panel_title : Panel title naming the cube.
        figsize     : Figure size in inches as (width, height), following the slice aspect ratio.
    """

    frame_order : int
    n_frames    : int
    data        : np.ndarray
    vmin        : float
    vmax        : float
    extent      : list
    x_label     : str
    y_label     : str
    cmap        : str
    dpi         : int
    origin      : str
    title       : str
    panel_title : str
    figsize     : tuple


class Animator:
    """Renders cube-walk GIFs comparing predicted and ground-truth cubes.

    Layout constants give every panel one image box sized to the slice aspect
    ratio (PANEL_LONG_SIDE_IN for single-cube GIFs, COMPOSITE_LONG_SIDE_IN for
    the composite, PANEL_SHORT_SIDE_IN as the floor of the short side), the gap
    between panels as a fraction of the box (PANEL_GAP), the colorbar width and
    the clearance after its tick labels relative to a panel (COLORBAR_RATIO,
    COLORBAR_CLEARANCE), the progress strip height as a figure fraction
    (PROGRESS_HEIGHT) and the figure margins in inches (MARGIN_*).

    Attributes:
        logger          : Logger for the configuration banner and written-GIF lines.
        cmap            : Colormap of the GT and prediction panels.
        err_cmap        : Colormap of the absolute-error panel.
        dpi             : Rendering resolution of each frame.
        fps             : Playback rate of the GIF when no total duration is set.
        duration        : Total playback length of each GIF in seconds, or None to let fps govern.
        max_frames      : Cap on the number of frames per GIF.
        separate_panels : Whether every cube is also written as its own single-panel GIF.
        num_workers     : Render worker processes, or None to size from the CPU count.
    """

    PANEL_TITLES = {
        "pred" : "Prediction",
        "gt"   : "GT (Gaussian)",
        "full" : "Full cube (raw)",
    }

    PANEL_LONG_SIDE_IN     = 6.0
    COMPOSITE_LONG_SIDE_IN = 4.5
    PANEL_SHORT_SIDE_IN    = 2.4
    PANEL_GAP              = 0.08
    COLORBAR_RATIO         = 0.035
    COLORBAR_CLEARANCE     = 0.22
    PROGRESS_HEIGHT        = 0.008
    MARGIN_LEFT            = 0.85
    MARGIN_RIGHT           = 0.75
    MARGIN_TOP             = 0.75
    MARGIN_BOTTOM          = 0.65
    MARGIN_W_IN            = MARGIN_LEFT + MARGIN_RIGHT
    MARGIN_H_IN            = MARGIN_TOP + MARGIN_BOTTOM

    def __init__(
        self,
        logger          : Logger,
        *,
        cmap            : str = "jet",
        err_cmap        : str = "magma",
        dpi             : int = 110,
        fps             : int = 12,
        duration        : float | None = None,
        max_frames      : int = 150,
        separate_panels : bool = False,
        num_workers     : int | None = None,
    ) -> None:
        """Stores the rendering settings and logs them.

        Args:
            logger          : Logger for the configuration banner and written-GIF lines.
            cmap            : Colormap of the GT and prediction panels.
            err_cmap        : Colormap of the absolute-error panel.
            dpi             : Rendering resolution of each frame.
            fps             : Playback rate of the GIF when duration is None.
            duration        : Total playback length of each GIF in seconds; the per-frame delay is duration divided by the frame count. None lets fps govern.
            max_frames      : Cap on the number of frames per GIF; slices are subsampled evenly when the axis is longer.
            separate_panels : Whether every cube is also written as its own single-panel GIF sized to the slice aspect ratio.
            num_workers     : Render worker processes, or None to size from the CPU count.

        Raises:
            ValueError : If max_frames is below one, or duration is set and not positive.
        """
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}; use a large value instead of 0 to leave GIFs uncapped.")

        if duration is not None and duration <= 0.0:
            raise ValueError(f"duration must be positive seconds or None, got {duration}.")

        self.logger          = logger
        self.cmap            = cmap
        self.err_cmap        = err_cmap
        self.dpi             = dpi
        self.fps             = fps
        self.duration        = duration
        self.max_frames      = max_frames
        self.separate_panels = separate_panels
        self.num_workers     = num_workers

        self.logger.section("[Animator]")
        self.logger.subsection(f"colormap for GT and prediction : {cmap}")
        self.logger.subsection(f"colormap for error             : {err_cmap}")
        self.logger.subsection(f"figure DPI                     : {dpi}")
        self.logger.subsection(f"GIF frames per second          : {fps if duration is None else f'ignored (duration {duration:.2f} s)'}")
        self.logger.subsection(f"GIF total duration             : {'from fps' if duration is None else f'{duration:.2f} s'}")
        self.logger.subsection(f"max frames per GIF             : {max_frames}")
        self.logger.subsection(f"separate single-cube GIFs      : {separate_panels}")
        self.logger.subsection(f"CPU workers for rendering      : {num_workers if num_workers is not None else 'auto'} \n")

    def _build_axis(self, axis: str, cubes: tuple, z_axis: np.ndarray, row_offset: int, col_offset: int) -> dict:
        """Builds the slicing and labelling recipe for one walk axis.

        Args:
            axis       : Walk axis, "z", "col" or "row".
            cubes      : Cubes to slice, each of shape (n_z, rows, cols).
            z_axis     : The z axis, shape (n_z,).
            row_offset : Row index of the cube's first row in the full scene.
            col_offset : Column index of the cube's first column in the full scene.

        Returns:
            A dict holding the slice count, the slicing callable, the axis extent,
            both axis labels, the per-index title function and the image origin.

        Raises:
            ValueError : If the axis name is not one of the three supported walks.
        """
        N_z, row, col = cubes[0].shape
        sort_idx = None

        if axis in ("col", "row"):
            sort_idx = np.argsort(z_axis)
            z_axis   = z_axis[sort_idx]

        if axis == "z":
            return dict(
                n_total   = N_z,
                get_slice = lambda i: self._slice_z(cubes, i),
                extent    = [col_offset, col_offset + col, row_offset + row, row_offset],
                x_label   = "col index",
                y_label   = "row index",
                title_fn  = lambda i: f"z = {z_axis[i]:.2f} m  (idx {i}/{N_z - 1})",
                origin    = "upper",
            )

        if axis == "col":
            return dict(
                n_total   = col,
                get_slice = lambda i: self._slice_col(cubes, sort_idx, i),
                extent    = [row_offset, row_offset + row, float(z_axis[0]), float(z_axis[-1])],
                x_label   = "row index",
                y_label   = "z",
                title_fn  = lambda i: f"col = {i + col_offset}",
                origin    = "lower",
            )

        if axis == "row":
            return dict(
                n_total   = row,
                get_slice = lambda i: self._slice_row(cubes, sort_idx, i),
                extent    = [col_offset, col_offset + col, float(z_axis[0]), float(z_axis[-1])],
                x_label   = "col index",
                y_label   = "z",
                title_fn  = lambda i: f"row = {i + row_offset}",
                origin    = "lower",
            )

        raise ValueError(f"axis must be z|col|row, got {axis!r}")

    @staticmethod
    def _slice_z(cubes: tuple, i: int) -> tuple:
        """Returns the (row, col) slice of every cube at z bin i."""
        return tuple(cube[i] if cube is not None else None for cube in cubes)

    @staticmethod
    def _slice_col(cubes: tuple, sort_idx: np.ndarray | None, i: int) -> tuple:
        """Returns the (z, row) slice of every cube at column i, z-sorted."""
        slices = tuple(cube[:, :, i] if cube is not None else None for cube in cubes)
        if sort_idx is not None:
            slices = tuple(s[sort_idx] if s is not None else None for s in slices)
        return slices

    @staticmethod
    def _slice_row(cubes: tuple, sort_idx: np.ndarray | None, i: int) -> tuple:
        """Returns the (z, col) slice of every cube at row i, z-sorted."""
        slices = tuple(cube[:, i, :] if cube is not None else None for cube in cubes)
        if sort_idx is not None:
            slices = tuple(s[sort_idx] if s is not None else None for s in slices)
        return slices

    def _render(self, specs: Iterator, n_frames: int, label: str, render=None) -> dict[int, bytes]:
        """Renders the frame specs in a process pool with a bounded submission window.

        Progress goes through the logger's bar, bound to the shared console, so
        the tile shows which GIF is rendering and no second progress writer
        floods the detached .out file.

        Args:
            specs    : Frame specifications, consumed lazily.
            n_frames : Total number of frames, for the progress bar and pool sizing.
            label    : Progress-bar description naming the GIF being rendered.
            render   : Picklable callable turning one spec into (order, png bytes); defaults to the composite frame renderer.

        Returns:
            Mapping from frame order to the encoded PNG bytes.
        """
        render    = Animator._render_frame if render is None else render
        n_workers = self.num_workers if self.num_workers is not None else min(n_frames, os.cpu_count() or 1)
        window    = 2 * n_workers
        png_bytes: dict[int, bytes] = {}

        rc = {k: plt.rcParams[k] for k in (*PlotBase.SCIENTIFIC_RC, *PlotBase.STYLE_RC[PlotBase.style], "figure.dpi", "savefig.dpi")}

        with ProcessPoolExecutor(max_workers=n_workers, initializer=Animator._init_worker, initargs=(rc,)) as pool, self.logger.track() as progress:
            task         = progress.add_task(label, total=n_frames)
            pending: set = set()

            for spec in specs:
                if len(pending) >= window:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        order, data = fut.result()
                        png_bytes[order] = data
                        progress.advance(task)
                pending.add(pool.submit(render, spec))

            for fut in as_completed(pending):
                order, data = fut.result()
                png_bytes[order] = data
                progress.advance(task)

        return png_bytes

    @classmethod
    def _panel_box(cls, rows: int, cols: int, long_side: float) -> tuple:
        """Returns the image box size in inches that follows the slice aspect ratio.

        The longer slice side spans long_side inches and the shorter one scales
        with the ratio, floored at PANEL_SHORT_SIDE_IN so ticks and the colorbar
        stay legible on very elongated slices.

        Args:
            rows      : Number of slice rows.
            cols      : Number of slice columns.
            long_side : Length of the longer box side in inches.

        Returns:
            Tuple of (width, height) of the image box in inches.
        """
        if cols >= rows:
            return (long_side, max(cls.PANEL_SHORT_SIDE_IN, long_side * rows / cols))

        return (max(cls.PANEL_SHORT_SIDE_IN, long_side * cols / rows), long_side)

    @classmethod
    def _panel_figsize(cls, rows: int, cols: int) -> tuple:
        """Returns the single-panel figure size: the image box plus margins for ticks, labels and the colorbar.

        Args:
            rows : Number of slice rows.
            cols : Number of slice columns.

        Returns:
            Tuple of (width, height) of the figure in inches.
        """
        box_w, box_h = cls._panel_box(rows, cols, cls.PANEL_LONG_SIDE_IN)

        return (box_w + cls.MARGIN_W_IN, box_h + cls.MARGIN_H_IN)

    @staticmethod
    def _draw_progress(fig, frame_order: int, n_frames: int) -> None:
        """Draws the thin playback progress strip along the bottom edge of the figure.

        Args:
            fig         : Figure receiving the strip.
            frame_order : Position of the current frame.
            n_frames    : Total number of frames.
        """
        progress = (frame_order + 1) / max(1, n_frames)
        ax       = fig.add_axes([0.0, 0.0, 1.0, Animator.PROGRESS_HEIGHT])

        ax.barh(0, progress,       height=1, color="steelblue", left=0.0)
        ax.barh(0, 1.0 - progress, height=1, color="#333333",   left=progress)
        ax.set_xlim(0, 1)
        ax.set_axis_off()

    @staticmethod
    def _draw_image(ax, data: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, extent: list, origin: str):
        """Draws one slice into an axes that fills its box, titled, with ticks inward.

        Args:
            ax     : Axes receiving the image.
            data   : Slice of shape (rows, columns).
            title  : Panel title.
            cmap   : Colormap name.
            vmin   : Lower colour limit.
            vmax   : Upper colour limit.
            extent : Data coordinates of the image edges.
            origin : Row origin, 'upper' or 'lower'.

        Returns:
            The AxesImage, for colorbar attachment.
        """
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto", origin=origin, interpolation="nearest")
        ax.set_title(title)

        return im

    @staticmethod
    def _finish_frame(fig, spec, dpi: int) -> tuple[int, bytes]:
        """Titles the figure, rasterises it at the full figure size and returns the PNG bytes.

        The tight bounding box of the publication rcParams is switched off here, since
        every GIF frame must have identical pixel dimensions.

        Args:
            fig  : Figure to rasterise.
            spec : Frame specification carrying the title and frame order.
            dpi  : Rendering resolution.

        Returns:
            A tuple of the frame order and the encoded PNG bytes.
        """
        fig.suptitle(spec.title, y=0.995, va="top")

        buf = BytesIO()
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)

        return spec.frame_order, buf.read()

    @staticmethod
    def _render_frame(spec: FrameSpec) -> tuple[int, bytes]:
        """Renders one composite frame into PNG bytes inside a worker process.

        Every panel gets the same image box sized to the slice aspect ratio. The
        vertical axis is labelled and ticked once on the left, the horizontal
        axis once below the row, the intensity panels share one colorbar and the
        error panel has its own.

        Args:
            spec : Frame specification holding the slices, limits and labels.

        Returns:
            A tuple of the frame order and the encoded PNG bytes.
        """
        g, p = spec.gt, spec.pred
        eg   = np.abs(p - g)

        intensity = []
        if spec.full is not None:
            intensity.append((spec.full, Animator.PANEL_TITLES["full"]))
        intensity += [(g, Animator.PANEL_TITLES["gt"]), (p, Animator.PANEL_TITLES["pred"])]

        n_int        = len(intensity)
        box_w, box_h = Animator._panel_box(*g.shape, Animator.COMPOSITE_LONG_SIDE_IN)
        cb_ratio     = Animator.COLORBAR_RATIO
        widths       = [1.0] * n_int + [cb_ratio, Animator.COLORBAR_CLEARANCE, 1.0, cb_ratio]
        fig_w        = box_w * (sum(widths) + Animator.PANEL_GAP * len(widths)) + Animator.MARGIN_W_IN
        fig_h        = box_h + Animator.MARGIN_H_IN

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=Animator.PANEL_GAP)

        axes = []
        for k, (data, title) in enumerate(intensity):
            ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
            im = Animator._draw_image(ax, data, title, spec.cmap, spec.vmin, spec.vmax, spec.extent, spec.origin)
            axes.append(ax)

        fig.colorbar(im, cax=fig.add_subplot(gs[0, n_int]), label="intensity")

        ax_err = fig.add_subplot(gs[0, n_int + 2], sharey=axes[0])
        im_err = Animator._draw_image(ax_err, eg, "|Pred - GT|", spec.err_cmap, 0.0, spec.emax_gt, spec.extent, spec.origin)
        axes.append(ax_err)

        fig.colorbar(im_err, cax=fig.add_subplot(gs[0, n_int + 3]), label="|error|")
        Animator._draw_progress(fig, spec.frame_order, spec.n_frames)

        for ax in axes[1:]:
            ax.tick_params(labelleft=False)

        axes[0].set_ylabel(spec.y_label)
        fig.supxlabel(spec.x_label, y=0.01, va="bottom")
        fig.subplots_adjust(left=Animator.MARGIN_LEFT / fig_w, right=1.0 - Animator.MARGIN_RIGHT / fig_w, top=1.0 - Animator.MARGIN_TOP / fig_h, bottom=Animator.MARGIN_BOTTOM / fig_h)

        return Animator._finish_frame(fig, spec, spec.dpi)

    @staticmethod
    def _init_worker(rc: dict) -> None:
        """Selects the headless backend and applies the publication rcParams inside a render worker.

        Args:
            rc : rcParams captured in the parent process, so forked and spawned workers draw identically.
        """
        matplotlib.use("Agg")
        plt.rcParams.update(rc)

    def _frame_specs(
        self,
        frame_indices : np.ndarray,
        get_slice,
        axis_spec     : dict,
        full_pos      : int | None,
        vmin          : float,
        vmax          : float,
        emax_gt       : float,
    ) -> Iterator[FrameSpec]:
        """Yields one frame specification per selected slice index.

        Args:
            frame_indices : Slice indices to render along the walk axis.
            get_slice     : Callable returning the cube slices at one index.
            axis_spec     : Axis recipe holding the extent, labels, origin and titles.
            full_pos      : Position of the raw cube in the slice tuple, or None when absent.
            vmin          : Lower intensity limit shared by the GT and prediction panels.
            vmax          : Upper intensity limit shared by the GT and prediction panels.
            emax_gt       : Upper limit of the absolute-error panel.

        Yields:
            The frame specification of each slice, in animation order.
        """

        n_frames = len(frame_indices)

        for frame_order, fi in enumerate(frame_indices):
            i     = int(fi)
            parts = get_slice(i)
            p, g  = parts[0], parts[1]
            f     = parts[full_pos] if full_pos is not None else None

            yield FrameSpec(
                frame_order = frame_order,
                n_frames    = n_frames,
                gt          = g.copy(),
                pred        = p.copy(),
                full        = f.copy() if f is not None else None,
                vmin        = vmin,
                vmax        = vmax,
                emax_gt     = emax_gt,
                extent      = axis_spec["extent"],
                x_label     = axis_spec["x_label"],
                y_label     = axis_spec["y_label"],
                cmap        = self.cmap,
                err_cmap    = self.err_cmap,
                dpi         = self.dpi,
                origin      = axis_spec["origin"],
                title       = axis_spec["title_fn"](i),
            )

    @staticmethod
    def _render_panel_frame(spec: PanelFrameSpec) -> tuple[int, bytes]:
        """Renders one single-cube frame into PNG bytes inside a worker process.

        Args:
            spec : Frame specification holding the slice, limits, labels and figure size.

        Returns:
            A tuple of the frame order and the encoded PNG bytes.
        """
        fig_w, fig_h = spec.figsize
        fig          = plt.figure(figsize=spec.figsize)
        gs           = fig.add_gridspec(1, 2, width_ratios=[1.0, Animator.COLORBAR_RATIO], wspace=Animator.PANEL_GAP)

        ax = fig.add_subplot(gs[0, 0])
        im = Animator._draw_image(ax, spec.data, spec.panel_title, spec.cmap, spec.vmin, spec.vmax, spec.extent, spec.origin)
        fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]), label="intensity")
        Animator._draw_progress(fig, spec.frame_order, spec.n_frames)

        ax.set_ylabel(spec.y_label)
        ax.set_xlabel(spec.x_label)
        fig.subplots_adjust(left=Animator.MARGIN_LEFT / fig_w, right=1.0 - Animator.MARGIN_RIGHT / fig_w, top=1.0 - Animator.MARGIN_TOP / fig_h, bottom=Animator.MARGIN_BOTTOM / fig_h)

        return Animator._finish_frame(fig, spec, spec.dpi)

    def _panel_specs(
        self,
        frame_indices : np.ndarray,
        get_slice,
        axis_spec     : dict,
        cube_pos      : int,
        panel_title   : str,
        vmin          : float,
        vmax          : float,
    ) -> Iterator[PanelFrameSpec]:
        """Yields one single-cube frame specification per selected slice index.

        Args:
            frame_indices : Slice indices to render along the walk axis.
            get_slice     : Callable returning the cube slices at one index.
            axis_spec     : Axis recipe holding the extent, labels, origin and titles.
            cube_pos      : Position of the animated cube in the slice tuple.
            panel_title   : Panel title naming the cube.
            vmin          : Lower intensity limit of the panel.
            vmax          : Upper intensity limit of the panel.

        Yields:
            The frame specification of each slice, in animation order.
        """

        n_frames = len(frame_indices)

        for frame_order, fi in enumerate(frame_indices):
            i    = int(fi)
            data = get_slice(i)[cube_pos]

            yield PanelFrameSpec(
                frame_order = frame_order,
                n_frames    = n_frames,
                data        = data.copy(),
                vmin        = vmin,
                vmax        = vmax,
                extent      = axis_spec["extent"],
                x_label     = axis_spec["x_label"],
                y_label     = axis_spec["y_label"],
                cmap        = self.cmap,
                dpi         = self.dpi,
                origin      = axis_spec["origin"],
                title       = axis_spec["title_fn"](i),
                panel_title = panel_title,
                figsize     = self._panel_figsize(*data.shape),
            )

    def _frame_delay_ms(self, n_frames: int) -> int:
        """Returns the per-frame delay in milliseconds from the duration or the fps setting.

        Args:
            n_frames : Number of frames in the GIF.

        Returns:
            Delay between frames in whole milliseconds, at least one.
        """
        if self.duration is not None:
            return max(1, int(round(1000.0 * self.duration / max(1, n_frames))))

        return max(1, int(round(1000.0 / max(1, self.fps))))

    def _write_gif(self, png_bytes: dict[int, bytes], out_path: Path) -> None:
        """Assembles rendered PNG frames into a looping GIF on disk.

        Args:
            png_bytes : Mapping from frame order to encoded PNG bytes; consumed.
            out_path  : Destination GIF path.
        """
        n_frames = len(png_bytes)
        decoded  = (Image.open(BytesIO(png_bytes.pop(k))).convert("P", dither=Image.Dither.NONE) for k in sorted(png_bytes))

        out_path.parent.mkdir(parents=True, exist_ok=True)

        next(decoded).save(
            fp            = str(out_path),
            format        = "GIF",
            save_all      = True,
            append_images = decoded,
            loop          = 0,
            duration      = self._frame_delay_ms(n_frames),
            optimize      = False,
        )

    def _log_written(self, axis: str, out_path: Path, n_frames: int, n_total: int, started: float) -> None:
        """Logs a written GIF with its frame count against the slice count, the frame delay and the render time.

        Args:
            axis     : Walk axis of the GIF.
            out_path : Written GIF path.
            n_frames : Frames the GIF holds, after subsampling to max_frames.
            n_total  : Slices along the axis the frames were drawn from.
            started  : perf_counter reading taken before rendering began.
        """
        elapsed = time.perf_counter() - started
        self.logger.subsection(f"GIF ({axis:<3}) : {out_path}  [{n_frames} of {n_total} slices, {self._frame_delay_ms(n_frames)} ms/frame, {elapsed:.1f} s]")

    def walk_gif(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        axis       : str,
        out_path   : Path,
        *,
        z_axis     : np.ndarray,
        row_offset : int,
        col_offset : int,
        full_cube  : np.ndarray | None = None,
    ) -> dict[str, Path]:
        """Renders and writes the GIFs walking one axis of the prediction/GT cube pair.

        The composite GIF lands at out_path. With separate_panels enabled, one
        single-cube GIF per available cube is written next to it, named
        "<stem>_<cube>.gif" for cube in pred, gt and full.

        Args:
            pred_cube  : Predicted cube of shape (n_z, rows, cols).
            gt_cube    : Ground-truth cube of the same shape.
            axis       : Walk axis, "z", "col" or "row".
            out_path   : Destination path of the composite GIF.
            z_axis     : Z axis in metres, shape (n_z,).
            row_offset : Row index of the cube's first row in the full scene.
            col_offset : Column index of the cube's first column in the full scene.
            full_cube  : Raw cube of the same shape shown as an extra panel, or None.

        Returns:
            Mapping from GIF name to its written path: out_path's stem for the composite, plus "<stem>_<cube>" per separate GIF.
        """
        plt.rcParams.update(PlotBase.SCIENTIFIC_RC)
        plt.rcParams.update(PlotBase.STYLE_RC[PlotBase.style])
        plt.rcParams["figure.dpi"]  = self.dpi
        plt.rcParams["savefig.dpi"] = self.dpi

        named    = [(name, cube) for name, cube in (("pred", pred_cube), ("gt", gt_cube), ("full", full_cube)) if cube is not None]
        names    = [name for name, _ in named]
        cubes    = tuple(cube for _, cube in named)
        full_pos = names.index("full") if "full" in names else None

        spec      = self._build_axis(axis, cubes, z_axis, row_offset, col_offset)
        n_total   = spec["n_total"]
        get_slice = spec["get_slice"]

        frame_indices = (np.linspace(0, n_total - 1, self.max_frames).round().astype(int) if n_total > self.max_frames else np.arange(n_total))

        sample_idx  = frame_indices[:: max(1, len(frame_indices) // 16)]
        samples     = [get_slice(int(i)) for i in sample_idx]
        pred_sample = np.stack([s[0] for s in samples])
        gt_sample   = np.stack([s[1] for s in samples])
        vmin, vmax  = Plotter._shared_clim(pred_sample, gt_sample)
        emax_gt     = float(np.percentile(np.abs(pred_sample - gt_sample), 99.0))

        if emax_gt <= 0.0:
            emax_gt = 1.0

        started   = time.perf_counter()
        specs     = self._frame_specs(frame_indices, get_slice, spec, full_pos, vmin, vmax, emax_gt)
        png_bytes = self._render(specs, len(frame_indices), f"GIF {out_path.name}")
        self._write_gif(png_bytes, out_path)
        self._log_written(axis, out_path, len(frame_indices), n_total, started)

        written = {out_path.stem: out_path}

        if not self.separate_panels:
            return written

        for pos, name in enumerate(names):
            started     = time.perf_counter()
            panel_path  = out_path.with_name(f"{out_path.stem}_{name}.gif")
            panel_specs = self._panel_specs(frame_indices, get_slice, spec, pos, self.PANEL_TITLES[name], vmin, vmax)
            panel_bytes = self._render(panel_specs, len(frame_indices), f"GIF {panel_path.name}", render=Animator._render_panel_frame)
            self._write_gif(panel_bytes, panel_path)
            self._log_written(axis, panel_path, len(frame_indices), n_total, started)

            written[panel_path.stem] = panel_path

        return written
