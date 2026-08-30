"""Animated walks through predicted and ground-truth tomographic cubes.

A cube pair is sliced along elevation, range or azimuth and each slice is
rendered in a worker process as a side-by-side panel of ground truth,
prediction and absolute error, then assembled into a GIF.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses        import dataclass
from io                 import BytesIO
from pathlib            import Path
from typing             import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np
from PIL  import Image
from tqdm import tqdm

from pipelines.backbone.inference.plots import PlotTools, Plotter
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
        full        : Raw tomogram slice of the same shape, or None when not shown.
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


class Animator:
    """Renders cube-walk GIFs comparing predicted and ground-truth tomograms.

    Attributes:
        logger      : Logger for the configuration banner and written-GIF lines.
        cmap        : Colormap of the GT and prediction panels.
        err_cmap    : Colormap of the absolute-error panel.
        dpi         : Rendering resolution of each frame.
        fps         : Playback rate of the GIF.
        max_frames  : Cap on the number of frames per GIF.
        num_workers : Render worker processes, or None to size from the CPU count.
    """

    def __init__(
        self,
        logger      : Logger,
        *,
        cmap        : str = "jet",
        err_cmap    : str = "magma",
        dpi         : int = 110,
        fps         : int = 12,
        max_frames  : int = 150,
        num_workers : int | None = None,
    ) -> None:
        """Stores the rendering settings and logs them.

        Args:
            logger      : Logger for the configuration banner and written-GIF lines.
            cmap        : Colormap of the GT and prediction panels.
            err_cmap    : Colormap of the absolute-error panel.
            dpi         : Rendering resolution of each frame.
            fps         : Playback rate of the GIF.
            max_frames  : Cap on the number of frames per GIF; slices are subsampled evenly when the axis is longer.
            num_workers : Render worker processes, or None to size from the CPU count.

        Raises:
            ValueError : If max_frames is below one.
        """
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}; use a large value instead of 0 to leave GIFs uncapped.")

        self.logger      = logger
        self.cmap        = cmap
        self.err_cmap    = err_cmap
        self.dpi         = dpi
        self.fps         = fps
        self.max_frames  = max_frames
        self.num_workers = num_workers

        self.logger.section("[Animator]")
        self.logger.subsection(f"colormap for GT and prediction : {cmap}")
        self.logger.subsection(f"colormap for error             : {err_cmap}")
        self.logger.subsection(f"figure DPI                     : {dpi}")
        self.logger.subsection(f"GIF frames per second          : {fps}")
        self.logger.subsection(f"max frames per GIF             : {max_frames}")
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

    def _render(self, specs: Iterator[FrameSpec], n_frames: int) -> dict[int, bytes]:
        """Renders the frame specs in a process pool with a bounded submission window.

        Args:
            specs    : Frame specifications, consumed lazily.
            n_frames : Total number of frames, for the progress bar and pool sizing.

        Returns:
            Mapping from frame order to the encoded PNG bytes.
        """
        n_workers = self.num_workers if self.num_workers is not None else min(n_frames, os.cpu_count() or 1)
        window    = 2 * n_workers
        png_bytes: dict[int, bytes] = {}

        with ProcessPoolExecutor(max_workers=n_workers, initializer=Animator._init_worker) as pool:
            pending: set = set()
            with tqdm(total=n_frames, desc="Rendering frames", unit="frame") as pbar:
                for spec in specs:
                    if len(pending) >= window:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            order, data = fut.result()
                            png_bytes[order] = data
                            pbar.update(1)
                    pending.add(pool.submit(Animator._render_frame, spec))

                for fut in as_completed(pending):
                    order, data = fut.result()
                    png_bytes[order] = data
                    pbar.update(1)

        return png_bytes

    @staticmethod
    def _render_frame(spec: FrameSpec) -> tuple[int, bytes]:
        """Renders one frame into PNG bytes inside a worker process.

        Args:
            spec : Frame specification holding the slices, limits and labels.

        Returns:
            A tuple of the frame order and the encoded PNG bytes.
        """
        g, p = spec.gt, spec.pred
        eg   = np.abs(p - g)

        panels = []
        if spec.full is not None:
            panels.append((spec.full, "Full cube (raw)", spec.cmap, spec.vmin, spec.vmax))

        panels += [
            (g,  "GT (Gaussian)", spec.cmap,     spec.vmin, spec.vmax),
            (p,  "Prediction",    spec.cmap,     spec.vmin, spec.vmax),
            (eg, "|Pred - GT|",   spec.err_cmap, 0.0,       spec.emax_gt),
        ]

        n_col   = len(panels)
        fig     = plt.figure(figsize=(6.7 * n_col, 6), constrained_layout=False)
        gs      = fig.add_gridspec(2, n_col, height_ratios=[1, 0.03], hspace=0.35, wspace=0.35)
        axes    = [fig.add_subplot(gs[0, k]) for k in range(n_col)]
        pbar_ax = fig.add_subplot(gs[1, :])

        PlotTools._triple_panel(fig, axes, panels, spec.x_label, "intensity", spec.extent, origin=spec.origin)
        axes[0].set_ylabel(spec.y_label)

        progress = (spec.frame_order + 1) / max(1, spec.n_frames)
        pbar_ax.barh(0, progress,        height=1, color="steelblue", left=0.0)
        pbar_ax.barh(0, 1.0 - progress,  height=1, color="#333333",   left=progress)
        pbar_ax.set_xlim(0, 1)
        pbar_ax.set_axis_off()

        fig.suptitle(spec.title, fontsize=13, y=0.98)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.08)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=spec.dpi)
        plt.close(fig)
        buf.seek(0)

        return spec.frame_order, buf.read()

    @staticmethod
    def _init_worker() -> None:
        """Selects the headless matplotlib backend inside a render worker."""
        matplotlib.use("Agg")

    def _frame_specs(
        self,
        frame_indices : np.ndarray,
        get_slice,
        axis_spec     : dict,
        has_full      : bool,
        vmin          : float,
        vmax          : float,
        emax_gt       : float,
    ) -> Iterator[FrameSpec]:
        """Yields one frame specification per selected slice index.

        Args:
            frame_indices : Slice indices to render along the walk axis.
            get_slice     : Callable returning the cube slices at one index.
            axis_spec     : Axis recipe holding the extent, labels, origin and titles.
            has_full      : Whether the raw tomogram cube is present as a third panel.
            vmin          : Lower intensity limit shared by the GT and prediction panels.
            vmax          : Upper intensity limit shared by the GT and prediction panels.
            emax_gt       : Upper limit of the absolute-error panel.

        Yields:
            The frame specification of each slice, in animation order.
        """

        n_frames = len(frame_indices)

        for frame_order, fi in enumerate(frame_indices):
            i    = int(fi)
            parts = get_slice(i)
            p, g = parts[0], parts[1]
            f    = parts[2] if has_full else None

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

    def walk_gif(
        self,
        pred_cube : np.ndarray,
        gt_cube   : np.ndarray,
        axis      : str,
        out_path  : Path,
        *,
        z_axis     : np.ndarray,
        row_offset : int,
        col_offset : int,
        full_cube  : np.ndarray | None = None,
    ) -> Path:
        """Renders and writes a GIF walking one axis of the prediction/GT cube pair.

        Args:
            pred_cube : Predicted cube of shape (n_elevation, azimuth, range).
            gt_cube   : Ground-truth cube of the same shape.
            axis      : Walk axis, "elevation", "range" or "azimuth".
            out_path  : Destination GIF path.
            x_axis    : Elevation axis in metres, shape (n_elevation,).
            az_offset : Azimuth index of the cube's first row in the full scene.
            rg_offset : Range index of the cube's first column in the full scene.
            full_cube : Raw tomogram cube of the same shape shown as an extra panel, or None.

        Returns:
            The written GIF path.
        """
        plt.rcParams.update(PlotBase.SCIENTIFIC_RC)
        plt.rcParams.update(PlotBase.STYLE_RC[PlotBase.style])
        plt.rcParams["figure.dpi"]  = self.dpi
        plt.rcParams["savefig.dpi"] = self.dpi

        cubes     = (pred_cube, gt_cube) if full_cube is None else (pred_cube, gt_cube, full_cube)
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

        specs     = self._frame_specs(frame_indices, get_slice, spec, full_cube is not None, vmin, vmax, emax_gt)
        png_bytes = self._render(specs, len(frame_indices))
        decoded   = (Image.open(BytesIO(png_bytes.pop(k))).convert("P", dither=Image.Dither.NONE) for k in sorted(png_bytes))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = int(round(1000.0 / max(1, self.fps)))

        next(decoded).save(
            fp            = str(out_path),
            format        = "GIF",
            save_all      = True,
            append_images = decoded,
            loop          = 0,
            duration      = duration_ms,
            optimize      = False,
        )

        self.logger.subsection(f"GIF ({axis:<9}) : {out_path}")

        return out_path
