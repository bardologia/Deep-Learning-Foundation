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


@dataclass
class FrameSpec:
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

    def _build_axis(self, axis: str, cubes: tuple, x_axis: np.ndarray, az_offset: int, rg_offset: int) -> dict:
        N_elev, az, rg = cubes[0].shape
        sort_idx = None

        if axis in ("range", "azimuth"):
            sort_idx = np.argsort(x_axis)
            x_axis   = x_axis[sort_idx]

        if axis == "elevation":
            return dict(
                n_total   = N_elev,
                get_slice = lambda i: self._slice_elevation(cubes, i),
                extent    = [rg_offset, rg_offset + rg, az_offset + az, az_offset],
                x_label   = "range index",
                y_label   = "azimuth index",
                title_fn  = lambda i: f"elevation = {x_axis[i]:.2f} m  (idx {i}/{N_elev - 1})",
                origin    = "upper",
            )

        if axis == "range":
            return dict(
                n_total   = rg,
                get_slice = lambda i: self._slice_range(cubes, sort_idx, i),
                extent    = [az_offset, az_offset + az, float(x_axis[0]), float(x_axis[-1])],
                x_label   = "azimuth index",
                y_label   = "elevation [m]",
                title_fn  = lambda i: f"range = {i + rg_offset}",
                origin    = "lower",
            )

        if axis == "azimuth":
            return dict(
                n_total   = az,
                get_slice = lambda i: self._slice_azimuth(cubes, sort_idx, i),
                extent    = [rg_offset, rg_offset + rg, float(x_axis[0]), float(x_axis[-1])],
                x_label   = "range index",
                y_label   = "elevation [m]",
                title_fn  = lambda i: f"azimuth = {i + az_offset}",
                origin    = "lower",
            )

        raise ValueError(f"axis must be elevation|range|azimuth, got {axis!r}")

    @staticmethod
    def _slice_elevation(cubes: tuple, i: int) -> tuple:
        return tuple(cube[i] if cube is not None else None for cube in cubes)

    @staticmethod
    def _slice_range(cubes: tuple, sort_idx: np.ndarray | None, i: int) -> tuple:
        slices = tuple(cube[:, :, i] if cube is not None else None for cube in cubes)
        if sort_idx is not None:
            slices = tuple(s[sort_idx] if s is not None else None for s in slices)
        return slices

    @staticmethod
    def _slice_azimuth(cubes: tuple, sort_idx: np.ndarray | None, i: int) -> tuple:
        slices = tuple(cube[:, i, :] if cube is not None else None for cube in cubes)
        if sort_idx is not None:
            slices = tuple(s[sort_idx] if s is not None else None for s in slices)
        return slices

    def _render(self, specs: Iterator[FrameSpec], n_frames: int) -> dict[int, bytes]:
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
        g, p = spec.gt, spec.pred
        eg   = np.abs(p - g)

        panels = []
        if spec.full is not None:
            panels.append((spec.full, "Full tomogram (raw)", spec.cmap, spec.vmin, spec.vmax))

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

        n_frames = len(frame_indices)

        for frame_order, fi in enumerate(frame_indices):
            i    = int(fi)
            slc  = get_slice(i)
            p, g = slc[0], slc[1]
            f    = slc[2] if has_full else None

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
        x_axis    : np.ndarray,
        az_offset : int,
        rg_offset : int,
        full_cube : np.ndarray | None = None,
    ) -> Path:
        plt.rcParams.update(Plotter.SCIENTIFIC_RC)
        plt.rcParams.update(Plotter.REPORT_RC)
        plt.rcParams["figure.dpi"]  = self.dpi
        plt.rcParams["savefig.dpi"] = self.dpi

        cubes     = (pred_cube, gt_cube) if full_cube is None else (pred_cube, gt_cube, full_cube)
        spec      = self._build_axis(axis, cubes, x_axis, az_offset, rg_offset)
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
