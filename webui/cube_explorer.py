"""Interactive server-side backend for the web UI cube explorer tabs.

Opens the cubes written by inference (prediction, ground truth and the full
reference cube of the dataset run) and renders slices, planes, parameter maps,
metric overlays and point clouds as PNG or packed binary payloads. Cubes are
indexed by z bin, row pixel and column pixel. A load reads every file whole
through a shared pool of concurrent chunk reads, books its progress by bytes,
times every stage and keeps the last few loaded cubes in memory. The slice
collector renders the same cuts across many runs into one folder.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
import traceback
from collections        import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime           import datetime
from pathlib            import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

from catalog_roots            import CatalogRoots, RunScanner
from lru_cache                import LruCache
from tools.data.gaussians     import GaussianActivity
from tools.reporting.plotting import PlotBase
from web_logger               import WebLogger


class SliceFigureArchiver(PlotBase):
    """Renders publication-styled slice and transect figures to disk.

    Wraps the shared plotting base so every saved cut carries axis labels, a z
    axis and a colourbar, using the paper style regardless of the style active
    elsewhere in the process.

    Attributes:
        LABELS : Figure title of every cube source.
    """

    LABELS = {
        "pred"    : "Prediction",
        "predb"   : "Prediction B",
        "diff"    : "Prediction A − B",
        "gt"      : "GT (Gaussian)",
        "full"    : "Reference (raw)",
    }

    def render(self, data: np.ndarray, z_values: np.ndarray, vmin: float, vmax: float, source: str, axis: str, row: int, col: int, space: str, path: Path, cmap: str = "jet", label: str | None = None) -> Path:
        x_label   = "row index" if axis == "col" else "col index"
        title_pos = f"col = {col}" if axis == "col" else f"row = {row}"
        y_label   = "z bin" if source == "full" else "z"
        cbar      = "intensity (per-column normalised)" if space == "normalized" else "intensity"
        extent    = [0, int(data.shape[1]), float(z_values[0]), float(z_values[-1])]
        title     = f"{self.LABELS[source]} — {title_pos}" if label is None else f"{label} — {self.LABELS[source]} — {title_pos}"

        previous = PlotBase.style
        PlotBase.use_style("paper")
        try:
            return self._imshow_figure(
                data,
                x_label        = x_label,
                y_label        = y_label,
                title          = title,
                cmap           = cmap,
                vmin           = vmin,
                vmax           = vmax,
                extent         = extent,
                origin         = "lower",
                colorbar_label = cbar,
                figsize        = self.figsize(self.FULL_WIDTH),
                path           = path,
            )
        finally:
            PlotBase.use_style(previous)

    def render_transect(self, data: np.ndarray, z_values: np.ndarray, vmin: float, vmax: float, source: str, start: tuple, end: tuple, space: str, path: Path, cmap: str = "jet") -> Path:
        cbar   = "intensity (per-column normalised)" if space == "normalized" else "intensity"
        extent = [0, int(data.shape[1]), float(z_values[0]), float(z_values[-1])]

        previous = PlotBase.style
        PlotBase.use_style("paper")
        try:
            return self._imshow_figure(
                data,
                x_label        = "sample along transect",
                y_label        = "z bin" if source == "full" else "z",
                title          = f"{self.LABELS[source]} — transect row{start[0]},col{start[1]} to row{end[0]},col{end[1]}",
                cmap           = cmap,
                vmin           = vmin,
                vmax           = vmax,
                extent         = extent,
                origin         = "lower",
                colorbar_label = cbar,
                figsize        = self.figsize(self.FULL_WIDTH),
                path           = path,
            )
        finally:
            PlotBase.use_style(previous)


class LoadProgress:
    """Thread-safe byte counter a cube load reports its file reads through.

    Besides the overall fraction it tracks every file currently in flight with
    its bytes landed so far, so a stalled load names the file it is waiting on.

    Attributes:
        lock     : Guards the counters and the in-flight table.
        total    : Bytes the load plans to read.
        done     : Bytes read so far.
        files    : Files in flight, keyed by path, each as [bytes done, bytes total].
        listener : Callable receiving the fraction done and the in-flight snapshot after every advance, or None.
    """

    def __init__(self, listener=None) -> None:
        """Starts an empty counter, optionally forwarding progress to a listener.

        Args:
            listener : Callable taking the fraction done and the in-flight snapshot, or None for a silent counter.
        """
        self.lock     = threading.Lock()
        self.total    = 0
        self.done     = 0
        self.files    = {}
        self.listener = listener

    def plan(self, n_bytes: int) -> None:
        """Adds bytes to the planned total.

        Args:
            n_bytes : Bytes a later stage of the load will read.
        """
        with self.lock:
            self.total += int(n_bytes)

    def begin(self, path: Path, n_bytes: int) -> None:
        """Registers a file whose chunks are about to land.

        Args:
            path    : File being read.
            n_bytes : Bytes that will be read from it.
        """
        with self.lock:
            self.files[str(path)] = [0, int(n_bytes)]

    def advance(self, path: Path, n_bytes: int) -> None:
        """Books bytes of one file as read and publishes the fraction done.

        Args:
            path    : File the bytes came from, registered through ``begin``.
            n_bytes : Bytes that just landed.
        """
        with self.lock:
            key        = str(path)
            self.done += int(n_bytes)
            self.files[key][0] += int(n_bytes)
            if self.files[key][0] >= self.files[key][1]:
                del self.files[key]

            fraction = min(1.0, self.done / self.total) if self.total else 0.0
            reading  = self._snapshot()

        if self.listener is not None:
            self.listener(fraction, reading)

    def _snapshot(self) -> list:
        """Lists the files in flight, most bytes outstanding first, with the caller holding the lock."""
        rows = [{"file": "/".join(Path(key).parts[-5:]), "done": done, "total": total} for key, (done, total) in self.files.items()]
        rows.sort(key=lambda row: row["done"] - row["total"])
        return rows


class NpyReader:
    """Reads ``.npy`` files whole through a shared pool of concurrent chunk reads.

    Every file is split into fixed-size chunks that are pulled with ``pread``
    by a process-wide thread pool, so one large file and many small ones alike
    keep several requests in flight against the disk or mount, and every chunk
    books its bytes on the load's progress counter as it lands. Pool tasks are
    leaf reads only, so callers may wait on them from any thread without
    deadlock.

    Attributes:
        progress     : Counter receiving the byte count of every chunk read.
        CHUNK_BYTES  : Size of one pread request.
        READ_WORKERS : Threads of the shared read pool.
        POOL         : Process-wide pool the chunks are read on.
    """

    CHUNK_BYTES  = 16 * 1024 * 1024
    READ_WORKERS = min(32, os.cpu_count() or 4)
    POOL         = ThreadPoolExecutor(max_workers=READ_WORKERS, thread_name_prefix="npy-read")

    def __init__(self, progress: LoadProgress) -> None:
        """Stores the progress counter the reads report to.

        Args:
            progress : Counter receiving the byte count of every chunk read.
        """
        self.progress = progress

    def header(self, path: Path) -> tuple[tuple, np.dtype]:
        """Returns the shape and dtype recorded in a ``.npy`` header without reading the data.

        Args:
            path : File written by ``np.save``.
        """
        with open(path, "rb") as handle:
            shape, _, dtype = self._parse_header(handle, path)

        return shape, dtype

    def read(self, path: Path) -> np.ndarray:
        """Reads one whole ``.npy`` file into a fresh C-ordered array.

        Args:
            path : File written by ``np.save``.
        """
        return self.read_many([path])[0]

    def read_many(self, paths: list) -> list:
        """Reads several ``.npy`` files at once, all their chunks in flight together.

        Args:
            paths : Files written by ``np.save``.

        Returns:
            One array per path, in order, with the shape and dtype each header declares.

        Raises:
            ValueError : If a header version is unsupported, an array is Fortran-ordered or a file ends before its data does.
        """
        handles = [open(path, "rb") for path in paths]
        try:
            targets = []
            chunks  = []
            for handle, path in zip(handles, paths):
                shape, dtype, offset = self._open_header(handle, path)
                target               = self._target(path, handle, offset, shape, dtype)
                targets.append(target)
                chunks.extend((target, start, start, min(self.CHUNK_BYTES, target["buffer"].nbytes - start)) for start in range(0, target["buffer"].nbytes, self.CHUNK_BYTES))

            list(self.POOL.map(self._pull, chunks))
        finally:
            for handle in handles:
                handle.close()

        return [target["array"] for target in targets]

    def read_band(self, path: Path, lo: int, hi: int) -> np.ndarray:
        """Reads rows ``lo:hi`` along the second axis of a 3D ``.npy`` cube.

        Each z plane stores its rows contiguously, so the band is one contiguous
        read per plane and the bytes outside it are never touched.

        Args:
            path : File written by ``np.save`` holding a C-ordered cube of shape (planes, rows, columns).
            lo   : First row of the band.
            hi   : One past the last row of the band.

        Returns:
            Array of shape (planes, hi - lo, columns).

        Raises:
            ValueError : If the file is not a 3D cube, the band falls outside its rows, or the file ends before its data does.
        """
        with open(path, "rb") as handle:
            shape, dtype, offset = self._open_header(handle, path)
            if len(shape) != 3:
                raise ValueError(f"{path} is not a 3D cube: shape={shape}")
            if lo < 0 or hi > shape[1] or lo >= hi:
                raise ValueError(f"rows [{lo}:{hi}] fall outside the {shape[1]} rows of {path}")

            target     = self._target(path, handle, offset, (shape[0], hi - lo, shape[2]), dtype)
            plane_size = shape[1] * shape[2] * dtype.itemsize
            band_size  = (hi - lo) * shape[2] * dtype.itemsize
            row_start  = lo * shape[2] * dtype.itemsize
            chunks     = [(target, plane * plane_size + row_start, plane * band_size, band_size) for plane in range(shape[0])]
            list(self.POOL.map(self._pull, chunks))

        return target["array"]

    def _open_header(self, handle, path: Path) -> tuple[tuple, np.dtype, int]:
        """Parses an open file's header and returns its shape, dtype and data offset, refusing Fortran order."""
        shape, fortran, dtype = self._parse_header(handle, path)
        if fortran:
            raise ValueError(f"{path} is Fortran-ordered; cube files are written C-ordered")

        return shape, dtype, handle.tell()

    def _target(self, path: Path, handle, offset: int, shape: tuple, dtype: np.dtype) -> dict:
        """Allocates the array a file's chunks land in and registers it with the progress counter."""
        array  = np.empty(shape, dtype=dtype)
        buffer = memoryview(array.reshape(-1).view(np.uint8))
        self.progress.begin(path, buffer.nbytes)
        return {"path": path, "fd": handle.fileno(), "offset": offset, "array": array, "buffer": buffer}

    def _pull(self, chunk: tuple) -> None:
        """Reads one chunk with pread into its place in the target buffer and books its bytes.

        Args:
            chunk : Tuple of (target, byte offset into the file's data, byte offset into the buffer, byte count).

        Raises:
            ValueError : If the file ends before the chunk does.
        """
        target, src, dst, count = chunk
        done                    = 0
        while done < count:
            got = os.preadv(target["fd"], [target["buffer"][dst + done:dst + count]], target["offset"] + src + done)
            if not got:
                raise ValueError(f"{target['path']} ends after {src + done} data bytes")

            done += got
            self.progress.advance(target["path"], got)

    @staticmethod
    def _parse_header(handle, path: Path) -> tuple[tuple, bool, np.dtype]:
        """Consumes the magic and header of an open ``.npy`` file and returns (shape, fortran, dtype).

        Raises:
            ValueError : If the file uses a format version other than 1.0 or 2.0.
        """
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(handle)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(handle)

        raise ValueError(f"{path} uses .npy format version {version}; only 1.0 and 2.0 are read")


class StampReader:
    """Reads the cubes, metrics and metric maps of one inference stamp directory.

    A stamp directory lives at ``<run>/inference/<stamp>`` and holds a ``cubes``
    folder plus ``metrics.json``.

    Attributes:
        stamp_dir       : Directory of the completed inference stamp.
        npy             : Chunked file reader booking every whole-file read on the load's progress counter.
        METRIC_EXCLUDED : Name fragments of the cube files that are not metric maps.
        CURVE_SOURCES   : Curve cube keys a full load reads from the stamp.
        PARAM_SOURCES   : Parameter block keys a full load reads from the stamp.
    """

    METRIC_EXCLUDED = ("_curves", "params_")
    CURVE_SOURCES   = ("pred", "gt")
    PARAM_SOURCES   = ("pred", "gt")

    def __init__(self, stamp_dir: Path, progress: LoadProgress) -> None:
        """Stores the inference stamp directory and the load's progress counter.

        Args:
            stamp_dir : Directory of the completed inference stamp.
            progress  : Counter the whole-file reads report to.
        """
        self.stamp_dir = stamp_dir
        self.npy       = NpyReader(progress)

    @property
    def run_dir(self) -> Path:
        """Run directory that owns this inference stamp."""
        return self.stamp_dir.parent.parent

    def metrics(self) -> dict:
        """Returns the parsed ``metrics.json`` payload of the stamp.

        Raises:
            FileNotFoundError : If the stamp holds no ``metrics.json``.
        """
        path = self.stamp_dir / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"metrics.json missing in {self.stamp_dir}; rerun inference to regenerate it")

        return json.loads(path.read_text(encoding="utf-8"))

    def cache_key(self) -> str:
        """Returns the key under which a load of this stamp is kept in memory."""
        return str(self.stamp_dir)

    def byte_plan(self) -> int:
        """Returns the bytes a full load of this stamp reads: curve cubes, parameter blocks and metric maps."""
        planned = 0
        for path in (self.stamp_dir / "cubes").glob("*.npy"):
            stem     = path.stem
            is_curve = stem.endswith("_curves")
            is_param = stem.startswith("params_")
            if is_curve and stem[:-len("_curves")] not in self.CURVE_SOURCES:
                continue
            if is_param and stem[len("params_"):] not in self.PARAM_SOURCES:
                continue

            planned += path.stat().st_size

        return planned

    def curves(self, name: str, view: bool = False) -> np.ndarray | None:
        """Reads a curve cube of shape (z bins, rows, columns).

        The whole file is read into memory in one pass of concurrent chunk
        reads, which is the fastest access pattern on the disks and network
        mounts the runs live on. Callers that only touch a few lines of the
        cube ask for a memory-mapped view.

        Args:
            name : Cube key such as "pred" or "gt".
            view : When True the file is memory-mapped instead of read into RAM.

        Returns:
            The cube, or None when the stamp carries no such cube.
        """
        path = self.stamp_dir / "cubes" / f"{name}_curves.npy"
        if not path.is_file():
            return None

        return np.load(path, mmap_mode="r") if view else self.npy.read(path)

    def param_block(self, name: str) -> np.ndarray | None:
        """Loads a Gaussian parameter cube of shape (3 * n_slots, rows, columns).

        Channels run amplitude, mu and sigma per slot.

        Args:
            name : Parameter source, "pred" or "gt".

        Returns:
            The float32 parameter cube, or None when the stamp stores none.
        """
        path = self.stamp_dir / "cubes" / f"params_{name}.npy"
        if not path.is_file():
            return None

        return np.asarray(self.npy.read(path), dtype=np.float32)

    def metric_maps(self, n_rows: int, n_cols: int) -> dict:
        """Loads every per-pixel metric map matching the cube footprint.

        Args:
            n_rows : Row extent the map must have, in pixels.
            n_cols : Column extent the map must have, in pixels.

        Returns:
            Mapping from map name to a float32 array of shape (n_rows, n_cols);
            curve and parameter cubes are skipped, as are arrays of other shapes.
        """
        kept = []
        for path in sorted((self.stamp_dir / "cubes").glob("*.npy")):
            if any(marker in path.name for marker in self.METRIC_EXCLUDED):
                continue

            shape, _ = self.npy.header(path)
            if shape == (n_rows, n_cols):
                kept.append(path)

        arrays = self.npy.read_many(kept)
        return {path.stem: np.asarray(raw, dtype=np.float32) for path, raw in zip(kept, arrays)}


class LoadTimer:
    """Books the wall time of the named stages of one cube load.

    Attributes:
        stages  : Seconds spent per stage, in completion order.
        started : Clock reading when the load began.
        last    : Clock reading at the end of the previous stage.
    """

    def __init__(self) -> None:
        """Starts the clock."""
        self.stages  = {}
        self.started = time.perf_counter()
        self.last    = self.started

    def lap(self, stage: str) -> None:
        """Books the time elapsed since the previous lap under ``stage``.

        Args:
            stage : Name of the stage that just ended; repeated names accumulate.
        """
        now                = time.perf_counter()
        self.stages[stage] = round(self.stages.get(stage, 0.0) + now - self.last, 3)
        self.last          = now

    def total(self) -> float:
        """Seconds elapsed since the clock started."""
        return time.perf_counter() - self.started

    def summary(self) -> str:
        """Renders the stage times as comma-separated ``stage 1.23 s`` items."""
        return ", ".join(f"{stage} {seconds:.2f} s" for stage, seconds in self.stages.items())


class CubeExplorer:
    """Loads one cube at a time and serves views of it to the web UI.

    Serves one loaded cube at a time behind a lock: its per-source curve cubes,
    the primary amplitude map in dB, Gaussian parameter blocks, metric maps and
    the cropped aux map. Loading runs in a background thread whose progress is
    polled through ``load_status``; the last few loaded cubes stay in memory so
    switching back to one is immediate.

    Attributes:
        logger      : Web logger for load and save reporting.
        archiver    : Figure writer used when saving slices and transects.
        roots       : Catalogue of runs roots the user has opened.
        scanner     : Scanner listing inference stamps under a root.
        lock        : Guards the loaded cube and the load status.
        loaded      : Currently loaded cube state, or None.
        load_thread : Background thread of the running load, or None.
        status_at   : Clock reading of the last change to the load status.
        cube_cache  : Recently loaded cube states keyed by reader cache key.
        status      : Load state machine payload polled by the front end.
    """

    SOURCES             = ("pred", "predb", "diff", "gt", "full")
    PARAM_SOURCES       = ("pred", "gt")
    CLOUD_CURVE_SOURCES = ("full",)
    PARAM_FIELDS        = {"amp": 0, "mu": 1, "sigma": 2}
    PARAM_BAD           = "#10151a"
    CUBE_CACHE_ENTRIES  = 8
    LIMIT_SAMPLE        = 1_000_000
    STALL_FRAMES        = 4
    PNG_ENCODE          = {"format": "png", "pil_kwargs": {"compress_level": 1}}

    CMAPS = ("jet", "viridis", "inferno", "turbo", "gray")

    METRIC_LABELS = {
        "pixel_mse"                : "MSE",
        "pixel_mae"                : "MAE",
        "pixel_r2"                 : "R2",
        "pixel_cos"                : "cosine",
        "pixel_peak"               : "peak shift",
        "seed_std_profile"         : "seed std profile",
        "seed_std_amp"             : "seed std amp",
        "seed_std_mu"              : "seed std mu",
        "seed_std_sigma"           : "seed std sigma",
        "label_r2"                 : "label R2",
        "flip_consistency"         : "flip disagreement",
        "failure_mode"             : "failure mode",
        "label_suspect"            : "label suspect",
    }

    def __init__(self, logger: WebLogger) -> None:
        """Prepares an idle explorer with an empty runs-root catalogue and an empty cube cache.

        Args:
            logger : Web logger for load and save reporting.
        """
        self.logger      = logger
        self.archiver    = SliceFigureArchiver()
        self.roots       = CatalogRoots()
        self.scanner     = RunScanner(self.roots)
        self.lock        = threading.Lock()
        self.loaded      = None
        self.load_thread = None
        self.status_at   = time.perf_counter()
        self.cube_cache  = LruCache(self.CUBE_CACHE_ENTRIES)
        self.status      = {"state": "idle", "id": None, "progress": 0.0, "stage": "", "error": ""}

    def list_cubes(self, base: str) -> dict:
        """Lists the inference stamps carrying a prediction cube under a runs root.

        Args:
            base : Runs root to scan.

        Returns:
            Dict with ``ok``, the resolved ``root`` and the ``cubes`` entries, or ``ok`` False with the reason.
        """
        scanned = self.scanner.stamps(base)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "cubes": []}

        return {"ok": True, "root": scanned["root"], "cubes": scanned["entries"]}

    def start_load(self, cube_id: str) -> dict:
        """Starts loading a cube in a background thread, or serves it from memory.

        A cube loaded recently is swapped in at once, with any attached
        comparison dropped; otherwise a background thread reads it and stores
        the result for later switches.

        Args:
            cube_id : Stamp directory path.

        Returns:
            Dict with ``ok`` True once the load was started or the cube is
            already loaded, otherwise ``ok`` False with the reason.
        """
        reader = self._open_source(cube_id, LoadProgress(self._publish_progress))
        if reader is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}

        cached = self.cube_cache.get(reader.cache_key())

        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": f"a load is already running for {self.status['id']}"}
            if self.status["state"] == "ready" and self.status["id"] == cube_id and self.loaded is not None:
                return {"ok": True}

            if cached is not None:
                cached["meta"] = {**cached["meta"], "timings": {"total": 0.0, "stages": {"memory": 0.0}}}
                self.loaded    = cached
                self.status    = {"state": "ready", "id": cube_id, "progress": 1.0, "stage": "ready", "error": ""}
            else:
                self.loaded = None
                self.status = {"state": "loading", "id": cube_id, "progress": 0.0, "stage": "scanning sources", "reading": [], "error": ""}

        if cached is not None:
            self.detach_second(cube_id)
            self.logger.info(f"cube served from memory: {cube_id}")
            return {"ok": True}

        self.status_at   = time.perf_counter()
        self.load_thread = threading.Thread(target=self._load_worker, args=(cube_id, reader), daemon=True)
        self.load_thread.start()
        return {"ok": True}

    def _publish_progress(self, fraction: float, reading: list) -> None:
        """Writes the fraction of planned bytes read and the files in flight into the polled load status."""
        with self.lock:
            if self.status["state"] == "loading":
                self.status["progress"] = fraction
                self.status["reading"]  = reading
                self.status_at          = time.perf_counter()

    def load_status(self) -> dict:
        """Returns the current load state, with the cube metadata once ready.

        While a load runs the payload also carries ``stalled_for``, the seconds
        since the status last changed, and ``where``, the innermost frames of
        the load thread, so a load that stops moving names the code it sits in.
        """
        with self.lock:
            payload = dict(self.status)
            if payload["state"] == "ready" and self.loaded is not None:
                payload["cube"] = self.loaded["meta"]
            if payload["state"] == "loading":
                payload["stalled_for"] = round(time.perf_counter() - self.status_at, 1)
                payload["where"]       = self._load_thread_frames()

        return payload

    def _load_thread_frames(self) -> list:
        """Returns the innermost STALL_FRAMES frames of the load thread as ``file:line function`` strings."""
        thread = self.load_thread
        frame  = sys._current_frames().get(thread.ident) if thread is not None else None
        if frame is None:
            return []

        return [f"{Path(entry.filename).name}:{entry.lineno} {entry.name}" for entry in traceback.extract_stack(frame)[-self.STALL_FRAMES:]]

    def primary_png(self, cube_id: str) -> bytes | None:
        """Renders the primary amplitude map of the loaded cube in dB as a greyscale PNG.

        Args:
            cube_id : Id of the loaded cube.

        Returns:
            PNG bytes, or None when another cube is loaded.
        """
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            primary = self.loaded["primary"]
            clim    = self.loaded["primary_clim"]

        return self._encode_primary(primary, clim)

    def attach_second(self, cube_id: str, other_id: str) -> dict:
        """Attaches a second run's prediction cube for side-by-side comparison.

        Args:
            cube_id  : Id of the loaded cube.
            other_id : Stamp directory of the run to compare against.

        Returns:
            Dict with ``ok`` and the updated ``cube`` metadata carrying the
            ``predb`` and ``diff`` sources, or ``ok`` False with the reason.
        """
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            pred = self.loaded["entries"]["pred"]

        other = self._open_source(other_id)
        if other is None:
            return {"ok": False, "error": f"unknown cube id: {other_id}"}
        if other_id == cube_id:
            return {"ok": False, "error": "pick a different inference result to compare against"}

        raw = other.curves("pred")
        if raw.shape != pred["cube"].shape:
            return {"ok": False, "error": f"comparison cube shape {tuple(raw.shape)} does not match {tuple(pred['cube'].shape)}"}

        other_axis = self._curve_axis(other, raw.shape[0])
        if not np.allclose(other_axis, pred["z_axis"]):
            return {"ok": False, "error": "comparison cube covers a different z axis"}

        predb = self._ingest(raw, pred["z_axis"])
        diff  = self._diff_entry(pred, predb)

        run_dir   = other.run_dir
        other_dir = other.stamp_dir
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}

            self.loaded["entries"]["predb"] = predb
            self.loaded["entries"]["diff"]  = diff

            meta = dict(self.loaded["meta"])
            meta["sources"]   = [s for s in self.SOURCES if s in self.loaded["entries"]]
            meta["n_z"]    = {s: int(self.loaded["entries"][s]["cube"].shape[0]) for s in self.loaded["entries"]}
            meta["intensity"] = {s: [e["vmin"], e["vmax"]] for s, e in self.loaded["entries"].items()}
            meta["attached"]  = {"id": other_id, "run": run_dir.name, "stamp": other_dir.name}
            self.loaded["meta"] = meta

        self.logger.ok(f"attached comparison cube: {other_id}")
        return {"ok": True, "cube": meta}

    def detach_second(self, cube_id: str) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}

            self.loaded["entries"].pop("predb", None)
            self.loaded["entries"].pop("diff", None)

            meta = dict(self.loaded["meta"])
            meta["sources"]   = [s for s in self.SOURCES if s in self.loaded["entries"]]
            meta["n_z"]    = {s: int(self.loaded["entries"][s]["cube"].shape[0]) for s in self.loaded["entries"]}
            meta["intensity"] = {s: [e["vmin"], e["vmax"]] for s, e in self.loaded["entries"].items()}
            meta["attached"]  = None
            self.loaded["meta"] = meta

        return {"ok": True, "cube": meta}

    @staticmethod
    def _diff_entry(pred: dict, predb: dict) -> dict:
        cube   = pred["cube"] - predb["cube"]
        sample = np.abs(CubeExplorer._limit_sample(cube))
        peak   = float(np.percentile(sample, 99.0)) if sample.size else 1.0
        peak   = peak if peak > 0.0 else 1.0

        return {
            "cube"      : cube,
            "z_axis"    : pred["z_axis"],
            "vmin"      : -peak,
            "vmax"      : peak,
            "diverging" : True,
        }

    def profiles(self, cube_id: str, row: int, col: int) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        row = int(np.clip(row, 0, meta["n_rows"] - 1))
        col = int(np.clip(col, 0, meta["n_cols"] - 1))

        sources = {}
        for source, entry in entries.items():
            order   = np.argsort(entry["z_axis"])
            z_values = np.asarray(entry["z_axis"])[order]
            values  = entry["cube"][:, row, col][order]
            sources[source] = {"z_values": z_values.tolist(), "values": values.astype(float).tolist()}

        return {"ok": True, "row": row, "col": col, "sources": sources}

    def slice_ssim(self, cube_id: str, row: int, col: int, space: str = "physical") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        gt = entries.get("gt")
        if gt is None:
            return {"ok": True, "col": {}, "row": {}}

        row = int(np.clip(row, 0, meta["n_rows"] - 1))
        col = int(np.clip(col, 0, meta["n_cols"] - 1))

        gt_cube = gt["cube"]
        out     = {"col": {}, "row": {}}

        for source in ("pred", "predb", "full"):
            entry = entries.get(source)
            if entry is None or entry["cube"].shape != gt_cube.shape:
                continue

            cube = entry["cube"]
            out["col"][source]   = self._ssim_score(cube[:, :, col], gt_cube[:, :, col], space)
            out["row"][source] = self._ssim_score(cube[:, row, :], gt_cube[:, row, :], space)

        return {"ok": True, "col": out["col"], "row": out["row"]}

    @staticmethod
    def _ssim_score(cur: np.ndarray, ref: np.ndarray, space: str) -> float | None:
        cur = np.nan_to_num(np.asarray(cur, dtype=np.float64))
        ref = np.nan_to_num(np.asarray(ref, dtype=np.float64))

        if space == "normalized":
            cur = cur / np.where((p := cur.max(axis=0, keepdims=True)) > 1e-12, p, 1.0)
            ref = ref / np.where((p := ref.max(axis=0, keepdims=True)) > 1e-12, p, 1.0)

        data_range = float(ref.max() - ref.min())
        if data_range <= 0.0:
            return None

        min_side = min(cur.shape)
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            return None

        value = float(ssim(ref, cur, data_range=data_range, win_size=win_size))
        return value if np.isfinite(value) else None

    def slice_png(self, cube_id: str, source: str, axis: str, row: int, col: int, space: str = "physical", cmap: str = "jet") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None or axis not in ("col", "row"):
            return None

        n_z, n_rows, n_cols = entry["cube"].shape
        row = int(np.clip(row, 0, n_rows - 1))
        col = int(np.clip(col, 0, n_cols - 1))

        data, z_values, vmin, vmax = self._cut(entry, axis, row, col, space)

        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(data), cmap=self._entry_cmap(entry, cmap), vmin=vmin, vmax=vmax, **self.PNG_ENCODE)
        return buf.getvalue()

    def plane_png(self, cube_id: str, source: str, frac: float, space: str = "physical", cmap: str = "jet") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None:
            return None

        cube  = entry["cube"]
        n_z   = cube.shape[0]
        order = np.argsort(entry["z_axis"])
        pos   = int(round(float(np.clip(frac, 0.0, 1.0)) * (n_z - 1)))
        z     = int(order[pos])

        data = np.asarray(cube[z], dtype=np.float32)

        if space == "normalized":
            if entry.get("diverging"):
                peak = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.0
                data = data / (peak if peak > 1e-12 else 1.0)
                vmin, vmax = -1.0, 1.0
            else:
                peak = float(np.nanmax(data)) if np.isfinite(data).any() else 0.0
                data = data / (peak if peak > 1e-12 else 1.0)
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = entry["vmin"], entry["vmax"]

        buf = io.BytesIO()
        plt.imsave(buf, np.nan_to_num(data, nan=vmin), cmap=self._entry_cmap(entry, cmap), vmin=vmin, vmax=vmax, **self.PNG_ENCODE)
        return buf.getvalue()

    def param_map_png(self, cube_id: str, source: str, field: str, slot: int) -> bytes | None:
        resolved = self._param_state(cube_id)
        if resolved is None or field not in (*self.PARAM_FIELDS, "count"):
            return None

        params, meta = resolved

        if source in self.PARAM_SOURCES:
            block = params.get(source)
            if block is None:
                return None
            data, vmin, vmax, cmap = self._param_source_map(block, meta, field, slot)
        elif source == "error":
            if not meta["error"]:
                return None
            data, vmin, vmax, cmap = self._param_error_map(params, meta, field, slot)
        else:
            return None

        palette = plt.get_cmap(cmap).copy()
        palette.set_bad(color=self.PARAM_BAD)

        buf = io.BytesIO()
        plt.imsave(buf, data, cmap=palette, vmin=vmin, vmax=vmax, **self.PNG_ENCODE)
        return buf.getvalue()

    def param_cbar_png(self, cube_id: str, source: str, field: str) -> bytes | None:
        resolved = self._param_state(cube_id)
        if resolved is None or field not in (*self.PARAM_FIELDS, "count"):
            return None

        _, meta = resolved
        cmap    = self._param_cmap(source, field, meta)
        if cmap is None:
            return None

        return self.cbar_png(cmap)

    def cbar_png(self, cmap: str) -> bytes | None:
        if cmap not in ("viridis", "inferno", "coolwarm", "jet", "gray"):
            return None

        ramp = np.tile(np.linspace(0.0, 1.0, 256), (12, 1))

        buf = io.BytesIO()
        plt.imsave(buf, ramp, cmap=cmap, vmin=0.0, vmax=1.0, **self.PNG_ENCODE)
        return buf.getvalue()

    def params_at(self, cube_id: str, row: int, col: int) -> dict:
        resolved = self._param_state(cube_id)
        if resolved is None:
            return {"ok": False, "error": "no parameter cubes are loaded"}

        params, meta = resolved
        n_slots      = meta["n_slots"]
        threshold    = meta["threshold"]

        row = int(np.clip(row, 0, next(iter(params.values())).shape[1] - 1))
        col = int(np.clip(col, 0, next(iter(params.values())).shape[2] - 1))

        sources = {}
        for source, block in params.items():
            slots = []
            for k in range(n_slots):
                amp   = float(block[3 * k,     row, col])
                mu    = float(block[3 * k + 1, row, col])
                sigma = float(block[3 * k + 2, row, col])
                slots.append({"amp": amp, "mu": mu, "sigma": sigma, "active": bool(GaussianActivity.is_active(amp, threshold))})
            sources[source] = slots

        return {"ok": True, "row": row, "col": col, "n_slots": n_slots, "threshold": threshold, "sources": sources}

    def _param_state(self, cube_id: str) -> tuple[dict, dict] | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            params = self.loaded["params"]
            meta   = self.loaded["meta"]["params"]

        if meta is None or not params:
            return None
        return params, meta

    def _param_source_map(self, block: np.ndarray, meta: dict, field: str, slot: int) -> tuple[np.ndarray, float, float, str]:
        threshold = meta["threshold"]
        n_slots   = meta["n_slots"]

        if field == "count":
            amps = block[0::3]
            data = GaussianActivity.is_active(amps, threshold).sum(axis=0).astype(np.float32)
            vmin, vmax = meta["ranges"]["count"]
            return data, vmin, vmax, "viridis"

        slot    = int(np.clip(slot, 0, n_slots - 1))
        channel = np.asarray(block[3 * slot + self.PARAM_FIELDS[field]], dtype=np.float32)

        if field in ("mu", "sigma"):
            active  = GaussianActivity.is_active(block[3 * slot], threshold)
            channel = np.where(active, channel, np.nan)

        vmin, vmax = meta["ranges"][field]
        return channel, vmin, vmax, "viridis"

    def _param_error_map(self, params: dict, meta: dict, field: str, slot: int) -> tuple[np.ndarray, float, float, str]:
        threshold = meta["threshold"]
        n_slots   = meta["n_slots"]
        pred      = params["pred"]
        gt        = params["gt"]

        if field == "count":
            count_pred = GaussianActivity.is_active(pred[0::3], threshold).sum(axis=0).astype(np.float32)
            count_gt   = GaussianActivity.is_active(gt[0::3],   threshold).sum(axis=0).astype(np.float32)
            vmin, vmax = meta["ranges"]["error_count"]
            return count_pred - count_gt, vmin, vmax, "coolwarm"

        slot   = int(np.clip(slot, 0, n_slots - 1))
        offset = 3 * slot + self.PARAM_FIELDS[field]
        diff   = np.abs(np.asarray(pred[offset], dtype=np.float32) - np.asarray(gt[offset], dtype=np.float32))

        if field in ("mu", "sigma"):
            active = GaussianActivity.is_active(pred[3 * slot], threshold) & GaussianActivity.is_active(gt[3 * slot], threshold)
            diff   = np.where(active, diff, np.nan)

        vmin, vmax = meta["ranges"][f"error_{field}"]
        return diff, vmin, vmax, "inferno"

    def _param_cmap(self, source: str, field: str, meta: dict):
        if source in self.PARAM_SOURCES:
            return "viridis"
        if source == "error" and meta["error"]:
            return "coolwarm" if field == "count" else "inferno"
        return None

    def points_bin(self, cube_id: str, source: str, amp_min: float, max_points: int) -> bytes | None:
        if source in self.PARAM_SOURCES:
            return self._param_points(cube_id, source, amp_min, max_points)
        if source in self.CLOUD_CURVE_SOURCES:
            entry = self._entry(cube_id, source)
            return None if entry is None else self._curve_points(entry, amp_min, max_points)
        return None

    def _param_points(self, cube_id: str, source: str, amp_min: float, max_points: int) -> bytes | None:
        resolved = self._param_state(cube_id)
        if resolved is None:
            return None

        params, _ = resolved
        block     = params.get(source)
        if block is None:
            return None

        amps = block[0::3]
        mus  = block[1::3]
        mask = np.isfinite(amps) & (amps >= amp_min) & np.isfinite(mus)

        k_idx, row_idx, col_idx = np.nonzero(mask)
        total                 = int(k_idx.size)

        if total > max_points > 0:
            keep    = np.linspace(0, total - 1, max_points).astype(int)
            k_idx   = k_idx[keep]
            row_idx = row_idx[keep]
            col_idx = col_idx[keep]

        rows = np.stack([
            row_idx.astype(np.float32),
            col_idx.astype(np.float32),
            mus[k_idx, row_idx, col_idx].astype(np.float32),
            amps[k_idx, row_idx, col_idx].astype(np.float32),
        ], axis=1)

        return self._points_blob(rows, total)

    @classmethod
    def _curve_points(cls, entry: dict, amp_min: float, max_points: int) -> bytes:
        cube   = entry["cube"]
        z_axis = np.asarray(entry["z_axis"], dtype=np.float32)
        n_cols = cube.shape[2]

        counts = np.array([int((np.isfinite(plane) & (plane >= amp_min)).sum()) for plane in cube], dtype=np.int64)
        total  = int(counts.sum())
        starts = np.concatenate([[0], np.cumsum(counts)])

        n_keep = min(total, max_points) if max_points > 0 else total
        picks  = np.linspace(0, total - 1, n_keep).astype(np.int64) if total else np.empty(0, dtype=np.int64)

        chunks = []
        for k, plane in enumerate(cube):
            lo, hi = np.searchsorted(picks, [starts[k], starts[k + 1]])
            if lo == hi:
                continue

            hits = np.flatnonzero(np.isfinite(plane) & (plane >= amp_min))[picks[lo:hi] - starts[k]]
            vals = plane.ravel()[hits]

            chunks.append(np.stack([
                (hits // n_cols).astype(np.float32),
                (hits %  n_cols).astype(np.float32),
                np.full(hits.size, z_axis[k], dtype=np.float32),
                vals.astype(np.float32),
            ], axis=1))

        rows = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4), dtype=np.float32)
        return cls._points_blob(rows, total)

    @staticmethod
    def _points_blob(rows: np.ndarray, total: int) -> bytes:
        header = np.array([rows.shape[0], total, 0.0, 0.0], dtype=np.float32)
        return header.tobytes() + np.ascontiguousarray(rows).tobytes()

    def aux_grid_bin(self, cube_id: str) -> bytes | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            aux = self.loaded["aux"]

        if aux is None:
            return None

        finite = np.isfinite(aux)
        median = float(np.median(aux[finite])) if finite.any() else 0.0
        grid   = (aux - median).astype(np.float32)

        header = np.array([aux.shape[0], aux.shape[1], median, 0.0], dtype=np.float32)
        return header.tobytes() + np.ascontiguousarray(grid).tobytes()

    def metric_overlay_png(self, cube_id: str, key: str, vmin: float, vmax: float, keep_min: float, keep_max: float, alpha: float) -> bytes | None:
        resolved = self._metric_state(cube_id, key)
        if resolved is None:
            return None

        data, primary = resolved

        if not vmax > vmin:
            vmin, vmax = self._metric_range(data)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        p_lo, p_hi = np.percentile(primary, [1.0, 99.0])
        base       = (np.clip((primary - p_lo) / max(p_hi - p_lo, 1e-12), 0.0, 1.0))[..., None].repeat(3, axis=2)

        norm    = np.clip((data - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        colored = plt.get_cmap("viridis")(norm)[..., :3]
        keep    = np.isfinite(data) & (data >= keep_min) & (data <= keep_max)
        blend   = np.where(keep[..., None], base * (1.0 - alpha) + colored * alpha, base)

        buf = io.BytesIO()
        plt.imsave(buf, blend.astype(np.float32), **self.PNG_ENCODE)
        return buf.getvalue()

    def metric_value_at(self, cube_id: str, key: str, row: int, col: int) -> dict:
        resolved = self._metric_state(cube_id, key)
        if resolved is None:
            return {"ok": False, "error": f"unknown metric map: {key}"}

        data, _ = resolved
        row = int(np.clip(row, 0, data.shape[0] - 1))
        col = int(np.clip(col, 0, data.shape[1] - 1))

        value = float(data[row, col])
        return {"ok": True, "row": row, "col": col, "key": key, "value": value if np.isfinite(value) else None}

    def _metric_state(self, cube_id: str, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            data    = self.loaded["metric_maps"].get(key)
            primary = self.loaded["primary"]

        if data is None:
            return None
        return data, primary

    SELECTIVE_KEYS   = ("pixel_mse", "pixel_mae", "pixel_r2", "pixel_cos", "pixel_peak")
    HIGH_IS_CONFIDENT = ("pixel_r2", "pixel_cos", "label_r2")

    def selective_metrics(self, cube_id: str, key: str, coverage: float) -> dict:
        resolved = self._metric_state(cube_id, key)
        if resolved is None:
            return {"ok": False, "error": f"unknown confidence layer: {key}"}

        confidence, _primary = resolved

        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube changed while computing"}
            maps = {name: data for name, data in self.loaded["metric_maps"].items() if name in self.SELECTIVE_KEYS}

        if not maps:
            return {"ok": False, "error": "no pixel metric maps are loaded for this cube"}

        coverage = float(np.clip(coverage, 0.01, 1.0))
        finite   = np.isfinite(confidence)

        if not finite.any():
            return {"ok": False, "error": f"confidence layer '{key}' holds no finite value"}

        keep_high = key in self.HIGH_IS_CONFIDENT

        if keep_high:
            threshold = float(np.quantile(confidence[finite], 1.0 - coverage))
            keep      = finite & (confidence >= threshold)
        else:
            threshold = float(np.quantile(confidence[finite], coverage))
            keep      = finite & (confidence <= threshold)

        rows = []
        for name, data in sorted(maps.items()):
            valid = np.isfinite(data)
            kept  = data[keep & valid]
            full  = data[valid]
            if not full.size:
                continue
            rows.append({
                "key"   : name,
                "label" : self.METRIC_LABELS.get(name, name),
                "kept"  : float(kept.mean()) if kept.size else None,
                "full"  : float(full.mean()),
            })

        return {
            "ok"        : True,
            "layer"     : key,
            "coverage"  : float(keep.sum() / max(finite.sum(), 1)),
            "threshold" : threshold,
            "direction" : "high" if keep_high else "low",
            "n_kept"    : int(keep.sum()),
            "n_total"   : int(finite.sum()),
            "rows"      : rows,
        }

    def transect_png(self, cube_id: str, source: str, row0: int, col0: int, row1: int, col1: int, space: str = "physical", cmap: str = "jet") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None:
            return None

        data, z_values, vmin, vmax = self._transect_cut(entry, row0, col0, row1, col1, space)

        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(data), cmap=self._entry_cmap(entry, cmap), vmin=vmin, vmax=vmax, **self.PNG_ENCODE)
        return buf.getvalue()

    def save_transect(self, cube_id: str, row0: int, col0: int, row1: int, col1: int, space: str = "physical", cmap: str = "jet") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        reader = self._open_source(cube_id)
        if reader is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}
        if space not in ("physical", "normalized"):
            return {"ok": False, "error": f"unknown space: {space}"}

        row0 = int(np.clip(row0, 0, meta["n_rows"] - 1))
        row1 = int(np.clip(row1, 0, meta["n_rows"] - 1))
        col0 = int(np.clip(col0, 0, meta["n_cols"] - 1))
        col1 = int(np.clip(col1, 0, meta["n_cols"] - 1))

        run_dir = reader.run_dir
        rel     = Path("figures") / "cube_transects" / f"row{row0:04d}_col{col0:04d}_to_row{row1:04d}_col{col1:04d}"
        out_dir = run_dir / rel

        files = []
        for source in meta["sources"]:
            entry = entries[source]
            data, z_values, vmin, vmax = self._transect_cut(entry, row0, col0, row1, col1, space)
            saved = self.archiver.render_transect(data, z_values, vmin, vmax, source, (row0, col0), (row1, col1), space, out_dir / f"transect_{source}_{space}.png", cmap=self._entry_cmap(entry, cmap))
            files.append(saved.name)

        self.logger.ok(f"saved {len(files)} transect figures to {out_dir}")
        return {"ok": True, "dir": str(out_dir), "rel": str(rel), "files": files}

    @classmethod
    def _transect_cut(cls, entry: dict, row0: int, col0: int, row1: int, col1: int, space: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        cube               = entry["cube"]
        n_z, n_rows, n_cols = cube.shape

        row0 = int(np.clip(row0, 0, n_rows - 1))
        row1 = int(np.clip(row1, 0, n_rows - 1))
        col0 = int(np.clip(col0, 0, n_cols - 1))
        col1 = int(np.clip(col1, 0, n_cols - 1))

        samples = int(max(abs(row1 - row0), abs(col1 - col0))) + 1
        row_idx = np.round(np.linspace(row0, row1, samples)).astype(int)
        col_idx = np.round(np.linspace(col0, col1, samples)).astype(int)
        data    = cube[:, row_idx, col_idx]

        data, vmin, vmax = cls._normalize_cut(entry, data, space)

        order   = np.argsort(entry["z_axis"])
        z_values = np.asarray(entry["z_axis"], dtype=np.float64)[order]
        return data[order], z_values, float(vmin), float(vmax)

    def save_slices(self, cube_id: str, row: int, col: int, space: str = "physical", cmap: str = "jet") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        reader = self._open_source(cube_id)
        if reader is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}
        if space not in ("physical", "normalized"):
            return {"ok": False, "error": f"unknown space: {space}"}

        row = int(np.clip(row, 0, meta["n_rows"] - 1))
        col = int(np.clip(col, 0, meta["n_cols"] - 1))

        run_dir = reader.run_dir
        rel     = Path("figures") / "cube_slices" / f"row{row:04d}_col{col:04d}"
        out_dir = run_dir / rel

        files = []
        for source in meta["sources"]:
            for axis in ("col", "row"):
                data, z_values, vmin, vmax = self._cut(entries[source], axis, row, col, space)
                saved = self.archiver.render(data, z_values, vmin, vmax, source, axis, row, col, space, out_dir / f"{axis}_{source}_{space}.png", cmap=self._entry_cmap(entries[source], cmap))
                files.append(saved.name)

        self.logger.ok(f"saved {len(files)} slice figures to {out_dir}")
        return {"ok": True, "dir": str(out_dir), "rel": str(rel), "row": row, "col": col, "files": files}

    @classmethod
    def _cut(cls, entry: dict, axis: str, row: int, col: int, space: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        cube = entry["cube"]
        data = cube[:, :, col] if axis == "col" else cube[:, row, :]

        data, vmin, vmax = cls._normalize_cut(entry, data, space)

        order   = np.argsort(entry["z_axis"])
        z_values = np.asarray(entry["z_axis"], dtype=np.float64)[order]
        return data[order], z_values, float(vmin), float(vmax)

    @staticmethod
    def _normalize_cut(entry: dict, data: np.ndarray, space: str) -> tuple[np.ndarray, float, float]:
        if space != "normalized":
            return data, entry["vmin"], entry["vmax"]

        peak = np.abs(data).max(axis=0, keepdims=True) if entry.get("diverging") else data.max(axis=0, keepdims=True)
        safe = np.where(peak > 1e-12, peak, 1.0)
        data = (data / safe).astype(np.float32)
        vmin, vmax = (-1.0, 1.0) if entry.get("diverging") else (0.0, 1.0)
        return data, vmin, vmax

    @classmethod
    def _entry_cmap(cls, entry: dict, cmap: str = "jet") -> str:
        if entry.get("diverging"):
            return "coolwarm"
        return cmap if cmap in cls.CMAPS else "jet"

    def _entry(self, cube_id: str, source: str) -> dict | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            return self.loaded["entries"].get(source)

    def _open_source(self, cube_id: str, progress: LoadProgress | None = None) -> StampReader | None:
        """Resolves a cube id to a reader.

        Args:
            cube_id  : Stamp directory path.
            progress : Byte counter the reader's whole-file reads report to; a silent counter when None.

        Returns:
            A StampReader for a stamp directory that lies under a catalogued root
            and carries a prediction cube, else None.
        """
        if not cube_id:
            return None

        stamp_dir = Path(cube_id).resolve()
        if not self.roots.contains(stamp_dir):
            return None
        if not (stamp_dir / "cubes" / "pred_curves.npy").is_file():
            return None

        return StampReader(stamp_dir, LoadProgress() if progress is None else progress)

    def _load_worker(self, cube_id: str, reader: StampReader) -> None:
        """Loads a cube in the background and publishes the ready or error status.

        Args:
            cube_id : Id of the cube being loaded.
            reader  : Reader opened for that id.
        """
        try:
            entries, meta, primary, params, metric_maps, aux, timer = self._load_all(reader)
            meta["timings"] = {"total": round(timer.total(), 3), "stages": timer.stages}

            with self.lock:
                self.loaded = {"id": cube_id, "entries": entries, "meta": meta, "primary": primary, "primary_clim": self._primary_clim(primary), "params": params, "metric_maps": metric_maps, "aux": aux}
                self.status = {"state": "ready", "id": cube_id, "progress": 1.0, "stage": "ready", "error": ""}
                self.cube_cache.put(reader.cache_key(), self.loaded)

            self.logger.info(f"cube ready in {timer.total():.1f} s: {cube_id} sources={meta['sources']} ({timer.summary()})")
        except Exception as exc:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "id": cube_id, "progress": 0.0, "stage": "", "error": str(exc)}

            self.logger.error(f"cube load failed: {cube_id}: {exc}")

    def _load_all(self, reader: StampReader) -> tuple[dict, dict, np.ndarray, dict, dict, np.ndarray | None, LoadTimer]:
        """Reads every available source and map of one cube.

        Every stage books its wall time on a LoadTimer and names itself in the
        load status the front end polls, so a slow load names the stage that
        cost it; the progress fraction itself follows the bytes the reader's
        whole-file reads book against the plan made here.

        Args:
            reader : Stamp reader to pull from.

        Returns:
            Tuple of the per-source entries, the front-end metadata payload, the
            primary amplitude map in dB of shape (rows, columns), the parameter
            blocks, the metric maps, the cropped aux map (None when the dataset
            run publishes none) and the stage timer.

        Raises:
            ValueError : If the prediction cube is not three-dimensional or another source's spatial shape disagrees with it.
        """
        timer = LoadTimer()

        def stage(name: str, load):
            """Names the running stage in the load status, runs it and books its time."""
            with self.lock:
                self.status["stage"] = name
                self.status_at       = time.perf_counter()

            result = load()
            timer.lap(name)
            return result

        entries, curve_axis = self._read_sources(reader, stage)
        _, n_rows, n_cols   = entries["pred"]["cube"].shape

        primary     = stage("primary",     lambda: self._primary_db(reader, n_rows, n_cols))
        params      = stage("params",      lambda: self._load_params(reader, n_rows, n_cols))
        metric_maps = stage("metric maps", lambda: reader.metric_maps(n_rows, n_cols))
        aux         = stage("aux",         lambda: self._load_aux(reader, n_rows, n_cols))

        meta = {
            "sources"     : [s for s in self.SOURCES if s in entries],
            "n_rows"      : n_rows,
            "n_cols"      : n_cols,
            "n_z"         : {s: int(entries[s]["cube"].shape[0]) for s in entries},
            "x_min"       : float(curve_axis[0]),
            "x_max"       : float(curve_axis[-1]),
            "intensity"   : {s: [entries[s]["vmin"], entries[s]["vmax"]] for s in entries},
            "params"      : self._params_meta(params, curve_axis, reader),
            "metric_maps" : self._metric_maps_meta(metric_maps),
            "attached"    : None,
            "aux"         : aux is not None,
        }
        return entries, meta, primary, params, metric_maps, aux, timer

    def _read_sources(self, reader: StampReader, stage) -> tuple[dict, np.ndarray]:
        """Reads the prediction cube, then the remaining sources concurrently.

        The prediction cube fixes the footprint every other source must match
        and the z axis the curve sources share. Ground truth and the full
        reference cube are then fetched in parallel threads and each is
        ingested as soon as it arrives.

        Args:
            reader : Stamp reader to pull from.
            stage  : Load-step runner taking a stage name and a thunk, returning the thunk's result.

        Returns:
            Tuple of the per-source entries and the curve z axis.

        Raises:
            ValueError : If the prediction cube is not three-dimensional or another source's spatial shape disagrees with it.
        """
        reader.npy.progress.plan(reader.byte_plan())

        pred_raw = stage("pred", lambda: reader.curves("pred"))
        if pred_raw.ndim != 3:
            raise ValueError(f"pred_curves.npy is not a 3D cube: shape={pred_raw.shape}")

        n_z, n_rows, n_cols = pred_raw.shape
        curve_axis          = self._curve_axis(reader, n_z)
        entries             = {"pred": stage("pred", lambda: self._ingest(pred_raw, curve_axis))}

        reader.npy.progress.plan(self._preproc_plan(reader, n_rows))
        with self.lock:
            self.status["stage"] = "gt, full"

        def fetch(source: str) -> tuple:
            """Reads one optional source and returns it with its z axis."""
            if source == "full":
                raw = self._full_raw(reader, n_rows, n_cols)
                return source, raw, None if raw is None else np.arange(raw.shape[0], dtype=np.float64)

            return source, reader.curves(source), curve_axis

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(fetch, source) for source in ("gt", "full")]

            for future in as_completed(futures):
                source, raw, z_axis = future.result()
                if raw is None:
                    stage(source, lambda: None)
                    continue
                if raw.shape[1:] != (n_rows, n_cols):
                    raise ValueError(f"source '{source}' spatial shape {raw.shape[1:]} does not match pred {(n_rows, n_cols)}")

                entries[source] = stage(source, lambda: self._ingest(raw, z_axis))

        return entries, curve_axis

    def _load_aux(self, reader: StampReader, n_rows: int, n_cols: int) -> np.ndarray | None:
        """Crops the aux map of the dataset run to the cube footprint.

        Args:
            reader : Stamp reader identifying the dataset run.
            n_rows : Row extent of the cube, in pixels.
            n_cols : Column extent of the cube, in pixels.

        Returns:
            Float32 aux map of shape (n_rows, n_cols), or None when the dataset run publishes none.

        Raises:
            ValueError : If the aux map does not cover the cube region.
        """
        resolved = self._preproc_layout(reader)
        if resolved is None:
            return None

        preproc_dir, layout = resolved

        aux_name = layout["artifacts"].get("aux_full")
        if not aux_name:
            return None

        aux_path = preproc_dir / "data" / aux_name
        if not aux_path.is_file():
            return None

        row_lo, row_hi, col_lo, col_hi = self._crop_bounds(reader, layout, n_rows, n_cols)

        raw = reader.npy.read(aux_path)
        if raw.ndim != 2 or row_hi > raw.shape[0] or col_hi > raw.shape[1] or row_lo < 0 or col_lo < 0:
            raise ValueError(f"aux_full shape {raw.shape} does not cover the cube region row[{row_lo}:{row_hi}] col[{col_lo}:{col_hi}]")

        return np.asarray(raw[row_lo:row_hi, col_lo:col_hi], dtype=np.float32)

    def _metric_maps_meta(self, metric_maps: dict) -> list:
        layers = []
        for key, data in metric_maps.items():
            vmin, vmax = self._metric_range(data)
            layers.append({
                "key"   : key,
                "label" : self.METRIC_LABELS.get(key, key),
                "vmin"  : vmin,
                "vmax"  : vmax,
            })
        return layers

    @staticmethod
    def _metric_range(data: np.ndarray) -> tuple[float, float]:
        finite = data[np.isfinite(data)]
        if not finite.size:
            return 0.0, 1.0

        vmin, vmax = (float(v) for v in np.percentile(finite, [1.0, 99.0]))
        if not vmax > vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
        if not vmax > vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def _load_params(self, reader: StampReader, n_rows: int, n_cols: int) -> dict:
        """Loads the Gaussian parameter blocks of a stamp and checks them against the cube footprint.

        Args:
            reader : Stamp reader to pull from.
            n_rows : Row extent of the cube, in pixels.
            n_cols : Column extent of the cube, in pixels.

        Returns:
            Mapping from source to its float32 block of shape (3 * n_slots, n_rows, n_cols).

        Raises:
            ValueError : If a block is not a (3K, row, col) cube or its footprint differs from the cube's.
        """
        params = {}
        for source in self.PARAM_SOURCES:
            raw = reader.param_block(source)
            if raw is None:
                continue

            if raw.ndim != 3 or raw.shape[0] % 3 != 0 or raw.shape[0] == 0:
                raise ValueError(f"params_{source}.npy is not a (3K, row, col) cube: shape={raw.shape}")
            if raw.shape[1:] != (n_rows, n_cols):
                raise ValueError(f"params_{source}.npy spatial shape {raw.shape[1:]} does not match the {n_rows}x{n_cols} cube")

            params[source] = raw

        return params

    def _params_meta(self, params: dict, curve_axis: np.ndarray, reader: StampReader) -> dict | None:
        if not params:
            return None

        n_slots   = next(iter(params.values())).shape[0] // 3
        threshold = self._activity_threshold(reader)
        has_error = "pred" in params and "gt" in params

        ranges = {
            "amp"   : [0.0, self._field_ceiling(params, 0)],
            "mu"    : [float(curve_axis[0]), float(curve_axis[-1])],
            "sigma" : [0.0, self._field_ceiling(params, 2)],
            "count" : [0.0, float(n_slots)],
        }

        if has_error:
            for field, offset in self.PARAM_FIELDS.items():
                diff = np.abs(params["pred"][offset::3] - params["gt"][offset::3])
                high = float(np.nanpercentile(diff, 99.0)) if diff.size else 1.0
                ranges[f"error_{field}"] = [0.0, high if high > 0.0 else 1.0]
            ranges["error_count"] = [-float(n_slots), float(n_slots)]

        return {
            "sources"   : [s for s in self.PARAM_SOURCES if s in params],
            "n_slots"   : n_slots,
            "threshold" : threshold,
            "error"     : has_error,
            "ranges"    : ranges,
        }

    def _activity_threshold(self, reader: StampReader) -> float:
        """Returns the activity threshold the stamp's parameter cubes were scored with.

        Args:
            reader : Reader of the inference stamp whose metrics.json is read.

        Raises:
            KeyError : If the metrics carry no activity_threshold, which inference written before the field existed lacks.
        """
        metrics = reader.metrics()
        if "activity_threshold" not in metrics:
            raise KeyError(f"metrics.json of {reader.stamp_dir} has no activity_threshold, so the parameter overlays cannot tell active slots apart; re-run inference for it")

        return float(metrics["activity_threshold"])

    @staticmethod
    def _field_ceiling(params: dict, offset: int) -> float:
        high = 0.0
        for block in params.values():
            channels = block[offset::3]
            if channels.size:
                high = max(high, float(np.nanpercentile(channels, 99.0)))
        return high if high > 0.0 else 1.0

    def _ingest(self, raw: np.ndarray, z_axis: np.ndarray) -> dict:
        """Wraps an in-memory cube as a float32 entry with display colour limits.

        Args:
            raw    : Cube of shape (z bins, rows, columns); complex cubes are reduced to their magnitude.
            z_axis : Curve z axis, or bin indices for the full reference cube.

        Returns:
            Entry with the float32 ``cube``, its ``z_axis`` and the 1st and 99th
            percentile colour limits estimated on a bounded subsample.
        """
        cube       = np.abs(raw).astype(np.float32) if np.iscomplexobj(raw) else np.asarray(raw, dtype=np.float32)
        vmin, vmax = self._colour_limits(cube)

        return {
            "cube"   : cube,
            "z_axis" : z_axis,
            "vmin"   : vmin,
            "vmax"   : vmax,
        }

    @classmethod
    def _colour_limits(cls, raw: np.ndarray) -> tuple[float, float]:
        """Estimates 1st and 99th percentile colour limits on a bounded subsample.

        Args:
            raw : Cube of any shape; complex values are reduced to their magnitude.

        Returns:
            Tuple (vmin, vmax), or (0, 1) when the cube holds no finite value.
        """
        sample = cls._limit_sample(raw)
        sample = np.abs(sample) if np.iscomplexobj(sample) else sample
        if not sample.size:
            return 0.0, 1.0

        vmin, vmax = np.percentile(sample, [1.0, 99.0])
        return float(vmin), float(vmax)

    @classmethod
    def _limit_sample(cls, raw: np.ndarray) -> np.ndarray:
        """Returns at most LIMIT_SAMPLE finite values taken at a fixed stride over the flattened cube."""
        flat   = raw.reshape(-1)
        sample = np.asarray(flat[:: max(1, flat.size // cls.LIMIT_SAMPLE)])

        return sample[np.isfinite(sample)]

    def _curve_axis(self, reader: StampReader, n_z: int) -> np.ndarray:
        """Returns the n_z evenly spaced z bin centres recorded in the stamp's metrics."""
        metrics = reader.metrics()
        return np.linspace(float(metrics["x_axis_min"]), float(metrics["x_axis_max"]), n_z)

    def _preproc_plan(self, reader: StampReader, n_rows: int) -> int:
        """Returns the bytes of the dataset artifacts a load reads: the full cube's row band, the primary channel and the aux map.

        Args:
            reader : Stamp reader identifying the dataset run.
            n_rows : Row extent of the cube, in pixels, sizing the full cube's band.
        """
        resolved = self._preproc_layout(reader)
        if resolved is None:
            return 0

        preproc_dir, layout = resolved
        planned             = 0
        for key in ("primary", "aux_full"):
            name = layout["artifacts"].get(key)
            path = preproc_dir / "data" / name if name else None
            if path is not None and path.is_file():
                planned += path.stat().st_size

        cube_name = layout["artifacts"].get("cube_full")
        cube_path = preproc_dir / "data" / cube_name if cube_name else None
        if cube_path is not None and cube_path.is_file():
            shape, dtype = reader.npy.header(cube_path)
            planned     += shape[0] * n_rows * shape[2] * dtype.itemsize

        return planned

    def _preproc_layout(self, reader: StampReader) -> tuple[Path, dict] | None:
        """Resolves the dataset run backing a cube and its dataset layout.

        Args:
            reader : Stamp reader whose run records the dataset run it was trained on.

        Returns:
            Tuple of the dataset run directory and its parsed ``dataset.json``, or
            None when the run records no dataset run or its layout is missing.
        """
        meta_path = reader.run_dir / "meta" / "dataset_creation_config.json"
        if not meta_path.is_file():
            return None

        payload     = json.loads(meta_path.read_text(encoding="utf-8"))
        preproc_dir = Path(payload["preprocessing_run_directory"])
        layout_path = preproc_dir / "data" / "dataset.json"
        if not layout_path.is_file():
            return None

        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        return preproc_dir, layout

    def _crop_bounds(self, reader: StampReader, layout: dict, n_rows: int, n_cols: int) -> tuple[int, int, int, int]:
        """Returns the cube region as row and column bounds into the dataset's cropped arrays.

        Args:
            reader : Stamp reader whose metrics record the split region.
            layout : Parsed ``dataset.json`` carrying the global crop.
            n_rows : Row extent of the cube, in pixels.
            n_cols : Column extent of the cube, in pixels.

        Raises:
            KeyError   : If the stamp's metrics carry no split_region.
            ValueError : If the split region does not match the cube footprint.
        """
        metrics = reader.metrics()
        if "split_region" not in metrics:
            raise KeyError(f"metrics.json in {reader.stamp_dir} has no split_region; rerun inference to regenerate it")

        row_start, row_end, col_start, col_end = (int(v) for v in metrics["split_region"])
        if (row_end - row_start, col_end - col_start) != (n_rows, n_cols):
            raise ValueError(f"split_region {metrics['split_region']} does not match the {n_rows}x{n_cols} cube")

        global_crop = layout["global_crop"]

        row_lo = row_start - global_crop[0]
        row_hi = row_end   - global_crop[0]
        col_lo = col_start - global_crop[2]
        col_hi = col_end   - global_crop[2]
        return row_lo, row_hi, col_lo, col_hi

    def _full_raw(self, reader: StampReader, n_rows: int, n_cols: int, view: bool = False) -> np.ndarray | None:
        """Crops the full reference cube of the dataset run to the cube footprint.

        Only the row band the cube covers is read from the file, one contiguous
        read per z plane; callers that only touch a few lines of it ask for a
        memory-mapped view instead.

        Args:
            reader : Stamp reader identifying the dataset run.
            n_rows : Row extent of the cube, in pixels.
            n_cols : Column extent of the cube, in pixels.
            view   : When True the file is memory-mapped and the crop returned as a view of it.

        Returns:
            Crop of shape (z bins, n_rows, n_cols), or None when the dataset run publishes no full cube.

        Raises:
            ValueError : If the full cube is not three-dimensional or does not cover the cube region.
        """
        resolved = self._preproc_layout(reader)
        if resolved is None:
            return None

        preproc_dir, layout = resolved

        cube_name = layout["artifacts"].get("cube_full")
        if not cube_name:
            return None

        cube_path = preproc_dir / "data" / cube_name
        if not cube_path.is_file():
            return None

        row_lo, row_hi, col_lo, col_hi = self._crop_bounds(reader, layout, n_rows, n_cols)

        shape, _ = reader.npy.header(cube_path)
        if len(shape) != 3:
            raise ValueError(f"full cube is not a 3D cube: shape={shape}")
        if row_lo < 0 or col_lo < 0 or row_hi > shape[1] or col_hi > shape[2]:
            raise ValueError(f"cube region row[{row_lo}:{row_hi}] col[{col_lo}:{col_hi}] falls outside the full cube {shape}")

        if view:
            return np.load(cube_path, mmap_mode="r")[:, row_lo:row_hi, col_lo:col_hi]

        return np.ascontiguousarray(reader.npy.read_band(cube_path, row_lo, row_hi)[:, :, col_lo:col_hi])

    def _primary_db(self, reader: StampReader, n_rows: int, n_cols: int) -> np.ndarray:
        """Crops the primary channel and returns its amplitude in dB.

        Args:
            reader : Stamp reader identifying the dataset run.
            n_rows : Row extent of the cube, in pixels.
            n_cols : Column extent of the cube, in pixels.

        Returns:
            Float32 map of shape (n_rows, n_cols) holding 20 log10 of the amplitude.

        Raises:
            FileNotFoundError : If the dataset run or its primary channel cannot be found.
            ValueError        : If the primary channel is not 2D or does not cover the cube region.
        """
        resolved = self._preproc_layout(reader)
        if resolved is None:
            raise FileNotFoundError(f"cannot resolve the preprocessing run for {reader.stamp_dir}; the primary channel is required for the cube map")

        preproc_dir, layout = resolved

        primary_name = layout["artifacts"].get("primary")
        if not primary_name:
            raise FileNotFoundError(f"dataset.json in {preproc_dir} lists no primary artifact")

        primary_path = preproc_dir / "data" / primary_name
        if not primary_path.is_file():
            raise FileNotFoundError(f"primary channel missing: {primary_path}")

        row_lo, row_hi, col_lo, col_hi = self._crop_bounds(reader, layout, n_rows, n_cols)

        raw = reader.npy.read(primary_path)
        if raw.ndim != 2:
            raise ValueError(f"primary channel is not a 2D image: shape={raw.shape}")
        if row_lo < 0 or col_lo < 0 or row_hi > raw.shape[0] or col_hi > raw.shape[1]:
            raise ValueError(f"cube region row[{row_lo}:{row_hi}] col[{col_lo}:{col_hi}] falls outside the primary channel {raw.shape}")

        amplitude = np.abs(np.asarray(raw[row_lo:row_hi, col_lo:col_hi])).astype(np.float32)
        return 20.0 * np.log10(np.maximum(amplitude, 1e-12))

    @staticmethod
    def _primary_clim(primary: np.ndarray) -> tuple[float, float]:
        """Returns the 1st and 99th percentile of a primary amplitude map, in dB."""
        return tuple(float(v) for v in np.percentile(primary, [1.0, 99.0]))

    @classmethod
    def _encode_primary(cls, primary: np.ndarray, clim: tuple[float, float]) -> bytes:
        """Encodes a primary amplitude map as a greyscale PNG.

        Args:
            primary : Float array of shape (rows, columns) holding the amplitude in dB.
            clim    : Grey-scale limits (vmin, vmax) in dB.

        Returns:
            PNG bytes of the map, one image pixel per cube pixel.
        """
        buf = io.BytesIO()
        plt.imsave(buf, primary, cmap="gray", vmin=float(clim[0]), vmax=float(clim[1]), **cls.PNG_ENCODE)
        return buf.getvalue()


class SliceCollector:

    SOURCES    = ("pred", "gt", "full")
    AXES       = ("col", "row")
    MAX_RUNS   = 24
    MAX_POINTS = 24
    CACHE_CAP  = 16

    def __init__(self, cubes: CubeExplorer, logger: WebLogger) -> None:
        self.cubes       = cubes
        self.logger      = logger
        self.lock        = threading.Lock()
        self.render_lock = threading.Lock()
        self.contexts    = OrderedDict()

    def info(self, cube_id: str) -> dict:
        try:
            context = self._context(cube_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        if context is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}

        entries = context["entries"]
        return {
            "ok"        : True,
            "id"        : str(context["stamp_dir"]),
            "run"       : context["run"],
            "group"     : context["group"],
            "stamp"     : context["stamp"],
            "n_rows"    : context["n_rows"],
            "n_cols"    : context["n_cols"],
            "sources"   : [s for s in self.SOURCES if s in entries],
            "intensity" : {s: [entries[s]["vmin"], entries[s]["vmax"]] for s in entries},
        }

    def slice_png(self, cube_id: str, source: str, axis: str, row: int, col: int, space: str = "physical", cmap: str = "jet", vmin: float | None = None, vmax: float | None = None) -> bytes | None:
        if source not in self.SOURCES or axis not in self.AXES or space not in ("physical", "normalized"):
            return None

        try:
            context = self._context(cube_id)
        except Exception as error:
            self.logger.warning(f"slice render for {cube_id} failed while opening the cube: {error}")
            return None

        if context is None:
            self.logger.warning(f"slice render refused: unknown cube id {cube_id}")
            return None

        entry = context["entries"].get(source)
        if entry is None:
            return None

        data, z_values, lo, hi = self._cut(entry, axis, row, col, space)
        if space == "physical" and vmin is not None and vmax is not None and vmax > vmin:
            lo, hi = float(vmin), float(vmax)

        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(data), cmap=CubeExplorer._entry_cmap(entry, cmap), vmin=lo, vmax=hi, **CubeExplorer.PNG_ENCODE)
        return buf.getvalue()

    def collect(self, ids: list, points: list, sources: list, axes: list, space: str = "physical", cmap: str = "jet", shared: bool = True, name: str = "") -> dict:
        ids     = list(dict.fromkeys(str(i) for i in ids if i))
        sources = list(dict.fromkeys(sources))
        axes    = list(dict.fromkeys(axes))

        if not ids:
            return {"ok": False, "error": "select at least one run"}
        if len(ids) > self.MAX_RUNS:
            return {"ok": False, "error": f"at most {self.MAX_RUNS} runs per collection, got {len(ids)}"}
        if space not in ("physical", "normalized"):
            return {"ok": False, "error": f"unknown space: {space}"}
        if not sources or any(s not in self.SOURCES for s in sources):
            return {"ok": False, "error": f"sources must be a non-empty subset of {self.SOURCES}"}
        if not axes or any(a not in self.AXES for a in axes):
            return {"ok": False, "error": f"axes must be a non-empty subset of {self.AXES}"}

        try:
            points   = self._points(points)
            contexts = []
            for cube_id in ids:
                context = self._context(cube_id)
                if context is None:
                    return {"ok": False, "error": f"unknown cube id: {cube_id}"}
                contexts.append(context)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        labels  = self._labels(contexts)
        title   = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("._")
        title   = title or datetime.now().strftime("collection_%Y%m%d_%H%M%S")
        out_dir = contexts[0]["root"] / "slice_collections" / title
        clims   = {source: self._shared_clim(contexts, source, space) if shared else None for source in sources}
        missing = [{"label": label, "source": source} for source in sources for context, label in zip(contexts, labels) if source not in context["entries"]]

        files = []
        with self.render_lock:
            for row, col in points:
                point_dir = out_dir / f"row{row:04d}_col{col:04d}"
                for source in sources:
                    clim = clims[source]
                    for axis in axes:
                        cut_dir = point_dir / f"{axis}_{source}_{space}"
                        for context, label in zip(contexts, labels):
                            entry = context["entries"].get(source)
                            if entry is None:
                                continue

                            row_i = int(np.clip(row, 0, context["n_rows"] - 1))
                            col_i = int(np.clip(col, 0, context["n_cols"] - 1))
                            data, z_values, vmin, vmax = self._cut(entry, axis, row_i, col_i, space)
                            if clim is not None:
                                vmin, vmax = clim

                            saved = self.cubes.archiver.render(data, z_values, vmin, vmax, source, axis, row_i, col_i, space, cut_dir / f"{label}.png", cmap=CubeExplorer._entry_cmap(entry, cmap), label=label)
                            files.append(str(saved.relative_to(out_dir)))

        if not files:
            return {"ok": False, "error": "no figures rendered; none of the selected runs carry the requested sources"}

        manifest = {
            "name"    : title,
            "created" : datetime.now().isoformat(timespec="seconds"),
            "space"   : space,
            "cmap"    : cmap,
            "shared"  : bool(shared),
            "axes"    : axes,
            "sources" : sources,
            "points"  : [{"row": row, "col": col} for row, col in points],
            "clims"   : {source: (list(clim) if clim else None) for source, clim in clims.items()},
            "runs"    : [{"id": str(c["stamp_dir"]), "run": c["run"], "group": c["group"], "stamp": c["stamp"], "label": label} for c, label in zip(contexts, labels)],
            "missing" : missing,
            "files"   : files,
        }
        (out_dir / "collection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self.logger.ok(f"collected {len(files)} slice figures from {len(contexts)} runs into {out_dir}")
        return {"ok": True, "dir": str(out_dir), "name": title, "files": files, "missing": missing, "runs": len(contexts)}

    def _points(self, raw) -> list:
        if not isinstance(raw, list) or not raw:
            raise ValueError("points must be a non-empty list of {row, col}")
        if len(raw) > self.MAX_POINTS:
            raise ValueError(f"at most {self.MAX_POINTS} points per collection, got {len(raw)}")

        points = []
        seen   = set()
        for entry in raw:
            row, col = int(entry["row"]), int(entry["col"])
            if row < 0 or col < 0:
                raise ValueError(f"point row={row}, col={col} is negative")
            if (row, col) in seen:
                continue
            seen.add((row, col))
            points.append((row, col))

        return points

    @staticmethod
    def _labels(contexts: list) -> list:
        counts = {}
        for context in contexts:
            counts[context["run"]] = counts.get(context["run"], 0) + 1

        grouped = []
        for context in contexts:
            label = context["run"]
            if counts[label] > 1 and context["group"] not in (".", ""):
                label = f"{context['group'].replace('/', '_')}__{label}"
            grouped.append(label)

        labels = []
        for context, label in zip(contexts, grouped):
            if grouped.count(label) > 1:
                label = f"{label}__{context['stamp']}"
            labels.append(label)

        return labels

    @staticmethod
    def _shared_clim(contexts: list, source: str, space: str) -> tuple[float, float] | None:
        if space != "physical":
            return None

        entries = [context["entries"][source] for context in contexts if source in context["entries"]]
        if not entries:
            return None

        vmin = min(entry["vmin"] for entry in entries)
        vmax = max(entry["vmax"] for entry in entries)
        return (vmin, vmax) if vmax > vmin else None

    @staticmethod
    def _cut(entry: dict, axis: str, row: int, col: int, space: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        cube               = entry["cube"]
        n_z, n_rows, n_cols = cube.shape

        row  = int(np.clip(row, 0, n_rows - 1))
        col  = int(np.clip(col, 0, n_cols - 1))
        data = np.asarray(cube[:, :, col] if axis == "col" else cube[:, row, :])
        data = np.abs(data).astype(np.float32) if np.iscomplexobj(data) else data.astype(np.float32)

        data, vmin, vmax = CubeExplorer._normalize_cut(entry, data, space)

        order   = np.argsort(entry["z_axis"])
        z_values = np.asarray(entry["z_axis"], dtype=np.float64)[order]
        return data[order], z_values, float(vmin), float(vmax)

    def _context(self, cube_id: str) -> dict | None:
        reader = self.cubes._open_source(cube_id)
        if reader is None:
            return None

        key = reader.cache_key()
        with self.lock:
            if key in self.contexts:
                self.contexts.move_to_end(key)
                return self.contexts[key]

        context = self._build_context(reader)

        with self.lock:
            self.contexts[key] = context
            while len(self.contexts) > self.CACHE_CAP:
                self.contexts.popitem(last=False)

        return context

    def _build_context(self, reader: StampReader) -> dict:
        stamp_dir = reader.stamp_dir
        root      = self._root_for(stamp_dir)
        if root is None:
            raise ValueError(f"{stamp_dir} is outside every catalogued runs root")

        pred_raw = reader.curves("pred", view=True)
        if pred_raw.ndim != 3:
            raise ValueError(f"pred_curves.npy is not a 3D cube: shape={pred_raw.shape}")

        n_z, n_rows, n_cols = pred_raw.shape
        curve_axis          = self.cubes._curve_axis(reader, n_z)

        entries = {"pred": self._entry(pred_raw, curve_axis)}
        gt_raw  = reader.curves("gt", view=True)
        if gt_raw is not None:
            if gt_raw.shape[1:] != (n_rows, n_cols):
                raise ValueError(f"source 'gt' spatial shape {gt_raw.shape[1:]} does not match pred {(n_rows, n_cols)}")
            entries["gt"] = self._entry(gt_raw, curve_axis)

        full_raw = self.cubes._full_raw(reader, n_rows, n_cols, view=True)
        if full_raw is not None:
            entries["full"] = self._entry(full_raw, np.arange(full_raw.shape[0], dtype=np.float64))

        run_dir = reader.run_dir
        return {
            "stamp_dir" : stamp_dir,
            "root"      : root,
            "run"       : run_dir.name,
            "group"     : str(run_dir.relative_to(root).parent),
            "stamp"     : stamp_dir.name,
            "n_rows"    : n_rows,
            "n_cols"    : n_cols,
            "entries"   : entries,
        }

    @staticmethod
    def _entry(raw: np.ndarray, z_axis: np.ndarray) -> dict:
        vmin, vmax = CubeExplorer._colour_limits(raw)

        return {
            "cube"   : raw,
            "z_axis" : z_axis,
            "vmin"   : vmin,
            "vmax"   : vmax,
        }

    def _root_for(self, stamp_dir: Path) -> Path | None:
        return self.cubes.roots.enclosing(stamp_dir)

