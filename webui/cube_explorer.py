from __future__ import annotations

import io
import json
import re
import threading
from collections import OrderedDict
from datetime    import datetime
from pathlib     import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

from catalog_roots              import CatalogRoots, RunScanner
from tools.loss.param_loss      import ParamMatcher
from tools.reporting.plotting   import PlotBase
from tools.sar.track_parameters import TrackParameters
from web_logger                 import WebLogger


class SliceFigureArchiver(PlotBase):

    LABELS = {
        "pred"    : "Prediction",
        "predb"   : "Prediction B",
        "diff"    : "Prediction A − B",
        "gt"      : "GT (Gaussian)",
        "reduced" : "Capon reduced",
        "full"    : "Capon full (raw)",
    }

    def render(self, data: np.ndarray, heights: np.ndarray, vmin: float, vmax: float, source: str, axis: str, az: int, rg: int, space: str, path: Path, cmap: str = "jet", label: str | None = None) -> Path:
        x_label   = "azimuth index" if axis == "range" else "range index"
        title_pos = f"range = {rg}" if axis == "range" else f"azimuth = {az}"
        y_label   = "elevation bin" if source == "full" else "elevation [m]"
        cbar      = "intensity (per-column normalised)" if space == "normalized" else "intensity"
        extent    = [0, int(data.shape[1]), float(heights[0]), float(heights[-1])]
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

    def render_transect(self, data: np.ndarray, heights: np.ndarray, vmin: float, vmax: float, source: str, start: tuple, end: tuple, space: str, path: Path, cmap: str = "jet") -> Path:
        cbar   = "intensity (per-column normalised)" if space == "normalized" else "intensity"
        extent = [0, int(data.shape[1]), float(heights[0]), float(heights[-1])]

        previous = PlotBase.style
        PlotBase.use_style("paper")
        try:
            return self._imshow_figure(
                data,
                x_label        = "sample along transect",
                y_label        = "elevation bin" if source == "full" else "elevation [m]",
                title          = f"{self.LABELS[source]} — transect az{start[0]},rg{start[1]} to az{end[0]},rg{end[1]}",
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


class CubeExplorer:

    SOURCES             = ("pred", "predb", "diff", "gt", "reduced", "full")
    PARAM_SOURCES       = ("pred", "gt")
    CLOUD_CURVE_SOURCES = ("reduced", "full")
    PARAM_FIELDS        = {"amp": 0, "mu": 1, "sigma": 2}
    PARAM_BAD           = "#10151a"

    CMAPS = ("jet", "viridis", "inferno", "turbo", "gray")

    METRIC_EXCLUDED = ("_curves", "params_")
    METRIC_LABELS   = {
        "pixel_mse"                : "MSE",
        "pixel_mae"                : "MAE",
        "pixel_r2"                 : "R2",
        "pixel_cos"                : "cosine",
        "pixel_peak"               : "peak shift",
        "physics_coherence_error"  : "coherence err",
        "physics_covariance_error" : "covariance err",
        "physics_valid_mask"       : "valid mask",
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
        self.logger   = logger
        self.archiver = SliceFigureArchiver()
        self.roots    = CatalogRoots()
        self.scanner  = RunScanner(self.roots)
        self.lock     = threading.Lock()
        self.loaded   = None
        self.status   = {"state": "idle", "id": None, "progress": 0.0, "stage": "", "error": ""}

    def list_cubes(self, base: str) -> dict:
        scanned = self.scanner.stamps(base)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "cubes": []}

        return {"ok": True, "root": scanned["root"], "cubes": scanned["entries"]}

    def start_load(self, cube_id: str) -> dict:
        stamp_dir = self._stamp_dir(cube_id)
        if stamp_dir is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}

        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": f"a load is already running for {self.status['id']}"}
            if self.status["state"] == "ready" and self.status["id"] == cube_id and self.loaded is not None:
                return {"ok": True}

            self.loaded = None
            self.status = {"state": "loading", "id": cube_id, "progress": 0.0, "stage": "scanning sources", "error": ""}

        threading.Thread(target=self._load_worker, args=(cube_id, stamp_dir), daemon=True).start()
        return {"ok": True}

    def load_status(self) -> dict:
        with self.lock:
            payload = dict(self.status)
            if payload["state"] == "ready" and self.loaded is not None:
                payload["cube"] = self.loaded["meta"]
        return payload

    def primary_png(self, cube_id: str) -> bytes | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            primary = self.loaded["primary"]

        vmin, vmax = np.percentile(primary, [1.0, 99.0])

        buf = io.BytesIO()
        plt.imsave(buf, primary, cmap="gray", vmin=float(vmin), vmax=float(vmax), format="png")
        return buf.getvalue()

    def attach_second(self, cube_id: str, other_id: str) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            pred = self.loaded["entries"]["pred"]

        other_dir = self._stamp_dir(other_id)
        if other_dir is None:
            return {"ok": False, "error": f"unknown cube id: {other_id}"}
        if other_id == cube_id:
            return {"ok": False, "error": "pick a different inference result to compare against"}

        raw = np.load(other_dir / "cubes" / "pred_curves.npy", mmap_mode="r")
        if raw.shape != pred["cube"].shape:
            return {"ok": False, "error": f"comparison cube shape {tuple(raw.shape)} does not match {tuple(pred['cube'].shape)}"}

        other_axis = self._curve_axis(other_dir, raw.shape[0])
        if not np.allclose(other_axis, pred["x_axis"]):
            return {"ok": False, "error": "comparison cube covers a different elevation axis"}

        predb = self._ingest(raw, pred["x_axis"], lambda: None)
        diff  = self._diff_entry(pred, predb)

        run_dir = other_dir.parent.parent
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}

            self.loaded["entries"]["predb"] = predb
            self.loaded["entries"]["diff"]  = diff

            meta = dict(self.loaded["meta"])
            meta["sources"]   = [s for s in self.SOURCES if s in self.loaded["entries"]]
            meta["n_elev"]    = {s: int(self.loaded["entries"][s]["cube"].shape[0]) for s in self.loaded["entries"]}
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
            meta["n_elev"]    = {s: int(self.loaded["entries"][s]["cube"].shape[0]) for s in self.loaded["entries"]}
            meta["intensity"] = {s: [e["vmin"], e["vmax"]] for s, e in self.loaded["entries"].items()}
            meta["attached"]  = None
            self.loaded["meta"] = meta

        return {"ok": True, "cube": meta}

    @staticmethod
    def _diff_entry(pred: dict, predb: dict) -> dict:
        cube = pred["cube"] - predb["cube"]

        sample = cube[:, :: max(1, cube.shape[1] // 256), :: max(1, cube.shape[2] // 256)]
        sample = np.abs(sample[np.isfinite(sample)])
        peak   = float(np.percentile(sample, 99.0)) if sample.size else 1.0
        peak   = peak if peak > 0.0 else 1.0

        return {
            "cube"      : cube,
            "x_axis"    : pred["x_axis"],
            "vmin"      : -peak,
            "vmax"      : peak,
            "diverging" : True,
        }

    def profiles(self, cube_id: str, az: int, rg: int) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        az = int(np.clip(az, 0, meta["n_az"] - 1))
        rg = int(np.clip(rg, 0, meta["n_rg"] - 1))

        sources = {}
        for source, entry in entries.items():
            order   = np.argsort(entry["x_axis"])
            heights = np.asarray(entry["x_axis"])[order]
            values  = entry["cube"][:, az, rg][order]
            sources[source] = {"heights": heights.tolist(), "values": values.astype(float).tolist()}

        return {"ok": True, "az": az, "rg": rg, "sources": sources}

    def slice_ssim(self, cube_id: str, az: int, rg: int, space: str = "physical") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        gt = entries.get("gt")
        if gt is None:
            return {"ok": True, "az": int(az), "rg": int(rg), "range": {}, "azimuth": {}}

        az = int(np.clip(az, 0, meta["n_az"] - 1))
        rg = int(np.clip(rg, 0, meta["n_rg"] - 1))

        gt_cube = gt["cube"]
        out     = {"range": {}, "azimuth": {}}

        for source in ("pred", "predb", "reduced", "full"):
            entry = entries.get(source)
            if entry is None or entry["cube"].shape != gt_cube.shape:
                continue

            cube = entry["cube"]
            out["range"][source]   = self._ssim_score(cube[:, :, rg], gt_cube[:, :, rg], space)
            out["azimuth"][source] = self._ssim_score(cube[:, az, :], gt_cube[:, az, :], space)

        return {"ok": True, "az": az, "rg": rg, "range": out["range"], "azimuth": out["azimuth"]}

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

    def slice_png(self, cube_id: str, source: str, axis: str, az: int, rg: int, space: str = "physical", cmap: str = "jet") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None or axis not in ("range", "azimuth"):
            return None

        n_elev, n_az, n_rg = entry["cube"].shape
        az = int(np.clip(az, 0, n_az - 1))
        rg = int(np.clip(rg, 0, n_rg - 1))

        data, heights, vmin, vmax = self._cut(entry, axis, az, rg, space)

        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(data), cmap=self._entry_cmap(entry, cmap), vmin=vmin, vmax=vmax, format="png")
        return buf.getvalue()

    def plane_png(self, cube_id: str, source: str, frac: float, space: str = "physical", cmap: str = "jet") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None:
            return None

        cube   = entry["cube"]
        n_elev = cube.shape[0]
        order  = np.argsort(entry["x_axis"])
        pos    = int(round(float(np.clip(frac, 0.0, 1.0)) * (n_elev - 1)))
        elev   = int(order[pos])

        data = np.asarray(cube[elev], dtype=np.float32)

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
        plt.imsave(buf, np.nan_to_num(data, nan=vmin), cmap=self._entry_cmap(entry, cmap), vmin=vmin, vmax=vmax, format="png")
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
        plt.imsave(buf, data, cmap=palette, vmin=vmin, vmax=vmax, format="png")
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
        plt.imsave(buf, ramp, cmap=cmap, vmin=0.0, vmax=1.0, format="png")
        return buf.getvalue()

    def params_at(self, cube_id: str, az: int, rg: int) -> dict:
        resolved = self._param_state(cube_id)
        if resolved is None:
            return {"ok": False, "error": "no parameter cubes are loaded"}

        params, meta = resolved
        n_slots      = meta["n_slots"]
        threshold    = meta["threshold"]

        az = int(np.clip(az, 0, next(iter(params.values())).shape[1] - 1))
        rg = int(np.clip(rg, 0, next(iter(params.values())).shape[2] - 1))

        sources = {}
        for source, block in params.items():
            slots = []
            for k in range(n_slots):
                amp   = float(block[3 * k,     az, rg])
                mu    = float(block[3 * k + 1, az, rg])
                sigma = float(block[3 * k + 2, az, rg])
                slots.append({"amp": amp, "mu": mu, "sigma": sigma, "active": bool(amp >= threshold)})
            sources[source] = slots

        return {"ok": True, "az": az, "rg": rg, "n_slots": n_slots, "threshold": threshold, "sources": sources}

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
            data = (amps >= threshold).sum(axis=0).astype(np.float32)
            vmin, vmax = meta["ranges"]["count"]
            return data, vmin, vmax, "viridis"

        slot    = int(np.clip(slot, 0, n_slots - 1))
        channel = np.asarray(block[3 * slot + self.PARAM_FIELDS[field]], dtype=np.float32)

        if field in ("mu", "sigma"):
            active  = block[3 * slot] >= threshold
            channel = np.where(active, channel, np.nan)

        vmin, vmax = meta["ranges"][field]
        return channel, vmin, vmax, "viridis"

    def _param_error_map(self, params: dict, meta: dict, field: str, slot: int) -> tuple[np.ndarray, float, float, str]:
        threshold = meta["threshold"]
        n_slots   = meta["n_slots"]
        pred      = params["pred"]
        gt        = params["gt"]

        if field == "count":
            count_pred = (pred[0::3] >= threshold).sum(axis=0).astype(np.float32)
            count_gt   = (gt[0::3]   >= threshold).sum(axis=0).astype(np.float32)
            vmin, vmax = meta["ranges"]["error_count"]
            return count_pred - count_gt, vmin, vmax, "coolwarm"

        slot   = int(np.clip(slot, 0, n_slots - 1))
        offset = 3 * slot + self.PARAM_FIELDS[field]
        diff   = np.abs(np.asarray(pred[offset], dtype=np.float32) - np.asarray(gt[offset], dtype=np.float32))

        if field in ("mu", "sigma"):
            active = (pred[3 * slot] >= threshold) & (gt[3 * slot] >= threshold)
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

        k_idx, az_idx, rg_idx = np.nonzero(mask)
        total                 = int(k_idx.size)

        if total > max_points > 0:
            keep   = np.linspace(0, total - 1, max_points).astype(int)
            k_idx  = k_idx[keep]
            az_idx = az_idx[keep]
            rg_idx = rg_idx[keep]

        rows = np.stack([
            az_idx.astype(np.float32),
            rg_idx.astype(np.float32),
            mus[k_idx, az_idx, rg_idx].astype(np.float32),
            amps[k_idx, az_idx, rg_idx].astype(np.float32),
        ], axis=1)

        return self._points_blob(rows, total)

    @classmethod
    def _curve_points(cls, entry: dict, amp_min: float, max_points: int) -> bytes:
        cube   = entry["cube"]
        x_axis = np.asarray(entry["x_axis"], dtype=np.float32)
        n_rg   = cube.shape[2]

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
                (hits // n_rg).astype(np.float32),
                (hits %  n_rg).astype(np.float32),
                np.full(hits.size, x_axis[k], dtype=np.float32),
                vals.astype(np.float32),
            ], axis=1))

        rows = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4), dtype=np.float32)
        return cls._points_blob(rows, total)

    @staticmethod
    def _points_blob(rows: np.ndarray, total: int) -> bytes:
        header = np.array([rows.shape[0], total, 0.0, 0.0], dtype=np.float32)
        return header.tobytes() + np.ascontiguousarray(rows).tobytes()

    def dem_grid_bin(self, cube_id: str) -> bytes | None:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return None
            dem = self.loaded["dem"]

        if dem is None:
            return None

        finite = np.isfinite(dem)
        median = float(np.median(dem[finite])) if finite.any() else 0.0
        grid   = (dem - median).astype(np.float32)

        header = np.array([dem.shape[0], dem.shape[1], median, 0.0], dtype=np.float32)
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
        plt.imsave(buf, blend.astype(np.float32), format="png")
        return buf.getvalue()

    def metric_value_at(self, cube_id: str, key: str, az: int, rg: int) -> dict:
        resolved = self._metric_state(cube_id, key)
        if resolved is None:
            return {"ok": False, "error": f"unknown metric map: {key}"}

        data, _ = resolved
        az = int(np.clip(az, 0, data.shape[0] - 1))
        rg = int(np.clip(rg, 0, data.shape[1] - 1))

        value = float(data[az, rg])
        return {"ok": True, "az": az, "rg": rg, "key": key, "value": value if np.isfinite(value) else None}

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
    HIGH_IS_CONFIDENT = ("pixel_r2", "pixel_cos", "label_r2", "physics_valid_mask")

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

    def transect_png(self, cube_id: str, source: str, az0: int, rg0: int, az1: int, rg1: int, space: str = "physical", cmap: str = "jet") -> bytes | None:
        entry = self._entry(cube_id, source)
        if entry is None:
            return None

        data, heights, vmin, vmax = self._transect_cut(entry, az0, rg0, az1, rg1, space)

        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(data), cmap=self._entry_cmap(entry, cmap), vmin=vmin, vmax=vmax, format="png")
        return buf.getvalue()

    def save_transect(self, cube_id: str, az0: int, rg0: int, az1: int, rg1: int, space: str = "physical", cmap: str = "jet") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        stamp_dir = self._stamp_dir(cube_id)
        if stamp_dir is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}
        if space not in ("physical", "normalized"):
            return {"ok": False, "error": f"unknown space: {space}"}

        az0 = int(np.clip(az0, 0, meta["n_az"] - 1))
        az1 = int(np.clip(az1, 0, meta["n_az"] - 1))
        rg0 = int(np.clip(rg0, 0, meta["n_rg"] - 1))
        rg1 = int(np.clip(rg1, 0, meta["n_rg"] - 1))

        run_dir = stamp_dir.parent.parent
        rel     = Path("figures") / "cube_transects" / f"az{az0:04d}_rg{rg0:04d}_to_az{az1:04d}_rg{rg1:04d}"
        out_dir = run_dir / rel

        files = []
        for source in meta["sources"]:
            entry = entries[source]
            data, heights, vmin, vmax = self._transect_cut(entry, az0, rg0, az1, rg1, space)
            saved = self.archiver.render_transect(data, heights, vmin, vmax, source, (az0, rg0), (az1, rg1), space, out_dir / f"transect_{source}_{space}.png", cmap=self._entry_cmap(entry, cmap))
            files.append(saved.name)

        self.logger.ok(f"saved {len(files)} transect figures to {out_dir}")
        return {"ok": True, "dir": str(out_dir), "rel": str(rel), "files": files}

    @classmethod
    def _transect_cut(cls, entry: dict, az0: int, rg0: int, az1: int, rg1: int, space: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        cube               = entry["cube"]
        n_elev, n_az, n_rg = cube.shape

        az0 = int(np.clip(az0, 0, n_az - 1))
        az1 = int(np.clip(az1, 0, n_az - 1))
        rg0 = int(np.clip(rg0, 0, n_rg - 1))
        rg1 = int(np.clip(rg1, 0, n_rg - 1))

        samples = int(max(abs(az1 - az0), abs(rg1 - rg0))) + 1
        az_idx  = np.round(np.linspace(az0, az1, samples)).astype(int)
        rg_idx  = np.round(np.linspace(rg0, rg1, samples)).astype(int)
        data    = cube[:, az_idx, rg_idx]

        data, vmin, vmax = cls._normalize_cut(entry, data, space)

        order   = np.argsort(entry["x_axis"])
        heights = np.asarray(entry["x_axis"], dtype=np.float64)[order]
        return data[order], heights, float(vmin), float(vmax)

    def save_slices(self, cube_id: str, az: int, rg: int, space: str = "physical", cmap: str = "jet") -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["id"] != cube_id:
                return {"ok": False, "error": "cube not loaded"}
            entries = self.loaded["entries"]
            meta    = self.loaded["meta"]

        stamp_dir = self._stamp_dir(cube_id)
        if stamp_dir is None:
            return {"ok": False, "error": f"unknown cube id: {cube_id}"}
        if space not in ("physical", "normalized"):
            return {"ok": False, "error": f"unknown space: {space}"}

        az = int(np.clip(az, 0, meta["n_az"] - 1))
        rg = int(np.clip(rg, 0, meta["n_rg"] - 1))

        run_dir = stamp_dir.parent.parent
        rel     = Path("figures") / "cube_slices" / f"az{az:04d}_rg{rg:04d}"
        out_dir = run_dir / rel

        files = []
        for source in meta["sources"]:
            for axis in ("range", "azimuth"):
                data, heights, vmin, vmax = self._cut(entries[source], axis, az, rg, space)
                saved = self.archiver.render(data, heights, vmin, vmax, source, axis, az, rg, space, out_dir / f"{axis}_{source}_{space}.png", cmap=self._entry_cmap(entries[source], cmap))
                files.append(saved.name)

        self.logger.ok(f"saved {len(files)} slice figures to {out_dir}")
        return {"ok": True, "dir": str(out_dir), "rel": str(rel), "az": az, "rg": rg, "files": files}

    @classmethod
    def _cut(cls, entry: dict, axis: str, az: int, rg: int, space: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        cube = entry["cube"]
        data = cube[:, :, rg] if axis == "range" else cube[:, az, :]

        data, vmin, vmax = cls._normalize_cut(entry, data, space)

        order   = np.argsort(entry["x_axis"])
        heights = np.asarray(entry["x_axis"], dtype=np.float64)[order]
        return data[order], heights, float(vmin), float(vmax)

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

    def _stamp_dir(self, cube_id: str) -> Path | None:
        if not cube_id:
            return None

        stamp_dir = Path(cube_id).resolve()
        if not self.roots.contains(stamp_dir):
            return None
        if not (stamp_dir / "cubes" / "pred_curves.npy").is_file():
            return None
        return stamp_dir

    def _load_worker(self, cube_id: str, stamp_dir: Path) -> None:
        try:
            entries, meta, primary, params, metric_maps, dem = self._load_all(stamp_dir)

            with self.lock:
                self.loaded = {"id": cube_id, "entries": entries, "meta": meta, "primary": primary, "params": params, "metric_maps": metric_maps, "dem": dem}
                self.status = {"state": "ready", "id": cube_id, "progress": 1.0, "stage": "ready", "error": ""}

            self.logger.muted(f"cube ready: {cube_id} sources={meta['sources']}")
        except Exception as exc:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "id": cube_id, "progress": 0.0, "stage": "", "error": str(exc)}

            self.logger.error(f"cube load failed: {cube_id}: {exc}")

    def _load_all(self, stamp_dir: Path) -> tuple[dict, dict, np.ndarray, dict, dict, np.ndarray | None]:
        cubes_dir = stamp_dir / "cubes"
        pred_raw  = np.load(cubes_dir / "pred_curves.npy", mmap_mode="r")
        if pred_raw.ndim != 3:
            raise ValueError(f"pred_curves.npy is not a 3D cube: shape={pred_raw.shape}")

        n_elev, n_az, n_rg = pred_raw.shape
        curve_axis         = self._curve_axis(stamp_dir, n_elev)

        plan = [("pred", pred_raw, curve_axis)]
        for source in ("gt", "reduced"):
            path = cubes_dir / f"{source}_curves.npy"
            if path.is_file():
                plan.append((source, np.load(path, mmap_mode="r"), curve_axis))

        full_raw = self._full_raw(stamp_dir, n_az, n_rg)
        if full_raw is not None:
            plan.append(("full", full_raw, np.arange(full_raw.shape[0], dtype=np.float64)))

        total = sum(raw.shape[0] for _, raw, _ in plan)
        done  = [0]

        def advance(source: str) -> None:
            done[0] += 1
            with self.lock:
                self.status["progress"] = done[0] / total
                self.status["stage"]    = source

        entries = {}
        for source, raw, x_axis in plan:
            if raw.shape[1:] != (n_az, n_rg):
                raise ValueError(f"source '{source}' spatial shape {raw.shape[1:]} does not match pred {(n_az, n_rg)}")
            entries[source] = self._ingest(raw, x_axis, lambda s=source: advance(s))

        primary     = self._primary_db(stamp_dir, n_az, n_rg)
        params      = self._load_params(cubes_dir, n_az, n_rg)
        metric_maps = self._load_metric_maps(cubes_dir, n_az, n_rg)
        dem         = self._load_dem(stamp_dir, n_az, n_rg)

        meta = {
            "sources"     : [s for s in self.SOURCES if s in entries],
            "n_az"        : n_az,
            "n_rg"        : n_rg,
            "n_elev"      : {s: int(entries[s]["cube"].shape[0]) for s in entries},
            "x_min"       : float(curve_axis[0]),
            "x_max"       : float(curve_axis[-1]),
            "intensity"   : {s: [entries[s]["vmin"], entries[s]["vmax"]] for s in entries},
            "params"      : self._params_meta(params, curve_axis),
            "metric_maps" : self._metric_maps_meta(metric_maps),
            "attached"    : None,
            "dem"         : dem is not None,
            "spacing"     : self._load_spacing(stamp_dir),
        }
        return entries, meta, primary, params, metric_maps, dem

    def _load_spacing(self, stamp_dir: Path) -> dict | None:
        resolved = self._preproc_layout(stamp_dir)
        if resolved is None:
            return None

        preproc_dir, _ = resolved

        params_path = preproc_dir / "meta" / TrackParameters.FILENAME
        if not params_path.is_file():
            return None

        reference = TrackParameters.load(params_path).parameters[0]
        return {"az": float(reference["ps_az"]), "rg": float(reference["ps_rg"])}

    def _load_dem(self, stamp_dir: Path, n_az: int, n_rg: int) -> np.ndarray | None:
        resolved = self._preproc_layout(stamp_dir)
        if resolved is None:
            return None

        preproc_dir, layout = resolved

        dem_name = layout["artifacts"].get("dem_full")
        if not dem_name:
            return None

        dem_path = preproc_dir / "data" / dem_name
        if not dem_path.is_file():
            return None

        az_lo, az_hi, rg_lo, rg_hi = self._crop_bounds(stamp_dir, layout, n_az, n_rg)

        raw = np.load(dem_path, mmap_mode="r")
        if raw.ndim != 2 or az_hi > raw.shape[0] or rg_hi > raw.shape[1] or az_lo < 0 or rg_lo < 0:
            raise ValueError(f"dem_full shape {raw.shape} does not cover the cube region az[{az_lo}:{az_hi}] rg[{rg_lo}:{rg_hi}]")

        return np.asarray(raw[az_lo:az_hi, rg_lo:rg_hi], dtype=np.float32)

    def _load_metric_maps(self, cubes_dir: Path, n_az: int, n_rg: int) -> dict:
        maps = {}
        for path in sorted(cubes_dir.glob("*.npy")):
            if any(marker in path.name for marker in self.METRIC_EXCLUDED):
                continue

            raw = np.load(path, mmap_mode="r")
            if raw.ndim != 2 or raw.shape != (n_az, n_rg):
                continue

            maps[path.stem] = np.asarray(raw, dtype=np.float32)

        return maps

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

    def _load_params(self, cubes_dir: Path, n_az: int, n_rg: int) -> dict:
        params = {}
        for source in self.PARAM_SOURCES:
            path = cubes_dir / f"params_{source}.npy"
            if not path.is_file():
                continue

            raw = np.asarray(np.load(path), dtype=np.float32)
            if raw.ndim != 3 or raw.shape[0] % 3 != 0 or raw.shape[0] == 0:
                raise ValueError(f"params_{source}.npy is not a (3K, az, rg) cube: shape={raw.shape}")
            if raw.shape[1:] != (n_az, n_rg):
                raise ValueError(f"params_{source}.npy spatial shape {raw.shape[1:]} does not match the {n_az}x{n_rg} cube")

            params[source] = raw

        return params

    def _params_meta(self, params: dict, curve_axis: np.ndarray) -> dict | None:
        if not params:
            return None

        n_slots   = next(iter(params.values())).shape[0] // 3
        threshold = ParamMatcher.ACTIVE_AMP_THR
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

    @staticmethod
    def _field_ceiling(params: dict, offset: int) -> float:
        high = 0.0
        for block in params.values():
            channels = block[offset::3]
            if channels.size:
                high = max(high, float(np.nanpercentile(channels, 99.0)))
        return high if high > 0.0 else 1.0

    def _ingest(self, raw: np.ndarray, x_axis: np.ndarray, advance) -> dict:
        cube = np.empty(raw.shape, dtype=np.float32)

        for i in range(raw.shape[0]):
            plane   = np.asarray(raw[i])
            cube[i] = np.abs(plane) if np.iscomplexobj(plane) else plane
            advance()

        sample = cube[:, :: max(1, cube.shape[1] // 256), :: max(1, cube.shape[2] // 256)]
        sample = sample[np.isfinite(sample)]
        vmin, vmax = (np.percentile(sample, [1.0, 99.0]) if sample.size else (0.0, 1.0))

        return {
            "cube"   : cube,
            "x_axis" : x_axis,
            "vmin"   : float(vmin),
            "vmax"   : float(vmax),
        }

    def _metrics(self, stamp_dir: Path) -> dict:
        metrics_path = stamp_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"metrics.json missing in {stamp_dir}; rerun inference to regenerate it")

        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _curve_axis(self, stamp_dir: Path, n_elev: int) -> np.ndarray:
        metrics = self._metrics(stamp_dir)
        return np.linspace(float(metrics["x_axis_min"]), float(metrics["x_axis_max"]), n_elev)

    def _preproc_layout(self, stamp_dir: Path) -> tuple[Path, dict] | None:
        meta_path = stamp_dir.parent.parent / "meta" / "dataset_creation_config.json"
        if not meta_path.is_file():
            return None

        payload     = json.loads(meta_path.read_text(encoding="utf-8"))
        preproc_dir = Path(payload["preprocessing_run_directory"])
        layout_path = preproc_dir / "data" / "dataset.json"
        if not layout_path.is_file():
            return None

        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        return preproc_dir, layout

    def _crop_bounds(self, stamp_dir: Path, layout: dict, n_az: int, n_rg: int) -> tuple[int, int, int, int]:
        metrics = self._metrics(stamp_dir)
        if "split_region" not in metrics:
            raise KeyError(f"metrics.json in {stamp_dir} has no split_region; rerun inference to regenerate it")

        az_start, az_end, rg_start, rg_end = (int(v) for v in metrics["split_region"])
        if (az_end - az_start, rg_end - rg_start) != (n_az, n_rg):
            raise ValueError(f"split_region {metrics['split_region']} does not match the {n_az}x{n_rg} cube")

        global_crop = layout["global_crop"]

        az_lo = az_start - global_crop[0]
        az_hi = az_end   - global_crop[0]
        rg_lo = rg_start - global_crop[2]
        rg_hi = rg_end   - global_crop[2]
        return az_lo, az_hi, rg_lo, rg_hi

    def _full_raw(self, stamp_dir: Path, n_az: int, n_rg: int) -> np.ndarray | None:
        resolved = self._preproc_layout(stamp_dir)
        if resolved is None:
            return None

        preproc_dir, layout = resolved

        tomo_name = layout["artifacts"].get("tomogram_full")
        if not tomo_name:
            return None

        tomo_path = preproc_dir / "data" / tomo_name
        if not tomo_path.is_file():
            return None

        az_lo, az_hi, rg_lo, rg_hi = self._crop_bounds(stamp_dir, layout, n_az, n_rg)

        raw = np.load(tomo_path, mmap_mode="r")
        if raw.ndim != 3:
            raise ValueError(f"full tomogram is not a 3D cube: shape={raw.shape}")
        if az_lo < 0 or rg_lo < 0 or az_hi > raw.shape[1] or rg_hi > raw.shape[2]:
            raise ValueError(f"cube region az[{az_lo}:{az_hi}] rg[{rg_lo}:{rg_hi}] falls outside the full tomogram {raw.shape}")

        return raw[:, az_lo:az_hi, rg_lo:rg_hi]

    def _primary_db(self, stamp_dir: Path, n_az: int, n_rg: int) -> np.ndarray:
        resolved = self._preproc_layout(stamp_dir)
        if resolved is None:
            raise FileNotFoundError(f"cannot resolve the preprocessing run for {stamp_dir}; the primary SLC is required for the cube map")

        preproc_dir, layout = resolved

        primary_name = layout["artifacts"].get("primary")
        if not primary_name:
            raise FileNotFoundError(f"dataset.json in {preproc_dir} lists no primary artifact")

        primary_path = preproc_dir / "data" / primary_name
        if not primary_path.is_file():
            raise FileNotFoundError(f"primary SLC missing: {primary_path}")

        az_lo, az_hi, rg_lo, rg_hi = self._crop_bounds(stamp_dir, layout, n_az, n_rg)

        raw = np.load(primary_path, mmap_mode="r")
        if raw.ndim != 2:
            raise ValueError(f"primary SLC is not a 2D image: shape={raw.shape}")
        if az_lo < 0 or rg_lo < 0 or az_hi > raw.shape[0] or rg_hi > raw.shape[1]:
            raise ValueError(f"cube region az[{az_lo}:{az_hi}] rg[{rg_lo}:{rg_hi}] falls outside the primary SLC {raw.shape}")

        amplitude = np.abs(np.asarray(raw[az_lo:az_hi, rg_lo:rg_hi])).astype(np.float32)
        return 20.0 * np.log10(np.maximum(amplitude, 1e-12))


class SliceCollector:

    SOURCES    = ("pred", "gt", "reduced", "full")
    AXES       = ("range", "azimuth")
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
            "n_az"      : context["n_az"],
            "n_rg"      : context["n_rg"],
            "sources"   : [s for s in self.SOURCES if s in entries],
            "intensity" : {s: [entries[s]["vmin"], entries[s]["vmax"]] for s in entries},
        }

    def slice_png(self, cube_id: str, source: str, axis: str, az: int, rg: int, space: str = "physical", cmap: str = "jet", vmin: float | None = None, vmax: float | None = None) -> bytes | None:
        if source not in self.SOURCES or axis not in self.AXES or space not in ("physical", "normalized"):
            return None

        try:
            context = self._context(cube_id)
        except Exception:
            return None

        entry = None if context is None else context["entries"].get(source)
        if entry is None:
            return None

        data, heights, lo, hi = self._cut(entry, axis, az, rg, space)
        if space == "physical" and vmin is not None and vmax is not None and vmax > vmin:
            lo, hi = float(vmin), float(vmax)

        buf = io.BytesIO()
        plt.imsave(buf, np.flipud(data), cmap=CubeExplorer._entry_cmap(entry, cmap), vmin=lo, vmax=hi, format="png")
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
            for az, rg in points:
                point_dir = out_dir / f"az{az:04d}_rg{rg:04d}"
                for source in sources:
                    clim = clims[source]
                    for axis in axes:
                        cut_dir = point_dir / f"{axis}_{source}_{space}"
                        for context, label in zip(contexts, labels):
                            entry = context["entries"].get(source)
                            if entry is None:
                                continue

                            az_i = int(np.clip(az, 0, context["n_az"] - 1))
                            rg_i = int(np.clip(rg, 0, context["n_rg"] - 1))
                            data, heights, vmin, vmax = self._cut(entry, axis, az_i, rg_i, space)
                            if clim is not None:
                                vmin, vmax = clim

                            saved = self.cubes.archiver.render(data, heights, vmin, vmax, source, axis, az_i, rg_i, space, cut_dir / f"{label}.png", cmap=CubeExplorer._entry_cmap(entry, cmap), label=label)
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
            "points"  : [{"az": az, "rg": rg} for az, rg in points],
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
            raise ValueError("points must be a non-empty list of {az, rg}")
        if len(raw) > self.MAX_POINTS:
            raise ValueError(f"at most {self.MAX_POINTS} points per collection, got {len(raw)}")

        points = []
        seen   = set()
        for entry in raw:
            az, rg = int(entry["az"]), int(entry["rg"])
            if az < 0 or rg < 0:
                raise ValueError(f"point az={az}, rg={rg} is negative")
            if (az, rg) in seen:
                continue
            seen.add((az, rg))
            points.append((az, rg))

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
    def _cut(entry: dict, axis: str, az: int, rg: int, space: str) -> tuple[np.ndarray, np.ndarray, float, float]:
        cube               = entry["cube"]
        n_elev, n_az, n_rg = cube.shape

        az   = int(np.clip(az, 0, n_az - 1))
        rg   = int(np.clip(rg, 0, n_rg - 1))
        data = np.asarray(cube[:, :, rg] if axis == "range" else cube[:, az, :])
        data = np.abs(data).astype(np.float32) if np.iscomplexobj(data) else data.astype(np.float32)

        data, vmin, vmax = CubeExplorer._normalize_cut(entry, data, space)

        order   = np.argsort(entry["x_axis"])
        heights = np.asarray(entry["x_axis"], dtype=np.float64)[order]
        return data[order], heights, float(vmin), float(vmax)

    def _context(self, cube_id: str) -> dict | None:
        stamp_dir = self.cubes._stamp_dir(cube_id)
        if stamp_dir is None:
            return None

        key = str(stamp_dir)
        with self.lock:
            if key in self.contexts:
                self.contexts.move_to_end(key)
                return self.contexts[key]

        context = self._build_context(stamp_dir)

        with self.lock:
            self.contexts[key] = context
            while len(self.contexts) > self.CACHE_CAP:
                self.contexts.popitem(last=False)

        return context

    def _build_context(self, stamp_dir: Path) -> dict:
        root = self._root_for(stamp_dir)
        if root is None:
            raise ValueError(f"{stamp_dir} is outside every catalogued runs root")

        cubes_dir = stamp_dir / "cubes"
        pred_raw  = np.load(cubes_dir / "pred_curves.npy", mmap_mode="r")
        if pred_raw.ndim != 3:
            raise ValueError(f"pred_curves.npy is not a 3D cube: shape={pred_raw.shape}")

        n_elev, n_az, n_rg = pred_raw.shape
        curve_axis         = self.cubes._curve_axis(stamp_dir, n_elev)

        entries = {"pred": self._entry(pred_raw, curve_axis)}
        for source in ("gt", "reduced"):
            path = cubes_dir / f"{source}_curves.npy"
            if path.is_file():
                raw = np.load(path, mmap_mode="r")
                if raw.shape[1:] != (n_az, n_rg):
                    raise ValueError(f"source '{source}' spatial shape {raw.shape[1:]} does not match pred {(n_az, n_rg)}")
                entries[source] = self._entry(raw, curve_axis)

        full_raw = self.cubes._full_raw(stamp_dir, n_az, n_rg)
        if full_raw is not None:
            entries["full"] = self._entry(full_raw, np.arange(full_raw.shape[0], dtype=np.float64))

        run_dir = stamp_dir.parent.parent
        return {
            "stamp_dir" : stamp_dir,
            "root"      : root,
            "run"       : run_dir.name,
            "group"     : str(run_dir.relative_to(root).parent),
            "stamp"     : stamp_dir.name,
            "n_az"      : n_az,
            "n_rg"      : n_rg,
            "entries"   : entries,
        }

    @staticmethod
    def _entry(raw: np.ndarray, x_axis: np.ndarray) -> dict:
        sample = np.asarray(raw[:, :: max(1, raw.shape[1] // 256), :: max(1, raw.shape[2] // 256)])
        sample = np.abs(sample) if np.iscomplexobj(sample) else sample
        sample = sample[np.isfinite(sample)]
        vmin, vmax = (np.percentile(sample, [1.0, 99.0]) if sample.size else (0.0, 1.0))

        return {
            "cube"   : raw,
            "x_axis" : x_axis,
            "vmin"   : float(vmin),
            "vmax"   : float(vmax),
        }

    def _root_for(self, stamp_dir: Path) -> Path | None:
        return self.cubes.roots.enclosing(stamp_dir)

