from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from catalog_roots         import CatalogRoots, RunScanner
from tools.metrics.scoring import MetricOrientation
from web_logger            import WebLogger


class AbAutopsy:

    BLOCK        = 16
    MAX_METRICS  = 40
    MAX_HOTSPOTS = 3
    SKIP_PREFIX  = ("tracks", "track_positions", "split", "x_axis")
    REQUIRED     = ("metrics.json", "cubes/pixel_mse.npy")

    def __init__(self, logger: WebLogger) -> None:
        self.logger  = logger
        self.scanner = RunScanner(CatalogRoots())

    def runs(self, base: str) -> dict:
        scanned = self.scanner.stamps(base, self.REQUIRED)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "runs": []}

        return {"ok": True, "root": scanned["root"], "runs": scanned["entries"]}

    def _metrics(self, stamp: Path) -> dict:
        path = stamp / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"no metrics.json under {stamp}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _cube(self, stamp: Path, name: str) -> np.ndarray:
        path = stamp / "cubes" / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; re-run inference with save_cubes")
        return np.load(path, mmap_mode="r")

    def _metric_rows(self, metrics_a: dict, metrics_b: dict) -> list[dict]:
        rows = []
        for key in sorted(set(metrics_a) & set(metrics_b)):
            if key.startswith(self.SKIP_PREFIX):
                continue

            value_a, value_b = metrics_a[key], metrics_b[key]
            if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
                continue
            if isinstance(value_a, bool) or not np.isfinite(value_a) or not np.isfinite(value_b):
                continue

            orientation = MetricOrientation.direction(key)
            if orientation is None:
                continue

            sign   = 1.0 if orientation == "higher" else -1.0
            delta  = float(value_a) - float(value_b)
            rel    = delta / max(abs(float(value_b)), 1e-12)
            winner = "tie" if delta == 0.0 else ("A" if sign * delta > 0 else "B")

            rows.append({
                "key"     : key,
                "a"       : float(value_a),
                "b"       : float(value_b),
                "delta"   : delta,
                "rel"     : rel,
                "winner"  : winner,
            })

        rows.sort(key=lambda row: abs(row["rel"]), reverse=True)
        return rows[: self.MAX_METRICS]

    def _hotspots(self, mse_a: np.ndarray, mse_b: np.ndarray) -> list[dict]:
        delta      = np.asarray(mse_a, dtype=np.float64) - np.asarray(mse_b, dtype=np.float64)
        n_az, n_rg = delta.shape

        blocks = []
        for az0 in range(0, n_az, self.BLOCK):
            for rg0 in range(0, n_rg, self.BLOCK):
                patch  = delta[az0:az0 + self.BLOCK, rg0:rg0 + self.BLOCK]
                finite = np.isfinite(patch)
                if finite.sum() < max(4, patch.size // 4):
                    continue

                mean_delta = float(patch[finite].mean())
                peak_flat  = np.nanargmax(np.where(finite, np.abs(patch), -np.inf))

                blocks.append({
                    "az0"        : az0,
                    "rg0"        : rg0,
                    "mean_delta" : mean_delta,
                    "az"         : az0 + int(peak_flat // patch.shape[1]),
                    "rg"         : rg0 + int(peak_flat % patch.shape[1]),
                })

        a_better = sorted((b for b in blocks if b["mean_delta"] < 0), key=lambda b: b["mean_delta"])[: self.MAX_HOTSPOTS]
        b_better = sorted((b for b in blocks if b["mean_delta"] > 0), key=lambda b: -b["mean_delta"])[: self.MAX_HOTSPOTS]

        for block in a_better:
            block["winner"] = "A"
        for block in b_better:
            block["winner"] = "B"

        return a_better + b_better

    def compare(self, a: str, b: str) -> dict:
        try:
            stamp_a, stamp_b = Path(a), Path(b)

            metrics_a = self._metrics(stamp_a)
            metrics_b = self._metrics(stamp_b)

            mse_a = self._cube(stamp_a, "pixel_mse")
            mse_b = self._cube(stamp_b, "pixel_mse")

            if mse_a.shape != mse_b.shape:
                return {"ok": False, "error": f"runs cover different regions: {mse_a.shape} vs {mse_b.shape}"}

            rows     = self._metric_rows(metrics_a, metrics_b)
            hotspots = self._hotspots(mse_a, mse_b)

            self.logger.info(f"autopsy: {len(rows)} metric deltas, {len(hotspots)} hotspots for {stamp_a.parent.parent.name} vs {stamp_b.parent.parent.name}")

            return {
                "ok"       : True,
                "a"        : a,
                "b"        : b,
                "run_a"    : stamp_a.parent.parent.name,
                "run_b"    : stamp_b.parent.parent.name,
                "region"   : list(mse_a.shape),
                "metrics"  : rows,
                "hotspots" : hotspots,
            }
        except (OSError, ValueError, FileNotFoundError) as error:
            return {"ok": False, "error": str(error)}

    def profile(self, a: str, b: str, az: int, rg: int) -> dict:
        try:
            stamp_a, stamp_b = Path(a), Path(b)

            pred_a = self._cube(stamp_a, "pred_curves")
            pred_b = self._cube(stamp_b, "pred_curves")
            gt     = self._cube(stamp_a, "gt_curves")

            if pred_b.shape != pred_a.shape:
                return {"ok": False, "error": f"runs cover different cubes: {pred_a.shape} vs {pred_b.shape}"}
            if not (0 <= az < pred_a.shape[1] and 0 <= rg < pred_a.shape[2]):
                return {"ok": False, "error": f"pixel ({az}, {rg}) is outside the region {pred_a.shape[1:]}"}

            metrics   = self._metrics(stamp_a)
            metrics_b = self._metrics(stamp_b)
            bounds    = (float(metrics["x_axis_min"]), float(metrics["x_axis_max"]))
            bounds_b  = (float(metrics_b["x_axis_min"]), float(metrics_b["x_axis_max"]))

            if bounds_b != bounds:
                return {"ok": False, "error": f"runs cover different elevation axes: {bounds} vs {bounds_b}"}

            x_axis = np.linspace(bounds[0], bounds[1], pred_a.shape[0])

            return {
                "ok"     : True,
                "az"     : az,
                "rg"     : rg,
                "x_axis" : [float(v) for v in x_axis],
                "a"      : [float(v) for v in pred_a[:, az, rg]],
                "b"      : [float(v) for v in pred_b[:, az, rg]],
                "gt"     : [float(v) for v in gt[:, az, rg]],
            }
        except (OSError, ValueError, FileNotFoundError, IndexError, KeyError) as error:
            return {"ok": False, "error": str(error)}
