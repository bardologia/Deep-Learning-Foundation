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
    REQUIRED     = ("metrics.json", "maps/predictions.npy", "maps/targets.npy")

    def __init__(self, logger: WebLogger) -> None:
        self.logger  = logger
        self.scanner = RunScanner(CatalogRoots())

    def runs(self, base: str) -> dict:
        scanned = self.scanner.stamps(base, self.REQUIRED)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "runs": []}

        return {"ok": True, "root": scanned["root"], "runs": scanned["entries"]}

    def _stamp(self, raw: str) -> Path:
        stamp = Path(raw).resolve()
        if not self.scanner.roots.contains(stamp):
            raise ValueError(f"{stamp} is outside every catalogued runs root; scan the runs directory first")
        return stamp

    @staticmethod
    def _metrics(stamp: Path) -> dict:
        return json.loads((stamp / "metrics.json").read_text(encoding="utf-8"))

    @staticmethod
    def _map(stamp: Path, name: str) -> np.ndarray:
        return np.load(stamp / "maps" / f"{name}.npy", mmap_mode="r")

    @staticmethod
    def _channels(stamp: Path) -> list[str]:
        return list(json.loads((stamp.parent.parent / "meta" / "run_summary.json").read_text(encoding="utf-8"))["target_channels"])

    @staticmethod
    def _error_map(stamp: Path) -> np.ndarray:
        predictions = np.asarray(AbAutopsy._map(stamp, "predictions"), dtype=np.float64)
        targets     = np.asarray(AbAutopsy._map(stamp, "targets"), dtype=np.float64)
        scale       = targets.reshape(targets.shape[0], -1).std(axis=1)[:, None, None]

        return (((predictions - targets) / np.where(scale > 0.0, scale, 1.0)) ** 2).mean(axis=0)

    def _metric_rows(self, metrics_a: dict, metrics_b: dict) -> list[dict]:
        rows = []
        for key in sorted(set(metrics_a) & set(metrics_b)):
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

            rows.append({"key": key, "a": float(value_a), "b": float(value_b), "delta": delta, "rel": rel, "winner": winner})

        rows.sort(key=lambda row: abs(row["rel"]), reverse=True)
        return rows[: self.MAX_METRICS]

    def _hotspots(self, error_a: np.ndarray, error_b: np.ndarray) -> list[dict]:
        delta          = error_a - error_b
        n_rows, n_cols = delta.shape

        blocks = []
        for row0 in range(0, n_rows, self.BLOCK):
            for col0 in range(0, n_cols, self.BLOCK):
                patch  = delta[row0:row0 + self.BLOCK, col0:col0 + self.BLOCK]
                finite = np.isfinite(patch)
                if finite.sum() < max(4, patch.size // 4):
                    continue

                peak_flat = np.nanargmax(np.where(finite, np.abs(patch), -np.inf))

                blocks.append({
                    "row0"       : row0,
                    "col0"       : col0,
                    "mean_delta" : float(patch[finite].mean()),
                    "row"        : row0 + int(peak_flat // patch.shape[1]),
                    "col"        : col0 + int(peak_flat % patch.shape[1]),
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
            stamp_a, stamp_b = self._stamp(a), self._stamp(b)

            error_a = self._error_map(stamp_a)
            error_b = self._error_map(stamp_b)

            if error_a.shape != error_b.shape:
                return {"ok": False, "error": f"runs cover different regions: {error_a.shape} vs {error_b.shape}"}

            rows     = self._metric_rows(self._metrics(stamp_a), self._metrics(stamp_b))
            hotspots = self._hotspots(error_a, error_b)

            self.logger.info(f"autopsy: {len(rows)} metric deltas, {len(hotspots)} hotspots for {stamp_a.parent.parent.name} vs {stamp_b.parent.parent.name}")

            return {
                "ok"       : True,
                "a"        : a,
                "b"        : b,
                "run_a"    : stamp_a.parent.parent.name,
                "run_b"    : stamp_b.parent.parent.name,
                "region"   : list(error_a.shape),
                "metrics"  : rows,
                "hotspots" : hotspots,
            }
        except (OSError, ValueError) as error:
            return {"ok": False, "error": str(error)}

    def pixel(self, a: str, b: str, row: int, col: int) -> dict:
        try:
            stamp_a, stamp_b = self._stamp(a), self._stamp(b)

            pred_a  = self._map(stamp_a, "predictions")
            pred_b  = self._map(stamp_b, "predictions")
            targets = self._map(stamp_a, "targets")

            if pred_b.shape != pred_a.shape:
                return {"ok": False, "error": f"runs cover different maps: {pred_a.shape} vs {pred_b.shape}"}
            if not (0 <= row < pred_a.shape[1] and 0 <= col < pred_a.shape[2]):
                return {"ok": False, "error": f"pixel ({row}, {col}) is outside the region {pred_a.shape[1:]}"}

            return {
                "ok"       : True,
                "row"      : row,
                "col"      : col,
                "channels" : self._channels(stamp_a),
                "target"   : [float(v) for v in targets[:, row, col]],
                "a"        : [float(v) for v in pred_a[:, row, col]],
                "b"        : [float(v) for v in pred_b[:, row, col]],
            }
        except (OSError, ValueError, KeyError) as error:
            return {"ok": False, "error": str(error)}
