"""Pairwise autopsy of two inference stamps for the web UI A/B tab.

Compares scalar metrics with their known orientation, locates the spatial blocks
where one run's per-pixel MSE beats the other, and serves single-pixel z
profiles from both runs against the shared ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from catalog_roots         import CatalogRoots, RunScanner
from tools.metrics.scoring import MetricOrientation
from web_logger            import WebLogger


class AbAutopsy:
    """Serves the metric deltas, error hotspots and profile cuts between two runs.

    Attributes:
        logger  : Web logger used to report comparison sizes.
        scanner : Scanner listing inference stamps under the catalogued runs roots.
    """

    BLOCK        = 16
    MAX_METRICS  = 40
    MAX_HOTSPOTS = 3
    SKIP_PREFIX  = ("tracks", "track_positions", "split", "z_axis")
    REQUIRED     = ("metrics.json", "cubes/pixel_mse.npy")

    def __init__(self, logger: WebLogger) -> None:
        """Builds the autopsy over a fresh runs-root catalogue.

        Args:
            logger : Web logger for progress messages.
        """
        self.logger  = logger
        self.scanner = RunScanner(CatalogRoots())

    def runs(self, base: str) -> dict:
        """Lists the inference stamps under a runs root that carry the needed files.

        Args:
            base : Absolute path of the runs directory to scan.

        Returns:
            Dictionary with "ok"; on success also "root" and "runs" (one entry per
            comparable stamp), on failure "error" and an empty "runs" list.
        """
        scanned = self.scanner.stamps(base, self.REQUIRED)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "runs": []}

        return {"ok": True, "root": scanned["root"], "runs": scanned["entries"]}

    def _stamp(self, raw: str) -> Path:
        """Resolves a stamp path and rejects anything outside the catalogued roots.

        Args:
            raw : Stamp directory path as sent by the client.

        Returns:
            The resolved stamp directory.

        Raises:
            ValueError : If the path lies outside every known runs root.
        """
        stamp = Path(raw).resolve()
        if not self.scanner.roots.contains(stamp):
            raise ValueError(f"{stamp} is outside every catalogued runs root; scan the runs directory first")
        return stamp

    def _metrics(self, stamp: Path) -> dict:
        """Loads the metrics.json payload of an inference stamp.

        Args:
            stamp : Stamp directory holding metrics.json.

        Returns:
            Parsed metrics dictionary.

        Raises:
            FileNotFoundError : If the stamp carries no metrics.json.
        """
        path = stamp / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"no metrics.json under {stamp}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _cube(self, stamp: Path, name: str) -> np.ndarray:
        """Memory-maps a saved cube from an inference stamp.

        Args:
            stamp : Stamp directory holding the cubes subdirectory.
            name  : Cube basename without the .npy suffix, e.g. "pixel_mse".

        Returns:
            The cube as a read-only memory map; profile cubes have shape
            (n_samples, n_rows, n_cols) and pixel_mse has shape
            (n_rows, n_cols).

        Raises:
            FileNotFoundError : If the cube was not saved by the inference run.
        """
        path = stamp / "cubes" / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; re-run inference with save_cubes")
        return np.load(path, mmap_mode="r")

    def _metric_rows(self, metrics_a: dict, metrics_b: dict) -> list[dict]:
        """Builds the per-metric delta rows for the metrics both runs share.

        Keys are kept only when both values are finite non-boolean numbers and the
        metric has a known better-is-higher or better-is-lower orientation. Rows are
        ranked by the magnitude of the relative delta and truncated.

        Args:
            metrics_a : Metrics payload of run A.
            metrics_b : Metrics payload of run B.

        Returns:
            Up to MAX_METRICS rows, each with the metric key, both values, the
            absolute and relative delta of A against B, and the winner.
        """
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
        """Finds the BLOCK-sized tiles where one run's pixel MSE clearly beats the other.

        Tiles with fewer than a quarter of their pixels finite are dropped; the rest
        are ranked by mean MSE difference in each direction.

        Args:
            mse_a : Per-pixel MSE of run A, shape (n_rows, n_cols).
            mse_b : Per-pixel MSE of run B, same shape.

        Returns:
            Up to MAX_HOTSPOTS tiles favouring A followed by up to MAX_HOTSPOTS
            favouring B, each with the tile origin, its mean delta, the row and
            range index of the largest absolute difference, and the winner.
        """
        delta      = np.asarray(mse_a, dtype=np.float64) - np.asarray(mse_b, dtype=np.float64)
        n_rows, n_cols = delta.shape

        blocks = []
        for row0 in range(0, n_rows, self.BLOCK):
            for col0 in range(0, n_cols, self.BLOCK):
                patch  = delta[row0:row0 + self.BLOCK, col0:col0 + self.BLOCK]
                finite = np.isfinite(patch)
                if finite.sum() < max(4, patch.size // 4):
                    continue

                mean_delta = float(patch[finite].mean())
                peak_flat  = np.nanargmax(np.where(finite, np.abs(patch), -np.inf))

                blocks.append({
                    "row0"       : row0,
                    "col0"       : col0,
                    "mean_delta" : mean_delta,
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
        """Compares two inference stamps by scalar metrics and per-pixel error tiles.

        Args:
            a : Stamp directory of run A.
            b : Stamp directory of run B.

        Returns:
            Dictionary with "ok"; on success also the two stamp paths, both run
            names, the region shape, the metric delta rows and the hotspot tiles.
            On failure, or when the two runs cover different regions, "error".
        """
        try:
            stamp_a, stamp_b = self._stamp(a), self._stamp(b)

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

    def profile(self, a: str, b: str, row: int, col: int) -> dict:
        """Returns both runs' predicted z profiles at one pixel, with the truth.

        The ground truth is read from run A, so the request is refused unless both
        runs share the cube shape and the z axis bounds.

        Args:
            a  : Stamp directory of run A, which also supplies the ground truth.
            b  : Stamp directory of run B.
            row : Row index of the pixel.
            col : Range index of the pixel.

        Returns:
            Dictionary with "ok"; on success also the pixel indices, the z
            axis of length n_samples in metres, and the A, B and ground-truth
            profiles sampled on it. On failure, "error".
        """
        try:
            stamp_a, stamp_b = self._stamp(a), self._stamp(b)

            pred_a = self._cube(stamp_a, "pred_curves")
            pred_b = self._cube(stamp_b, "pred_curves")
            gt     = self._cube(stamp_a, "gt_curves")

            if pred_b.shape != pred_a.shape:
                return {"ok": False, "error": f"runs cover different cubes: {pred_a.shape} vs {pred_b.shape}"}
            if not (0 <= row < pred_a.shape[1] and 0 <= col < pred_a.shape[2]):
                return {"ok": False, "error": f"pixel ({row}, {col}) is outside the region {pred_a.shape[1:]}"}

            metrics   = self._metrics(stamp_a)
            metrics_b = self._metrics(stamp_b)
            bounds    = (float(metrics["x_axis_min"]), float(metrics["x_axis_max"]))
            bounds_b  = (float(metrics_b["x_axis_min"]), float(metrics_b["x_axis_max"]))

            if bounds_b != bounds:
                return {"ok": False, "error": f"runs cover different z axes: {bounds} vs {bounds_b}"}

            z_axis = np.linspace(bounds[0], bounds[1], pred_a.shape[0])

            return {
                "ok"     : True,
                "row"    : row,
                "col"    : col,
                "z_axis" : [float(v) for v in z_axis],
                "a"      : [float(v) for v in pred_a[:, row, col]],
                "b"      : [float(v) for v in pred_b[:, row, col]],
                "gt"     : [float(v) for v in gt[:, row, col]],
            }
        except (OSError, ValueError, FileNotFoundError, IndexError, KeyError) as error:
            return {"ok": False, "error": str(error)}
