from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from project_paths import ProjectPaths
from web_logger    import WebLogger


class TriageStore:

    VERDICTS = ("", "label problem", "model problem", "interesting")

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)
        self.lock      = threading.Lock()

    def _path(self, cube_id: str) -> Path:
        digest = hashlib.sha1(cube_id.encode("utf-8")).hexdigest()[:16]
        return self.store_dir / f"{digest}.json"

    def load(self, cube_id: str) -> dict:
        path = self._path(cube_id)
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8")).get("cases", {})

    def annotate(self, cube_id: str, case_id: str, verdict: str, note: str) -> dict:
        if verdict not in self.VERDICTS:
            return {"ok": False, "error": f"unknown verdict '{verdict}', expected one of {[v or '(clear)' for v in self.VERDICTS]}"}

        with self.lock:
            path  = self._path(cube_id)
            cases = self.load(cube_id)

            if verdict == "" and not note:
                cases.pop(case_id, None)
            else:
                cases[case_id] = {"verdict": verdict, "note": note, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}

            self.store_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"cube": cube_id, "cases": cases}, indent=2), encoding="utf-8")

        return {"ok": True, "case": case_id, "annotation": cases.get(case_id)}


class TriageBoard:

    BLOCK      = 16
    AUX_LAYERS = ("label_r2", "seed_std_profile", "flip_consistency")

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.logger = logger
        self.store  = TriageStore(paths.logs_dir / "triage")

    @property
    def modes(self) -> tuple:
        from pipelines.backbone.inference.failure_modes import FailureModes

        return FailureModes.MODES

    def _load_map(self, cubes_dir: Path, name: str) -> np.ndarray | None:
        path = cubes_dir / f"{name}.npy"
        if not path.is_file():
            return None

        data = np.load(path, mmap_mode="r")
        return np.asarray(data, dtype=np.float64) if data.ndim == 2 else None

    def _block_rows(self, error_map: np.ndarray, mode_map: np.ndarray | None, aux: dict[str, np.ndarray]) -> list[dict]:
        n_rows, n_cols = error_map.shape
        rows       = []

        for row0 in range(0, n_rows, self.BLOCK):
            for col0 in range(0, n_cols, self.BLOCK):
                block = error_map[row0:row0 + self.BLOCK, col0:col0 + self.BLOCK]
                valid = np.isfinite(block)
                if valid.sum() < max(4, block.size // 4):
                    continue

                worst_flat = np.nanargmax(np.where(valid, block, -np.inf))
                worst_row  = row0 + int(worst_flat // block.shape[1])
                worst_col  = col0 + int(worst_flat % block.shape[1])

                row = {
                    "case"      : f"{row0}_{col0}",
                    "row0"      : row0,
                    "col0"      : col0,
                    "row1"      : min(row0 + self.BLOCK, n_rows),
                    "col1"      : min(col0 + self.BLOCK, n_cols),
                    "mse_mean"  : float(block[valid].mean()),
                    "mse_max"   : float(block[valid].max()),
                    "worst_row" : worst_row,
                    "worst_col" : worst_col,
                }

                if mode_map is not None:
                    modes            = mode_map[row0:row0 + self.BLOCK, col0:col0 + self.BLOCK].astype(np.int64)
                    failing          = modes[modes > 0]
                    dominant         = int(np.bincount(failing).argmax()) if failing.size else 0
                    mode_names       = self.modes
                    row["mode"]      = mode_names[dominant] if dominant < len(mode_names) else str(dominant)
                    row["fail_frac"] = float((modes > 0).mean())

                for name, layer in aux.items():
                    patch  = layer[row0:row0 + self.BLOCK, col0:col0 + self.BLOCK]
                    finite = patch[np.isfinite(patch)]
                    row[name] = float(finite.mean()) if finite.size else None

                rows.append(row)

        rows.sort(key=lambda row: row["mse_mean"], reverse=True)
        return rows

    def cases(self, cube_id: str, top_n: int = 40) -> dict:
        stamp_dir = Path(cube_id)
        cubes_dir = stamp_dir / "cubes"

        if not cubes_dir.is_dir():
            return {"ok": False, "error": f"no cubes directory under {cube_id}"}

        error_map = self._load_map(cubes_dir, "pixel_mse")
        if error_map is None:
            return {"ok": False, "error": f"{cubes_dir / 'pixel_mse.npy'} is missing; run inference with save_cubes first"}

        mode_map = self._load_map(cubes_dir, "failure_mode")
        aux      = {name: layer for name in self.AUX_LAYERS if (layer := self._load_map(cubes_dir, name)) is not None}

        rows        = self._block_rows(error_map, mode_map, aux)[: max(1, int(top_n))]
        annotations = self.store.load(cube_id)

        for row in rows:
            row["annotation"] = annotations.get(row["case"])

        self.logger.info(f"triage: {len(rows)} cases for {cube_id} (modes: {mode_map is not None}, aux: {sorted(aux)})")

        return {
            "ok"        : True,
            "id"        : cube_id,
            "run"       : stamp_dir.parent.parent.name,
            "block"     : self.BLOCK,
            "has_modes" : mode_map is not None,
            "aux"       : sorted(aux),
            "cases"     : rows,
        }

    def thumb(self, cube_id: str, row0: int, col0: int) -> bytes | None:
        error_map = self._load_map(Path(cube_id) / "cubes", "pixel_mse")
        if error_map is None:
            return None

        n_rows, n_cols = error_map.shape
        if not (0 <= row0 < n_rows and 0 <= col0 < n_cols):
            return None

        row_lo, row_hi = max(0, row0 - self.BLOCK), min(n_rows, row0 + 2 * self.BLOCK)
        col_lo, col_hi = max(0, col0 - self.BLOCK), min(n_cols, col0 + 2 * self.BLOCK)
        crop         = error_map[row_lo:row_hi, col_lo:col_hi]

        finite = np.isfinite(crop) & (crop > 0)
        logs   = np.log10(crop, out=np.full_like(crop, np.nan), where=finite)
        lo, hi = (np.nanpercentile(logs, [1.0, 99.0]) if finite.any() else (0.0, 1.0))

        norm    = np.clip((logs - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        colored = plt.get_cmap("viridis")(np.nan_to_num(norm))[..., :3]
        colored[~finite] = (0.06, 0.08, 0.10)

        b_az_lo, b_az_hi = row0 - row_lo, min(row0 + self.BLOCK, n_rows) - row_lo
        b_rg_lo, b_rg_hi = col0 - col_lo, min(col0 + self.BLOCK, n_cols) - col_lo

        colored[b_az_lo, b_rg_lo:b_rg_hi]         = 1.0
        colored[b_az_hi - 1, b_rg_lo:b_rg_hi]     = 1.0
        colored[b_az_lo:b_az_hi, b_rg_lo]         = 1.0
        colored[b_az_lo:b_az_hi, b_rg_hi - 1]     = 1.0

        block = crop[b_az_lo:b_az_hi, b_rg_lo:b_rg_hi]
        valid = np.isfinite(block)
        if valid.any():
            worst = np.nanargmax(np.where(valid, block, -np.inf))
            w_row = b_az_lo + int(worst // block.shape[1])
            w_col = b_rg_lo + int(worst % block.shape[1])
            colored[max(0, w_row - 1):w_row + 2, max(0, w_col - 1):w_col + 2] = (0.85, 0.10, 0.10)

        buf = io.BytesIO()
        plt.imsave(buf, colored.astype(np.float32), format="png")
        return buf.getvalue()

    def profile(self, cube_id: str, row: int, col: int) -> dict:
        stamp_dir = Path(cube_id)
        cubes_dir = stamp_dir / "cubes"

        try:
            pred = np.load(cubes_dir / "pred_curves.npy", mmap_mode="r")
            gt   = np.load(cubes_dir / "gt_curves.npy", mmap_mode="r")

            if not (0 <= row < pred.shape[1] and 0 <= col < pred.shape[2]):
                return {"ok": False, "error": f"pixel ({row}, {col}) is outside the region {pred.shape[1:]}"}

            metrics = json.loads((stamp_dir / "metrics.json").read_text(encoding="utf-8"))
            z_axis  = np.linspace(float(metrics["x_axis_min"]), float(metrics["x_axis_max"]), pred.shape[0])

            return {
                "ok"     : True,
                "row"    : row,
                "col"    : col,
                "z_axis" : [float(v) for v in z_axis],
                "pred"   : [float(v) for v in pred[:, row, col]],
                "gt"     : [float(v) for v in gt[:, row, col]],
            }
        except (OSError, ValueError, KeyError) as error:
            return {"ok": False, "error": str(error)}

    def annotate(self, body: dict) -> dict:
        cube_id = str(body.get("id", ""))
        case_id = str(body.get("case", ""))

        if not cube_id or not case_id:
            return {"ok": False, "error": "id and case are required"}

        return self.store.annotate(cube_id, case_id, str(body.get("verdict", "")), str(body.get("note", "")))
