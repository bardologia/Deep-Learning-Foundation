from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configuration.param_extraction import ExtractionConfig, FitMode
from tools.data.preprocessing       import ProfilePreprocessor
from tools.reporting.plotting       import PlotBase
from web_logger                     import WebLogger


class FitLab:

    MAX_PIXELS = 24
    BATCH_ROWS = 32
    K_LIMIT    = 8
    RANGE_STEP = 64
    ADAM_B1    = 0.95
    ADAM_B2    = 0.999

    CONFIG_KEYS = ("k_max", "lambda_k", "mode", "threshold_factor", "truncation_index", "prominence_frac", "sigma_init_divisor", "activity_threshold", "adam_steps", "adam_lr")

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger
        self.lock   = threading.Lock()

        self.loaded     = None
        self.status     = {"state": "idle", "path": "", "progress": 0.0, "stage": "", "error": ""}
        self.fit_state  = {"state": "idle", "progress": 0.0, "stage": "", "error": ""}
        self.fit_result = None

        self._kernel      = None
        self._initialiser = None

    def datasets(self, base: str) -> dict:
        root = Path(base).expanduser()
        if not base or not root.is_dir():
            return {"ok": False, "error": f"not a directory: {base or '(empty)'}"}

        candidates = [root] + sorted(child for child in root.iterdir() if child.is_dir())
        found      = [{"name": c.name, "path": str(c)} for c in candidates if (c / "data" / "tomogram_full.npy").is_file()]

        return {"ok": True, "base": str(root), "datasets": found}

    def start_load(self, path: str) -> dict:
        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": "a dataset load is already running"}
            if self.fit_state["state"] == "fitting":
                return {"ok": False, "error": "a fit is running; wait for it to finish"}
            self.loaded     = None
            self.fit_result = None
            self.fit_state  = {"state": "idle", "progress": 0.0, "stage": "", "error": ""}
            self.status     = {"state": "loading", "path": path, "progress": 0.0, "stage": "opening", "error": ""}

        threading.Thread(target=self._load_worker, args=(path,), name="FitLabLoad", daemon=True).start()
        return {"ok": True}

    def _load_worker(self, path: str) -> None:
        try:
            cfg          = ExtractionConfig(processed_data_path=Path(path))
            tomo_path    = cfg.discover_tomogram_path()
            height_range = cfg.discover_height_range()

            self._set_load(0.05, "mapping tomogram")
            tomogram = np.load(str(tomo_path), mmap_mode="r", allow_pickle=False)
            if tomogram.ndim != 3:
                raise ValueError(f"tomogram_full.npy must be 3-D (H, Az, R), got shape {tomogram.shape}")
            H, Az, R = tomogram.shape

            self._set_load(0.1, "reading primary SLC")
            primary = np.abs(np.load(str(cfg.data_directory / "primary.npy"), allow_pickle=False)).astype(np.float32)

            peak = np.empty((Az, R), dtype=np.float32)
            for r_start in range(0, R, self.RANGE_STEP):
                r_end                 = min(r_start + self.RANGE_STEP, R)
                peak[:, r_start:r_end] = np.abs(np.array(tomogram[:, :, r_start:r_end])).max(axis=0)
                self._set_load(0.1 + 0.9 * r_end / R, f"scanning tomogram {r_end}/{R}")

            meta = {
                "path"         : path,
                "name"         : Path(path).name,
                "h"            : int(H),
                "az"           : int(Az),
                "rg"           : int(R),
                "height_range" : [float(height_range[0]), float(height_range[1])],
            }

            with self.lock:
                self.loaded = {"path": path, "tomogram": tomogram, "maps": {"slc": primary, "peak": peak}, "height_range": height_range, "meta": meta}
                self.status = {"state": "ready", "path": path, "progress": 1.0, "stage": "ready", "error": ""}
            self.logger.ok(f"fitlab loaded {path} (H={H}, Az={Az}, R={R})")

        except Exception as exc:
            with self.lock:
                self.status = {"state": "error", "path": path, "progress": 0.0, "stage": "", "error": str(exc)}
            self.logger.error(f"fitlab load failed for {path}: {exc}")

    def _set_load(self, progress: float, stage: str) -> None:
        with self.lock:
            self.status["progress"] = float(progress)
            self.status["stage"]    = stage

    def load_status(self) -> dict:
        with self.lock:
            out = dict(self.status)
            if self.status["state"] == "ready" and self.loaded is not None:
                out["meta"] = self.loaded["meta"]
        return out

    def map_png(self, src: str) -> bytes | None:
        with self.lock:
            if self.loaded is None or src not in self.loaded["maps"]:
                return None
            image = self.loaded["maps"][src]

        if src == "slc":
            vmin, vmax = PlotBase._amplitude_clim(image)
            cmap       = "gray"
        else:
            vmin, vmax = np.percentile(image, [1.0, 99.0])
            cmap       = "viridis"

        buf = io.BytesIO()
        plt.imsave(buf, image.T, cmap=cmap, vmin=float(vmin), vmax=float(vmax), format="png")
        return buf.getvalue()

    def start_fit(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no dataset loaded"}
            if self.fit_state["state"] == "fitting":
                return {"ok": False, "error": "a fit is already running"}
            meta = self.loaded["meta"]

        try:
            pixels = self._parse_pixels(body.get("pixels"), meta["az"], meta["rg"])
            config = self._parse_config(body.get("config"))
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

        with self.lock:
            self.fit_state = {"state": "fitting", "progress": 0.0, "stage": "starting", "error": ""}

        threading.Thread(target=self._fit_worker, args=(pixels, config), name="FitLabFit", daemon=True).start()
        return {"ok": True}

    def _parse_pixels(self, raw, az_size: int, rg_size: int) -> list:
        if not isinstance(raw, list) or not raw:
            raise ValueError("pixels must be a non-empty list of {az, rg}")
        if len(raw) > self.MAX_PIXELS:
            raise ValueError(f"at most {self.MAX_PIXELS} pixels per fit, got {len(raw)}")

        pixels = []
        seen   = set()
        for entry in raw:
            az, rg = int(entry["az"]), int(entry["rg"])
            if not (0 <= az < az_size and 0 <= rg < rg_size):
                raise ValueError(f"pixel az={az}, rg={rg} outside tomogram ({az_size} x {rg_size})")
            if (az, rg) in seen:
                continue
            seen.add((az, rg))
            pixels.append((az, rg))

        return pixels

    def _parse_config(self, raw) -> dict:
        if not isinstance(raw, dict):
            raise ValueError("config must be an object")

        missing = [key for key in self.CONFIG_KEYS if key not in raw]
        if missing:
            raise ValueError(f"config missing keys: {', '.join(missing)}")

        config = {
            "k_max"              : int(raw["k_max"]),
            "lambda_k"           : float(raw["lambda_k"]),
            "mode"               : str(raw["mode"]),
            "threshold_factor"   : float(raw["threshold_factor"]),
            "truncation_index"   : int(raw["truncation_index"]),
            "prominence_frac"    : float(raw["prominence_frac"]),
            "sigma_init_divisor" : float(raw["sigma_init_divisor"]),
            "activity_threshold" : float(raw["activity_threshold"]),
            "adam_steps"         : int(raw["adam_steps"]),
            "adam_lr"            : float(raw["adam_lr"]),
        }

        if not (1 <= config["k_max"] <= self.K_LIMIT):
            raise ValueError(f"k_max must be in [1, {self.K_LIMIT}]")
        if config["adam_steps"] < 1:
            raise ValueError("adam_steps must be >= 1")
        FitMode.free_flags(config["mode"])

        return config

    def _fit_worker(self, pixels: list, config: dict) -> None:
        try:
            self._set_fit(0.02, "loading fit kernels")
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            import jax.numpy as jnp
            from pipelines.processing.param_extraction.sigma.initialiser import PeakInitialiser
            from pipelines.processing.param_extraction.sigma.kernels     import SigmaAdamKernel
            from tools.data.gaussians                                    import GaussianMixture

            with self.lock:
                tomogram     = self.loaded["tomogram"]
                height_range = self.loaded["height_range"]
                dataset      = self.loaded["path"]

            self._set_fit(0.05, "reading profiles")
            H      = tomogram.shape[0]
            raw_hn = np.abs(np.stack([np.array(tomogram[:, az, rg]) for az, rg in pixels], axis=1)).astype(np.float32)
            target = ProfilePreprocessor.apply(raw_hn, config["threshold_factor"], config["truncation_index"])
            pf     = np.ascontiguousarray(target.T)

            active = pf.max(axis=1) > config["activity_threshold"]
            pmax   = pf.max(axis=1, keepdims=True)
            scale  = np.where(active[:, None], pmax, 1.0).astype(np.float32)
            norm   = pf / scale

            height_axis = np.linspace(height_range[0], height_range[1], H, dtype=np.float32)
            h_span      = float(height_axis[-1] - height_axis[0])
            dh          = h_span / (H - 1)

            k_max      = config["k_max"]
            active_idx = np.where(active)[0]
            per_k_out  = {}

            if len(active_idx) > 0:
                self._set_fit(0.1, "peak initialisation")
                if self._initialiser is None:
                    self._initialiser = PeakInitialiser(n_workers=1)
                if self._kernel is None:
                    self._kernel = SigmaAdamKernel()

                fit_sigma, fit_amplitude, fit_mean = FitMode.free_flags(config["mode"])

                amp_mask      = jnp.float32(1.0 if fit_amplitude else 0.0)
                mu_mask       = jnp.float32(1.0 if fit_mean else 0.0)
                sigma_mask    = jnp.float32(1.0 if fit_sigma else 0.0)
                mu_lower_j    = jnp.float32(height_axis[0])
                mu_upper_j    = jnp.float32(height_axis[-1])
                sigma_lower_j = jnp.float32(dh)
                sigma_upper_j = jnp.float32(h_span / 2.0)
                height_ax_j   = jnp.array(height_axis)

                prof_raw_act  = pf[active_idx]
                prof_norm_act = norm[active_idx].astype(np.float32)
                scale_act     = scale[active_idx, 0]
                n_act         = len(active_idx)

                amps_km, mus_km, sigs_km = self._initialiser.run(prof_raw_act, height_axis, k_max, config["prominence_frac"], config["sigma_init_divisor"])

                for K in range(1, k_max + 1):
                    self._set_fit(0.15 + 0.8 * (K - 1) / k_max, f"fitting K={K}/{k_max}")

                    amps_norm = amps_km[:, :K] / scale_act[:, None]
                    out_a, out_m, out_s, out_e = self._kernel(
                        jnp.array(self._pad_rows(amps_norm, self.BATCH_ROWS)),
                        jnp.array(self._pad_rows(mus_km[:, :K], self.BATCH_ROWS)),
                        jnp.array(self._pad_rows(sigs_km[:, :K], self.BATCH_ROWS)),
                        height_ax_j,
                        jnp.array(self._pad_rows(prof_norm_act, self.BATCH_ROWS)),
                        amp_mask,
                        mu_mask,
                        sigma_mask,
                        mu_lower_j,
                        mu_upper_j,
                        sigma_lower_j,
                        sigma_upper_j,
                        config["adam_steps"],
                        config["adam_lr"],
                        self.ADAM_B1,
                        self.ADAM_B2,
                    )

                    fit_a = np.array(out_a[:n_act], dtype=np.float32)
                    fit_m = np.array(out_m[:n_act], dtype=np.float32)
                    fit_s = np.array(out_s[:n_act], dtype=np.float32)
                    mse   = np.array(out_e[:n_act], dtype=np.float32)

                    per_k_out[K] = (fit_a, fit_m, fit_s, mse, mse + config["lambda_k"] * K)

            self._set_fit(0.97, "packing results")
            result = self._pack_result(GaussianMixture, pixels, config, dataset, height_axis, raw_hn, pf, scale, active, active_idx, per_k_out)

            with self.lock:
                self.fit_result = result
                self.fit_state  = {"state": "done", "progress": 1.0, "stage": "done", "error": ""}
            self.logger.ok(f"fitlab fit done: {len(pixels)} pixels, k_max={config['k_max']}, mode={config['mode']}")

        except Exception as exc:
            with self.lock:
                self.fit_state = {"state": "error", "progress": 0.0, "stage": "", "error": str(exc)}
            self.logger.error(f"fitlab fit failed: {exc}")

    @staticmethod
    def _pad_rows(arr: np.ndarray, target: int) -> np.ndarray:
        pad = target - arr.shape[0]
        if pad == 0:
            return np.ascontiguousarray(arr, dtype=np.float32)
        return np.concatenate([arr.astype(np.float32, copy=False), np.zeros((pad, arr.shape[1]), dtype=np.float32)], axis=0)

    def _pack_result(self, mixture, pixels: list, config: dict, dataset: str, height_axis: np.ndarray, raw_hn: np.ndarray, pf: np.ndarray, scale: np.ndarray, active: np.ndarray, active_idx: np.ndarray, per_k_out: dict) -> dict:
        act_pos = {int(idx): pos for pos, idx in enumerate(active_idx)}
        entries = []

        for i, (az, rg) in enumerate(pixels):
            entry = {
                "az"     : az,
                "rg"     : rg,
                "active" : bool(active[i]),
                "scale"  : float(scale[i, 0]),
                "raw"    : raw_hn[:, i].tolist(),
                "target" : pf[i].tolist(),
            }

            if active[i]:
                pos   = act_pos[i]
                per_k = []
                for K, (fit_a, fit_m, fit_s, mse, pen) in sorted(per_k_out.items()):
                    order       = np.argsort(fit_m[pos])
                    amps        = fit_a[pos][order] * scale[i, 0]
                    mus         = fit_m[pos][order]
                    sigs        = fit_s[pos][order]
                    interleaved = np.stack([amps, mus, sigs], axis=1).reshape(-1)

                    total, components = mixture.evaluate_pixel(interleaved, height_axis, K)
                    per_k.append({
                        "k"          : K,
                        "mse"        : float(mse[pos]),
                        "penalised"  : float(pen[pos]),
                        "params"     : [{"amp": float(a), "mu": float(m), "sigma": float(s)} for a, m, s in zip(amps, mus, sigs)],
                        "total"      : total.tolist(),
                        "components" : [comp.tolist() for comp in components],
                    })

                entry["per_k"]  = per_k
                entry["best_k"] = int(np.argmin([row["penalised"] for row in per_k])) + 1

            entries.append(entry)

        return {
            "ok"      : True,
            "dataset" : dataset,
            "height"  : height_axis.tolist(),
            "config"  : config,
            "pixels"  : entries,
        }

    def _set_fit(self, progress: float, stage: str) -> None:
        with self.lock:
            self.fit_state["progress"] = float(progress)
            self.fit_state["stage"]    = stage

    def fit_status(self) -> dict:
        with self.lock:
            return dict(self.fit_state)

    def fit_result_payload(self) -> dict:
        with self.lock:
            if self.fit_result is None:
                return {"ok": False, "error": "no fit result available"}
            return self.fit_result
