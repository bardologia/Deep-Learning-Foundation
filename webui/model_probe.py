
from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from catalog_roots            import CatalogRoots, RunScanner
from tools.reporting.plotting import PlotBase
from web_logger               import WebLogger

from pipelines.shared.inference.run_classifier import RunArtifacts


class ModelProbe:

    FAMILIES        = ("amp", "mu", "sigma")
    FIELD_CMAPS     = {"amp": "magma", "mu": "viridis", "sigma": "cividis"}
    PERTURBATIONS   = ("drop_channel", "scale_channel", "noise")
    FLIPS           = (("none", ()), ("row", (2,)), ("col", (3,)), ("both", (2, 3)))
    CONTAINER_TYPES = ("ModuleList", "ModuleDict", "Sequential")
    FUSED_LEAVES    = ("attention_dropout",)
    LOADER_NAME     = "model_probe"
    CHECKPOINT      = "best_model.pt"
    CONFIG_NAME     = RunArtifacts.BACKBONE_CONFIG

    def __init__(self, logger: WebLogger) -> None:
        self.logger  = logger
        self.lock    = threading.Lock()
        self.scanner = RunScanner(CatalogRoots())

        self.loaded = None
        self.status = {"state": "idle", "path": "", "progress": 0.0, "stage": "", "error": "", "info": None}

    def runs(self, base: str) -> dict:
        scanned = self.scanner.checkpoint_runs(base, self.CHECKPOINT, self.CONFIG_NAME)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "runs": []}

        return {"ok": True, "root": scanned["root"], "runs": scanned["entries"]}

    @classmethod
    def _reject_unloadable(cls, run_path: str, roots: CatalogRoots) -> str:
        run_dir = Path(run_path)

        if not run_dir.is_dir():
            return f"'{run_path}' is not a directory; pick a training run from the list"
        if not roots.contains(run_dir.resolve()):
            return f"'{run_dir}' is outside every catalogued runs root; scan the runs directory first"
        if not (run_dir / cls.CHECKPOINT).is_file():
            return f"'{run_dir.name}' holds no {cls.CHECKPOINT}; it is not a finished training run (did you point at a runs root or an inference stamp?)"
        if not (run_dir / "meta" / cls.CONFIG_NAME).is_file():
            return f"'{run_dir.name}' holds no meta/{cls.CONFIG_NAME}; the microscope probes backbone runs only"

        return ""

    def start_load(self, run_path: str, split: str = "test", device: str = "cpu") -> dict:
        refusal = self._reject_unloadable(run_path, self.scanner.roots)
        if refusal:
            return {"ok": False, "error": refusal}

        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": "a model load is already running"}

            self.loaded = None
            self.status = {"state": "loading", "path": run_path, "progress": 0.0, "stage": "opening run", "error": "", "info": None}

        threading.Thread(target=self._load_worker, args=(run_path, split, device), name="ModelProbeLoad", daemon=True).start()
        return {"ok": True}

    def _set_load(self, progress: float, stage: str) -> None:
        with self.lock:
            self.status["progress"] = progress
            self.status["stage"]    = stage

    def _probe_layers(self, model) -> list[str]:
        from tools.diagnostics.activation_recorder import ActivationRecorder

        modules = dict(model.named_modules())
        return [name for name in ActivationRecorder(model).leaf_names() if type(modules[name]).__name__ not in self.CONTAINER_TYPES and name.rsplit(".", 1)[-1] not in self.FUSED_LEAVES]

    def _load_worker(self, run_path: str, split: str, device: str) -> None:
        try:
            from pipelines.backbone.inference.analysis.input_attribution import ChannelLabeler
            from pipelines.backbone.inference.loader            import RunLoader
            from pipelines.backbone.inference.probes            import PredictionCurves
            from tools.monitoring.logger                        import Logger

            self._set_load(0.1, "loading checkpoint and dataset")

            run = RunLoader(Path(run_path), logger=Logger(log_dir="", name=self.LOADER_NAME)).load(
                split           = split,
                batch_size      = 1,
                num_workers     = 0,
                device          = device,
                checkpoint_name = "best_model.pt",
            )

            self._set_load(0.8, "indexing layers")

            labels   = ChannelLabeler.build(run)
            layers   = self._probe_layers(run.model.module)
            modules  = dict(run.model.module.named_modules())
            renderer = PredictionCurves(run.n_gaussians, run.z_axis)

            info = {
                "run"          : run_path,
                "backbone"     : run.backbone_name,
                "split"        : split,
                "in_channels"  : run.in_channels,
                "n_gaussians"  : run.n_gaussians,
                "row_size" : run.split_region.row_size,
                "col_size"   : run.split_region.col_size,
                "row_offset"    : run.split_region.row_start,
                "col_offset"    : run.split_region.col_start,
                "patch"        : list(run.dataset_config.patch.size),
                "channels"     : labels,
                "n_layers"     : len(layers),
            }

            with self.lock:
                self.loaded = {
                    "run"      : run,
                    "labels"   : labels,
                    "layers"   : layers,
                    "types"    : {name: type(modules[name]).__name__ for name in layers},
                    "renderer" : renderer,
                    "patch"    : tuple(run.dataset_config.patch.size),
                    "device"   : device,
                }
                self.status = {"state": "ready", "path": run_path, "progress": 1.0, "stage": "ready", "error": "", "info": info}

            self.logger.info(f"model probe loaded {info['backbone']} from {run_path} ({len(layers)} layers)")

        except Exception as error:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "path": run_path, "progress": 0.0, "stage": "", "error": f"{type(error).__name__}: {error}", "info": None}
            self.logger.error(f"model probe load failed: {error}")

    def load_status(self) -> dict:
        with self.lock:
            return dict(self.status)

    def layers(self) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}
            return {"ok": True, "layers": [{"name": name, "type": self.loaded["types"][name]} for name in self.loaded["layers"]]}

    def map_png(self) -> bytes | None:
        with self.lock:
            if self.loaded is None:
                return None
            primary = np.abs(self.loaded["run"].complex_inputs[0]).astype(np.float32)

        vmin, vmax = PlotBase._amplitude_clim(primary)

        buf = io.BytesIO()
        plt.imsave(buf, primary, cmap="gray", vmin=float(vmin), vmax=float(vmax), format="png")
        return buf.getvalue()

    @classmethod
    def _window_at(cls, run, patch: tuple, row: int, col: int) -> tuple[np.ndarray, int, int]:
        ph, pw = patch
        n_rows   = run.split_region.row_size
        n_cols   = run.split_region.col_size

        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise ValueError(f"pixel ({row}, {col}) is outside the {n_rows}x{n_cols} region")

        top  = int(np.clip(row - ph // 2, 0, n_rows - ph))
        left = int(np.clip(col - pw // 2, 0, n_cols - pw))

        complex_window = run.complex_inputs[:, top:top + ph, left:left + pw]
        aux            = run.dataset.aux
        aux_window     = aux[top:top + ph, left:left + pw] if aux is not None else None

        window = run.dataset.assemble_window(complex_window, aux_window)
        return window[None].astype(np.float32), row - top, col - left

    def _window(self, row: int, col: int) -> tuple[np.ndarray, int, int]:
        return self._window_at(self.loaded["run"], self.loaded["patch"], row, col)

    def _slots(self, params_center: np.ndarray) -> list[dict]:
        from tools.data.gaussians import GaussianActivity

        run   = self.loaded["run"]
        slots = []
        for k in range(run.n_gaussians):
            amp = float(params_center[3 * k])
            slots.append({
                "slot"   : k,
                "amp"    : amp,
                "mu"     : float(params_center[3 * k + 1]),
                "sigma"  : float(params_center[3 * k + 2]),
                "active" : bool(GaussianActivity.is_active(amp, run.activity_threshold)),
            })
        return slots

    def _predict_window(self, window: np.ndarray, cy: int, cx: int) -> tuple[list[dict], list[float], np.ndarray]:
        run      = self.loaded["run"]
        renderer = self.loaded["renderer"]

        params = run.model(window)
        curve  = renderer.render(params)[0, :, cy, cx]
        center = np.asarray(params[0, :, cy, cx], dtype=np.float64)

        return self._slots(center), [float(v) for v in curve], center

    def _center_curves(self, params: np.ndarray, cy: int, cx: int) -> np.ndarray:
        center = params[:, :, cy, cx][:, :, None, None]
        return self.loaded["renderer"].render(center)[:, :, 0, 0]

    def _match_gt(self, pred_center: np.ndarray, gt_center: np.ndarray) -> list[dict]:
        import torch

        from tools.data.gaussians  import GaussianActivity
        from tools.loss.param_loss import ParamMatcher

        run    = self.loaded["run"]
        n_k    = run.n_gaussians
        pred_t = torch.from_numpy(np.asarray(pred_center, dtype=np.float32)).reshape(1, n_k, 3, 1, 1)
        gt_t   = torch.from_numpy(np.asarray(gt_center, dtype=np.float32)).reshape(1, n_k, 3, 1, 1)
        idx_t  = torch.arange(n_k, dtype=torch.float32).reshape(1, n_k, 1, 1, 1).expand_as(pred_t)

        matched_pred, matched_idx, sorted_gt, _ = ParamMatcher.match(pred_t, idx_t, gt_t, gt_t, active_thr=run.activity_threshold, method=ParamMatcher.HUNGARIAN)

        amps  = np.asarray(gt_center[0::3], dtype=np.float64)
        mus   = np.asarray(gt_center[1::3], dtype=np.float64)
        order = np.argsort(np.where(GaussianActivity.is_active(amps, run.activity_threshold), mus, np.inf), kind="stable")

        matches = []
        for j in range(n_k):
            if not GaussianActivity.is_active(float(sorted_gt[0, j, 0, 0, 0]), run.activity_threshold):
                continue
            matches.append({
                "gt_slot"   : int(order[j]),
                "pred_slot" : int(matched_idx[0, j, 0, 0, 0]),
                "d_amp"     : float(matched_pred[0, j, 0, 0, 0] - sorted_gt[0, j, 0, 0, 0]),
                "d_mu"      : float(matched_pred[0, j, 1, 0, 0] - sorted_gt[0, j, 1, 0, 0]),
                "d_sigma"   : float(matched_pred[0, j, 2, 0, 0] - sorted_gt[0, j, 2, 0, 0]),
            })

        return matches

    def _fit_facts(self, curve: list, gt_curve: list | None, raw_curve: list, slots: list, gt_slots: list | None) -> dict:
        pred = np.asarray(curve, dtype=np.float64)

        return {
            "curve_mse_raw" : float(((pred - np.asarray(raw_curve)) ** 2).mean()),
            "curve_mse_gt"  : float(((pred - np.asarray(gt_curve)) ** 2).mean()) if gt_curve is not None else None,
            "pred_active"   : sum(1 for slot in slots if slot["active"]),
            "gt_active"     : sum(1 for slot in gt_slots if slot["active"]) if gt_slots is not None else None,
        }

    def predict(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col                = int(body["row"]), int(body["col"])
                window, cy, cx        = self._window(row, col)
                slots, curve, center  = self._predict_window(window, cy, cx)

                run      = self.loaded["run"]
                renderer = self.loaded["renderer"]

                gt_curve  = None
                gt_slots  = None
                matches   = None
                gt_params = run.dataset.gt_parameters
                if gt_params is not None:
                    gt_center = np.asarray(gt_params[:, row, col], dtype=np.float64)
                    gt_slots  = self._slots(gt_center)
                    gt_curve  = [float(v) for v in renderer.render(gt_center.reshape(1, -1, 1, 1))[0, :, 0, 0]]
                    matches   = self._match_gt(center, gt_center)

                raw_curve = [float(v) for v in run.full_curves[:, row, col]]

                return {
                    "ok"        : True,
                    "row"       : row,
                    "col"       : col,
                    "z_axis"    : [float(v) for v in run.z_axis],
                    "slots"     : slots,
                    "curve"     : curve,
                    "gt_slots"  : gt_slots,
                    "gt_curve"  : gt_curve,
                    "raw_curve" : raw_curve,
                    "matches"   : matches,
                    "fit"       : self._fit_facts(curve, gt_curve, raw_curve, slots, gt_slots),
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def _field_png(self, cell: np.ndarray, cmap: str, vmin: float, vmax: float) -> str:
        buf = io.BytesIO()
        plt.imsave(buf, np.asarray(cell, dtype=np.float32), cmap=cmap, vmin=vmin, vmax=vmax, format="png")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _field_payload(self, cell: np.ndarray, cmap: str, vmin: float | None = None, vmax: float | None = None) -> dict:
        lo = float(cell.min()) if vmin is None else float(vmin)
        hi = float(cell.max()) if vmax is None else float(vmax)
        hi = hi if hi > lo else lo + 1.0

        return {"png": self._field_png(cell, cmap, lo, hi), "min": float(cell.min()), "max": float(cell.max())}

    def _error_map(self, params: np.ndarray, row: int, col: int, cy: int, cx: int) -> np.ndarray | None:
        run = self.loaded["run"]
        gt  = run.dataset.gt_parameters
        if gt is None:
            return None

        ph, pw    = params.shape[2], params.shape[3]
        top, left = row - cy, col - cx
        gt_window = np.asarray(gt[:, top:top + ph, left:left + pw], dtype=np.float32)[None]

        renderer    = self.loaded["renderer"]
        pred_curves = renderer.render(params)
        gt_curves   = renderer.render(gt_window)

        return ((pred_curves - gt_curves) ** 2).mean(axis=1)[0]

    def fields(self, body: dict) -> dict:
        from tools.data.gaussians import GaussianActivity

        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col         = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run    = self.loaded["run"]
                params = run.model(window)
                n_k    = run.n_gaussians

                slots = []
                for k in range(n_k):
                    families = [
                        {"family": family, **self._field_payload(params[0, 3 * k + offset], self.FIELD_CMAPS[family])}
                        for offset, family in enumerate(self.FAMILIES)
                    ]
                    slots.append({
                        "slot"     : k,
                        "active"   : bool(GaussianActivity.is_active(float(params[0, 3 * k, cy, cx]), run.activity_threshold)),
                        "families" : families,
                    })

                activity  = GaussianActivity.is_active(params[0, 0::3], run.activity_threshold).sum(axis=0).astype(np.float32)
                error_map = self._error_map(params, row, col, cy, cx)

                return {
                    "ok"       : True,
                    "center"   : [cy, cx],
                    "patch"    : [int(params.shape[2]), int(params.shape[3])],
                    "slots"    : slots,
                    "activity" : self._field_payload(activity, "viridis", vmin=0.0, vmax=float(n_k)),
                    "error"    : self._field_payload(error_map, "inferno") if error_map is not None else None,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    @classmethod
    def _family_gradients_at(cls, run, device: str, window: np.ndarray, cy: int, cx: int, slot: int) -> list[tuple[str, np.ndarray]]:
        import torch

        x = torch.from_numpy(window).to(device).requires_grad_(True)
        y = run.model.module(x)

        gradients = []
        for offset, family in enumerate(cls.FAMILIES):
            target = y[0, offset::3, cy, cx].abs().sum() if slot < 0 else y[0, 3 * slot + offset, cy, cx].abs()
            keep   = offset < len(cls.FAMILIES) - 1
            grad   = torch.autograd.grad(target, x, retain_graph=keep)[0].abs()[0].cpu().numpy()
            gradients.append((family, grad))

        return gradients

    def _family_gradients(self, window: np.ndarray, cy: int, cx: int, slot: int) -> list[tuple[str, np.ndarray]]:
        return self._family_gradients_at(self.loaded["run"], self.loaded["device"], window, cy, cx, slot)

    @classmethod
    def _cell_png(cls, cell: np.ndarray) -> str:
        buf = io.BytesIO()
        plt.imsave(buf, cell, cmap="magma", vmin=0.0, vmax=1.0, format="png")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @classmethod
    def _radial_profile(cls, grad: np.ndarray, cy: int, cx: int) -> dict:
        total_map = grad.sum(axis=0)
        total     = float(total_map.sum())

        yy, xx = np.mgrid[0:total_map.shape[0], 0:total_map.shape[1]]
        r      = np.hypot(yy - cy, xx - cx)
        radii  = np.arange(int(np.ceil(r.max())) + 1)

        cumulative = np.asarray([float(total_map[r <= radius].sum()) / total for radius in radii])
        r50        = int(radii[np.argmax(cumulative >= 0.5)])
        r90        = int(radii[np.argmax(cumulative >= 0.9)])

        return {"radii": [int(v) for v in radii], "cumulative": [float(v) for v in cumulative], "r50": r50, "r90": r90}

    def _family_payload(self, family: str, grad: np.ndarray, cy: int, cx: int) -> dict:
        total = float(grad.sum())
        if total <= 0.0:
            return {"family": family, "dead": True, "shares": [0.0] * grad.shape[0], "cells": [None] * grad.shape[0], "radial": None}

        shares = grad.sum(axis=(1, 2)) / total

        cells = []
        for channel in range(grad.shape[0]):
            peak = float(grad[channel].max())
            cells.append(self._cell_png(grad[channel] / peak) if peak > 0.0 else None)

        return {
            "family" : family,
            "dead"   : False,
            "shares" : [float(v) for v in shares],
            "cells"  : cells,
            "radial" : self._radial_profile(grad, cy, cx),
        }

    def attribution(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col = int(body["row"]), int(body["col"])
                slot   = int(body.get("slot", -1))
                n_k    = self.loaded["run"].n_gaussians

                if not -1 <= slot < n_k:
                    raise ValueError(f"slot {slot} is out of range; the model predicts {n_k} component slots (use -1 to sum over all of them)")

                window, cy, cx = self._window(row, col)

                families = [self._family_payload(family, grad, cy, cx) for family, grad in self._family_gradients(window, cy, cx, slot)]

                return {
                    "ok"       : True,
                    "channels" : self.loaded["labels"],
                    "center"   : [cy, cx],
                    "patch"    : [int(window.shape[2]), int(window.shape[3])],
                    "slot"     : slot,
                    "families" : families,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def _slot_shift(self, base_slots: list[dict], moved_slots: list[dict]) -> tuple[int, float]:
        flips    = 0
        mu_shift = 0.0

        for base, moved in zip(base_slots, moved_slots):
            if base["active"] != moved["active"]:
                flips += 1
            if base["active"] or moved["active"]:
                mu_shift = max(mu_shift, abs(moved["mu"] - base["mu"]))

        return flips, mu_shift

    def ablation(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col         = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run        = self.loaded["run"]
                n_channels = window.shape[1]

                batch = np.repeat(window, n_channels + 1, axis=0)
                for channel in range(n_channels):
                    batch[channel + 1, channel] = 0.0

                params = run.model(batch)
                curves = self._center_curves(params, cy, cx)

                base_curve = curves[0]
                base_slots = self._slots(params[0, :, cy, cx])
                base_power = float((base_curve ** 2).mean())

                channels = []
                for channel in range(n_channels):
                    slots           = self._slots(params[channel + 1, :, cy, cx])
                    flips, mu_shift = self._slot_shift(base_slots, slots)
                    channels.append({
                        "channel"      : channel,
                        "label"        : self.loaded["labels"][channel],
                        "delta_mse"    : float(((curves[channel + 1] - base_curve) ** 2).mean()),
                        "flips"        : flips,
                        "max_mu_shift" : mu_shift,
                    })

                channels.sort(key=lambda entry: entry["delta_mse"], reverse=True)

                return {"ok": True, "base_power": base_power, "channels": channels}
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def flips(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col         = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run = self.loaded["run"]

                batch  = np.concatenate([np.flip(window, axes) if axes else window for _, axes in self.FLIPS], axis=0)
                params = run.model(batch)

                entries    = []
                base_curve = None
                for index, (name, axes) in enumerate(self.FLIPS):
                    restored = np.flip(params[index:index + 1], axes) if axes else params[index:index + 1]
                    curve    = self._center_curves(restored, cy, cx)[0]

                    if base_curve is None:
                        base_curve = curve

                    entries.append({
                        "flip"      : name,
                        "slots"     : self._slots(restored[0, :, cy, cx]),
                        "curve"     : [float(v) for v in curve],
                        "delta_mse" : float(((curve - base_curve) ** 2).mean()),
                    })

                return {
                    "ok"         : True,
                    "z_axis"     : [float(v) for v in run.z_axis],
                    "base_power" : float((base_curve ** 2).mean()),
                    "entries"    : entries,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def occlusion(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col         = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run    = self.loaded["run"]
                ph, pw = window.shape[2], window.shape[3]
                oh, ow = max(2, ph // 8), max(2, pw // 8)
                rows   = int(np.ceil(ph / oh))
                cols   = int(np.ceil(pw / ow))

                batch = np.repeat(window, rows * cols + 1, axis=0)
                for row in range(rows):
                    for col in range(cols):
                        cell = 1 + row * cols + col
                        batch[cell, :, row * oh:min(ph, (row + 1) * oh), col * ow:min(pw, (col + 1) * ow)] = 0.0

                curves     = self._center_curves(run.model(batch), cy, cx)
                base_curve = curves[0]
                deltas     = ((curves[1:] - base_curve[None]) ** 2).mean(axis=1).reshape(rows, cols)

                worst = np.unravel_index(int(np.argmax(deltas)), deltas.shape)

                return {
                    "ok"          : True,
                    "center"      : [cy, cx],
                    "patch"       : [ph, pw],
                    "occluder"    : [oh, ow],
                    "grid"        : [rows, cols],
                    "center_cell" : [cy // oh, cx // ow],
                    "base_power"  : float((base_curve ** 2).mean()),
                    "worst"       : {"row": int(worst[0]), "col": int(worst[1]), "delta_mse": float(deltas[worst])},
                    "map"         : self._field_payload(deltas, "magma"),
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    @classmethod
    def _perturb(cls, window: np.ndarray, perturbation: dict) -> np.ndarray:
        kind      = perturbation.get("kind")
        perturbed = window.copy()

        if kind == "drop_channel":
            channel                  = int(perturbation["channel"])
            perturbed[:, channel]    = 0.0
        elif kind == "scale_channel":
            channel                  = int(perturbation["channel"])
            perturbed[:, channel]   *= float(perturbation.get("factor", 0.5))
        elif kind == "noise":
            sigma = float(perturbation.get("sigma", 0.5))
            rng   = np.random.default_rng(int(perturbation.get("seed", 0)))
            perturbed += rng.normal(0.0, sigma, size=perturbed.shape).astype(np.float32)
        else:
            raise ValueError(f"unknown perturbation '{kind}', expected one of {cls.PERTURBATIONS}")

        return perturbed

    def _sweep_variants(self, window: np.ndarray, kind: str, channel: int | None) -> tuple[np.ndarray, list[np.ndarray], int]:
        if kind == "scale_channel":
            values   = np.linspace(0.0, 2.0, 9)
            variants = [self._perturb(window, {"kind": kind, "channel": channel, "factor": float(v)}) for v in values]
            return values, variants, 1

        if kind == "noise":
            seeds    = (0, 1, 2)
            values   = np.linspace(0.0, 2.0, 9)
            variants = [self._perturb(window, {"kind": kind, "sigma": float(v), "seed": seed}) for v in values for seed in seeds]
            return values, variants, len(seeds)

        raise ValueError(f"kind '{kind}' has no strength to sweep; sweep one of ('scale_channel', 'noise')")

    def sweep(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col         = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                kind    = body.get("kind", "")
                channel = None
                if kind == "scale_channel":
                    channel = int(body["channel"])
                    if not 0 <= channel < window.shape[1]:
                        raise ValueError(f"channel {channel} is out of range; the model reads {window.shape[1]} input channels")

                values, variants, group = self._sweep_variants(window, kind, channel)

                run = self.loaded["run"]

                params = run.model(np.concatenate([window] + variants, axis=0))
                curves = self._center_curves(params, cy, cx)

                base_curve = curves[0]
                base_slots = self._slots(params[0, :, cy, cx])

                deltas = ((curves[1:] - base_curve[None]) ** 2).mean(axis=1)
                flips  = [self._slot_shift(base_slots, self._slots(params[index + 1, :, cy, cx]))[0] for index in range(len(variants))]

                points = []
                for index, value in enumerate(values):
                    chunk = deltas[index * group:(index + 1) * group]
                    points.append({
                        "value"     : float(value),
                        "delta_mse" : float(chunk.mean()),
                        "spread"    : float(chunk.std()) if group > 1 else 0.0,
                        "flips"     : int(max(flips[index * group:(index + 1) * group])),
                    })

                return {
                    "ok"         : True,
                    "kind"       : kind,
                    "channel"    : channel,
                    "label"      : self.loaded["labels"][channel] if channel is not None else None,
                    "base_power" : float((base_curve ** 2).mean()),
                    "points"     : points,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def whatif(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col         = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                base_slots, base_curve, _ = self._predict_window(window, cy, cx)

                perturbed                             = self._perturb(window, body.get("perturbation", {}))
                perturbed_slots, perturbed_curve, _   = self._predict_window(perturbed, cy, cx)

                base      = np.asarray(base_curve)
                shifted   = np.asarray(perturbed_curve)
                delta_mse = float(((base - shifted) ** 2).mean())

                return {
                    "ok"              : True,
                    "row"             : row,
                    "col"             : col,
                    "z_axis"          : [float(v) for v in self.loaded["run"].z_axis],
                    "base_slots"      : base_slots,
                    "base_curve"      : base_curve,
                    "perturbed_slots" : perturbed_slots,
                    "perturbed_curve" : perturbed_curve,
                    "delta_mse"       : delta_mse,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    @classmethod
    def _vital_flags(cls, stat: dict) -> list[str]:
        flags = []
        if stat["nonfinite_frac"] > 0.0:
            flags.append("nonfinite")
        if stat["zero_frac"] > 0.9:
            flags.append("mostly zero")
        if stat["dead_channel_frac"] is not None and stat["dead_channel_frac"] > 0.25:
            flags.append("dead channels")
        return flags

    @classmethod
    def _vitals_payload(cls, model, stats: dict) -> dict:
        from tools.diagnostics.activation_recorder import ActivationRecorder

        modules = dict(model.named_modules())
        entries = []
        for name, stat in sorted(stats.items(), key=lambda kv: kv[1]["first_seen"]):
            entries.append({
                "name"      : name,
                "type"      : stat["module_type"],
                "shape"     : stat["shape"],
                "params"    : int(sum(p.numel() for p in modules[ActivationRecorder.module_of(name)].parameters())),
                "zero_frac" : stat["zero_frac"],
                "dead"      : stat["dead_channels"],
                "channels"  : stat["n_channels"],
                "eff_frac"  : stat["effective_channel_frac"],
                "max_abs"   : stat["max_abs"],
                "flags"     : cls._vital_flags(stat),
            })

        summary = {
            "n_layers"     : len(entries),
            "total_params" : int(sum(p.numel() for p in model.parameters())),
            "flagged"      : sum(1 for entry in entries if entry["flags"]),
        }

        return {"entries": entries, "summary": summary}

    def vitals(self, body: dict) -> dict:
        import torch

        from tools.diagnostics.activation_recorder import ActivationRecorder

        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col           = int(body["row"]), int(body["col"])
                window, _cy, _cx = self._window(row, col)

                model    = self.loaded["run"].model.module
                recorder = ActivationRecorder(model)
                recorder.attach_stats()

                try:
                    with torch.no_grad():
                        model(torch.from_numpy(window).to(self.loaded["device"]))
                finally:
                    recorder.detach()

                return {"ok": True, **self._vitals_payload(model, recorder.stats())}
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def features_png(self, row: int, col: int, layer: str, max_channels: int = 16) -> bytes | None:
        import torch

        from tools.diagnostics.activation_recorder import ActivationRecorder

        with self.lock:
            if self.loaded is None or layer not in self.loaded["layers"]:
                return None

            window, _cy, _cx = self._window(row, col)

            recorder = ActivationRecorder(self.loaded["run"].model.module)
            recorder.attach_store([layer])

            try:
                with torch.no_grad():
                    self.loaded["run"].model.module(torch.from_numpy(window).to(self.loaded["device"]))
            finally:
                recorder.detach()

            stored = recorder.stored().get(layer)

        if stored is None:
            return None
        if stored.ndim != 4:
            raise ValueError(f"layer '{layer}' emits a {stored.ndim}-D activation; the feature grid renders 4-D (B, C, H, W) maps only")

        maps   = stored[0].numpy()
        energy = np.abs(maps).sum(axis=(1, 2))
        order  = np.argsort(energy)[::-1][:max_channels]

        n_cols = 4
        n_rows = int(np.ceil(len(order) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.0))
        axes      = np.atleast_2d(axes)

        for index, channel in enumerate(order):
            ax = axes[index // n_cols][index % n_cols]
            ax.imshow(maps[channel], cmap="magma", aspect="auto", interpolation="nearest")
            ax.set_title(f"ch {int(channel)}", fontsize=7)
            ax.axis("off")

        for index in range(len(order), n_rows * n_cols):
            axes[index // n_cols][index % n_cols].axis("off")

        return self._figure_png(fig)

    def _figure_png(self, fig) -> bytes:
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        return buf.getvalue()

    def _kernel_matrix_png(self, w: np.ndarray) -> bytes:
        matrix = w[:, :, 0, 0]
        peak   = float(np.abs(matrix).max()) or 1.0

        fig, ax = plt.subplots(figsize=(min(10.0, 1.6 + matrix.shape[1] * 0.11), min(8.0, 1.2 + matrix.shape[0] * 0.11)))
        ax.imshow(matrix, cmap="RdBu_r", vmin=-peak, vmax=peak, aspect="auto", interpolation="nearest")
        ax.set_xlabel("input channel", fontsize=8)
        ax.set_ylabel("output channel", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"1×1 convolution — {matrix.shape[0]}×{matrix.shape[1]} weight matrix, ±{peak:.2g}", fontsize=8)

        return self._figure_png(fig)

    def kernels_png(self, layer: str, max_out: int = 12, max_in: int = 8) -> bytes | None:
        with self.lock:
            if self.loaded is None or layer not in self.loaded["layers"]:
                return None
            module = dict(self.loaded["run"].model.module.named_modules())[layer]

        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 4:
            raise ValueError(f"layer '{layer}' holds no 4-D convolution weight; kernels render convolution layers only")

        w = weight.detach().cpu().numpy()
        if "Transpose" in type(module).__name__:
            w = w.transpose(1, 0, 2, 3)

        out_ch, in_ch, kh, kw = w.shape

        if kh == 1 and kw == 1:
            return self._kernel_matrix_png(w)

        order_out = np.argsort(np.linalg.norm(w.reshape(out_ch, -1), axis=1))[::-1][:max_out]
        order_in  = np.argsort(np.abs(w).sum(axis=(0, 2, 3)))[::-1][:max_in]
        peak      = float(np.abs(w[np.ix_(order_out, order_in)]).max()) or 1.0

        fig, axes = plt.subplots(len(order_out), len(order_in), figsize=(len(order_in) * 1.15 + 0.9, len(order_out) * 1.15 + 0.7))
        axes      = np.asarray(axes).reshape(len(order_out), len(order_in))

        for row, out_index in enumerate(order_out):
            for col, in_index in enumerate(order_in):
                ax = axes[row][col]
                ax.imshow(w[out_index, in_index], cmap="RdBu_r", vmin=-peak, vmax=peak, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"in {int(in_index)}", fontsize=7)
                if col == 0:
                    ax.set_ylabel(f"out {int(out_index)}", fontsize=7)

        fig.suptitle(f"{out_ch}×{in_ch}×{kh}×{kw} weight — strongest {len(order_out)}×{len(order_in)} kernels, ±{peak:.2g}", fontsize=8)

        return self._figure_png(fig)
