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


class ModelProbe:

    FAMILIES        = ("amp", "mu", "sigma")
    PERTURBATIONS   = ("drop_channel", "scale_channel", "noise")
    CONTAINER_TYPES = ("ModuleList", "ModuleDict", "Sequential")
    LOADER_NAME     = "model_probe"
    CHECKPOINT      = "best_model.pt"
    CONFIG_NAME     = "model_config.json"

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

    def _reject_unloadable(self, run_path: str) -> str:
        run_dir = Path(run_path)

        if not run_dir.is_dir():
            return f"'{run_path}' is not a directory; pick a training run from the list"
        if not (run_dir / self.CHECKPOINT).is_file():
            return f"'{run_dir.name}' holds no {self.CHECKPOINT}; it is not a finished training run (did you point at a runs root or an inference stamp?)"
        if not (run_dir / "meta" / self.CONFIG_NAME).is_file():
            return f"'{run_dir.name}' holds no meta/{self.CONFIG_NAME}; the microscope probes backbone runs only, and this run is another family or predates config persistence"

        return ""

    def start_load(self, run_path: str, split: str = "test", device: str = "cpu") -> dict:
        refusal = self._reject_unloadable(run_path)
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
        return [name for name in ActivationRecorder(model).leaf_names() if type(modules[name]).__name__ not in self.CONTAINER_TYPES]

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
            renderer = PredictionCurves(run.n_gaussians, run.x_axis)

            info = {
                "run"          : run_path,
                "backbone"     : run.backbone_name,
                "split"        : split,
                "in_channels"  : run.in_channels,
                "n_gaussians"  : run.n_gaussians,
                "azimuth_size" : run.split_region.azimuth_size,
                "range_size"   : run.split_region.range_size,
                "az_offset"    : run.split_region.azimuth_start,
                "rg_offset"    : run.split_region.range_start,
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

            self.logger.info(f"model probe loaded {run.backbone_name} from {run_path} ({len(layers)} layers)")

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

    def _window(self, az: int, rg: int) -> tuple[np.ndarray, int, int]:
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]
        n_az   = run.split_region.azimuth_size
        n_rg   = run.split_region.range_size

        if not (0 <= az < n_az and 0 <= rg < n_rg):
            raise ValueError(f"pixel ({az}, {rg}) is outside the {n_az}x{n_rg} region")

        top  = int(np.clip(az - ph // 2, 0, n_az - ph))
        left = int(np.clip(rg - pw // 2, 0, n_rg - pw))

        complex_window = run.complex_inputs[:, top:top + ph, left:left + pw]
        dem            = run.dataset.dem
        dem_window     = dem[top:top + ph, left:left + pw] if dem is not None else None

        window = run.dataset.assemble_window(complex_window, dem_window)
        return window[None].astype(np.float32), az - top, rg - left

    def _slots(self, params_center: np.ndarray) -> list[dict]:
        from tools.loss.param_loss import ParamMatcher

        n_k   = self.loaded["run"].n_gaussians
        slots = []
        for k in range(n_k):
            amp = float(params_center[3 * k])
            slots.append({
                "slot"   : k,
                "amp"    : amp,
                "mu"     : float(params_center[3 * k + 1]),
                "sigma"  : float(params_center[3 * k + 2]),
                "active" : bool(ParamMatcher.is_active(amp)),
            })
        return slots

    def _predict_window(self, window: np.ndarray, cy: int, cx: int) -> tuple[list[dict], list[float]]:
        run      = self.loaded["run"]
        renderer = self.loaded["renderer"]

        params = run.model(window)
        curve  = renderer.render(params)[0, :, cy, cx]

        return self._slots(params[0, :, cy, cx]), [float(v) for v in curve]

    def predict(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                az, rg           = int(body["az"]), int(body["rg"])
                window, cy, cx   = self._window(az, rg)
                slots, curve     = self._predict_window(window, cy, cx)

                run      = self.loaded["run"]
                renderer = self.loaded["renderer"]

                gt_curve  = None
                gt_slots  = None
                gt_params = run.dataset.gt_parameters
                if gt_params is not None:
                    center    = np.asarray(gt_params[:, az, rg], dtype=np.float64)
                    gt_slots  = self._slots(center)
                    gt_curve  = [float(v) for v in renderer.render(center.reshape(1, -1, 1, 1))[0, :, 0, 0]]

                raw_curve = [float(v) for v in run.full_curves[:, az, rg]]

                return {
                    "ok"        : True,
                    "az"        : az,
                    "rg"        : rg,
                    "x_axis"    : [float(v) for v in run.x_axis],
                    "slots"     : slots,
                    "curve"     : curve,
                    "gt_slots"  : gt_slots,
                    "gt_curve"  : gt_curve,
                    "raw_curve" : raw_curve,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def _family_gradients(self, window: np.ndarray, cy: int, cx: int) -> list[tuple[str, np.ndarray]]:
        import torch

        x = torch.from_numpy(window).to(self.loaded["device"]).requires_grad_(True)
        y = self.loaded["run"].model.module(x)

        gradients = []
        for offset, family in enumerate(self.FAMILIES):
            target = y[0, offset::3, cy, cx].abs().sum()
            keep   = offset < len(self.FAMILIES) - 1
            grad   = torch.autograd.grad(target, x, retain_graph=keep)[0].abs()[0].cpu().numpy()
            gradients.append((family, grad))

        return gradients

    def _cell_png(self, cell: np.ndarray) -> str:
        buf = io.BytesIO()
        plt.imsave(buf, cell, cmap="magma", vmin=0.0, vmax=1.0, format="png")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _family_payload(self, family: str, grad: np.ndarray) -> dict:
        total = float(grad.sum())
        if total <= 0.0:
            return {"family": family, "dead": True, "shares": [0.0] * grad.shape[0], "cells": [None] * grad.shape[0]}

        shares = grad.sum(axis=(1, 2)) / total

        cells = []
        for channel in range(grad.shape[0]):
            peak = float(grad[channel].max())
            cells.append(self._cell_png(grad[channel] / peak) if peak > 0.0 else None)

        return {"family": family, "dead": False, "shares": [float(v) for v in shares], "cells": cells}

    def attribution(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                az, rg         = int(body["az"]), int(body["rg"])
                window, cy, cx = self._window(az, rg)

                families = [self._family_payload(family, grad) for family, grad in self._family_gradients(window, cy, cx)]

                return {
                    "ok"       : True,
                    "channels" : self.loaded["labels"],
                    "center"   : [cy, cx],
                    "patch"    : [int(window.shape[2]), int(window.shape[3])],
                    "families" : families,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def _perturb(self, window: np.ndarray, perturbation: dict) -> np.ndarray:
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
            raise ValueError(f"unknown perturbation '{kind}', expected one of {self.PERTURBATIONS}")

        return perturbed

    def whatif(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                az, rg         = int(body["az"]), int(body["rg"])
                window, cy, cx = self._window(az, rg)

                base_slots, base_curve = self._predict_window(window, cy, cx)

                perturbed                          = self._perturb(window, body.get("perturbation", {}))
                perturbed_slots, perturbed_curve   = self._predict_window(perturbed, cy, cx)

                base      = np.asarray(base_curve)
                shifted   = np.asarray(perturbed_curve)
                delta_mse = float(((base - shifted) ** 2).mean())

                return {
                    "ok"              : True,
                    "az"              : az,
                    "rg"              : rg,
                    "x_axis"          : [float(v) for v in self.loaded["run"].x_axis],
                    "base_slots"      : base_slots,
                    "base_curve"      : base_curve,
                    "perturbed_slots" : perturbed_slots,
                    "perturbed_curve" : perturbed_curve,
                    "delta_mse"       : delta_mse,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def features_png(self, az: int, rg: int, layer: str, max_channels: int = 16) -> bytes | None:
        import torch

        from tools.diagnostics.activation_recorder import ActivationRecorder

        with self.lock:
            if self.loaded is None or layer not in self.loaded["layers"]:
                return None

            window, _cy, _cx = self._window(az, rg)

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

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)

        return buf.getvalue()
