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
            from pipelines.backbone.inference.loader import RunLoader
            from tools.monitoring.logger             import Logger

            self._set_load(0.1, "loading checkpoint and dataset")

            run = RunLoader(Path(run_path), logger=Logger(log_dir="", name=self.LOADER_NAME)).load(
                split           = split,
                batch_size      = 1,
                num_workers     = 0,
                device          = device,
                checkpoint_name = self.CHECKPOINT,
            )

            self._set_load(0.8, "indexing layers")

            layers  = self._probe_layers(run.model.module)
            modules = dict(run.model.module.named_modules())

            info = {
                "run"             : run_path,
                "backbone"        : run.backbone_name,
                "split"           : split,
                "input_channels"  : run.input_channels,
                "target_channels" : run.target_channels,
                "row_size"        : run.split_region.row_size,
                "col_size"        : run.split_region.col_size,
                "row_offset"      : run.split_region.row_start,
                "col_offset"      : run.split_region.col_start,
                "patch"           : list(run.dataset_config.patch.size),
                "n_layers"        : len(layers),
            }

            with self.lock:
                self.loaded = {
                    "run"    : run,
                    "layers" : layers,
                    "types"  : {name: type(modules[name]).__name__ for name in layers},
                    "patch"  : tuple(run.dataset_config.patch.size),
                    "device" : device,
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
            target = np.asarray(self.loaded["run"].targets[0], dtype=np.float32)

        vmin, vmax = PlotBase._shared_clim(target)

        buf = io.BytesIO()
        plt.imsave(buf, target, cmap="viridis", vmin=vmin, vmax=vmax, format="png")
        return buf.getvalue()

    @classmethod
    def _window_at(cls, run, patch: tuple, row: int, col: int) -> tuple[np.ndarray, int, int]:
        ph, pw = patch
        n_rows = run.split_region.row_size
        n_cols = run.split_region.col_size

        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise ValueError(f"pixel ({row}, {col}) is outside the {n_rows}x{n_cols} region")

        top  = int(np.clip(row - ph // 2, 0, n_rows - ph))
        left = int(np.clip(col - pw // 2, 0, n_cols - pw))

        window = run.dataset.window(run.inputs[:, top:top + ph, left:left + pw])
        return window[None].astype(np.float32), row - top, col - left

    def _window(self, row: int, col: int) -> tuple[np.ndarray, int, int]:
        return self._window_at(self.loaded["run"], self.loaded["patch"], row, col)

    @staticmethod
    def _outputs(run, device: str, batch: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return run.model.module(torch.from_numpy(np.ascontiguousarray(batch)).to(device)).cpu().numpy()

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        return self._outputs(self.loaded["run"], self.loaded["device"], batch)

    @staticmethod
    def _center_shift(outputs: np.ndarray, cy: int, cx: int) -> np.ndarray:
        center = outputs[:, :, cy, cx]
        return ((center - center[:1]) ** 2).mean(axis=1)

    def predict(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col       = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run        = self.loaded["run"]
                prediction = run.model(window)[0, :, cy, cx]
                target     = np.asarray(run.targets[:, row, col], dtype=np.float64)

                return {
                    "ok"      : True,
                    "row"     : row,
                    "col"     : col,
                    "targets" : [{"name": name, "prediction": float(prediction[index]), "target": float(target[index]), "error": float(prediction[index] - target[index])} for index, name in enumerate(run.target_channels)],
                    "inputs"  : [{"name": name, "value": float(run.inputs[index, row, col])} for index, name in enumerate(run.input_channels)],
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

    def fields(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col       = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run         = self.loaded["run"]
                predictions = run.model(window)[0]
                ph, pw      = predictions.shape[1], predictions.shape[2]
                top, left   = row - cy, col - cx
                targets     = np.asarray(run.targets[:, top:top + ph, left:left + pw], dtype=np.float32)

                fields = []
                for index, name in enumerate(run.target_channels):
                    lo, hi = float(min(targets[index].min(), predictions[index].min())), float(max(targets[index].max(), predictions[index].max()))
                    fields.append({
                        "name"       : name,
                        "prediction" : self._field_payload(predictions[index], "viridis", lo, hi),
                        "target"     : self._field_payload(targets[index], "viridis", lo, hi),
                        "error"      : self._field_payload(np.abs(predictions[index] - targets[index]), "inferno", vmin=0.0),
                    })

                return {"ok": True, "center": [cy, cx], "patch": [ph, pw], "targets": fields}
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    @classmethod
    def _target_gradients_at(cls, run, device: str, window: np.ndarray, cy: int, cx: int) -> list[tuple[str, np.ndarray]]:
        import torch

        x = torch.from_numpy(window).to(device).requires_grad_(True)
        y = run.model.module(x)

        gradients = []
        for index, name in enumerate(run.target_channels):
            keep = index < len(run.target_channels) - 1
            grad = torch.autograd.grad(y[0, index, cy, cx], x, retain_graph=keep)[0].abs()[0].cpu().numpy()
            gradients.append((name, grad))

        return gradients

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

    @classmethod
    def _target_payload(cls, name: str, grad: np.ndarray, cy: int, cx: int) -> dict:
        total = float(grad.sum())
        if total <= 0.0:
            return {"name": name, "dead": True, "shares": [0.0] * grad.shape[0], "cells": [None] * grad.shape[0], "radial": None}

        cells = []
        for channel in range(grad.shape[0]):
            peak = float(grad[channel].max())
            cells.append(cls._cell_png(grad[channel] / peak) if peak > 0.0 else None)

        return {
            "name"   : name,
            "dead"   : False,
            "shares" : [float(v) for v in grad.sum(axis=(1, 2)) / total],
            "cells"  : cells,
            "radial" : cls._radial_profile(grad, cy, cx),
        }

    def attribution(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col       = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run     = self.loaded["run"]
                targets = [self._target_payload(name, grad, cy, cx) for name, grad in self._target_gradients_at(run, self.loaded["device"], window, cy, cx)]

                return {
                    "ok"       : True,
                    "channels" : run.input_channels,
                    "center"   : [cy, cx],
                    "patch"    : [int(window.shape[2]), int(window.shape[3])],
                    "targets"  : targets,
                }
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def ablation(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col       = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run        = self.loaded["run"]
                n_channels = window.shape[1]

                batch = np.repeat(window, n_channels + 1, axis=0)
                for channel in range(n_channels):
                    batch[channel + 1, channel] = 0.0

                shifts = self._center_shift(self._forward(batch), cy, cx)

                channels = [{"channel": channel, "label": run.input_channels[channel], "delta": float(shifts[channel + 1])} for channel in range(n_channels)]
                channels.sort(key=lambda entry: entry["delta"], reverse=True)

                return {"ok": True, "channels": channels}
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def flips(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col       = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                run     = self.loaded["run"]
                batch   = np.concatenate([np.flip(window, axes) if axes else window for _, axes in self.FLIPS], axis=0)
                outputs = self._forward(batch)

                restored = np.concatenate([np.flip(outputs[index:index + 1], axes) if axes else outputs[index:index + 1] for index, (_, axes) in enumerate(self.FLIPS)], axis=0)
                shifts   = self._center_shift(restored, cy, cx)
                values   = run.model.denormalize_target(restored[:, :, cy, cx][:, :, None, None])[:, :, 0, 0]

                entries = [{"flip": name, "values": [float(v) for v in values[index]], "delta": float(shifts[index])} for index, (name, _) in enumerate(self.FLIPS)]

                return {"ok": True, "channels": run.target_channels, "entries": entries}
            except (ValueError, KeyError) as error:
                return {"ok": False, "error": str(error)}

    def occlusion(self, body: dict) -> dict:
        with self.lock:
            if self.loaded is None:
                return {"ok": False, "error": "no model loaded"}

            try:
                row, col       = int(body["row"]), int(body["col"])
                window, cy, cx = self._window(row, col)

                ph, pw = window.shape[2], window.shape[3]
                oh, ow = max(2, ph // 8), max(2, pw // 8)
                rows   = int(np.ceil(ph / oh))
                cols   = int(np.ceil(pw / ow))

                batch = np.repeat(window, rows * cols + 1, axis=0)
                for cell_row in range(rows):
                    for cell_col in range(cols):
                        cell = 1 + cell_row * cols + cell_col
                        batch[cell, :, cell_row * oh:min(ph, (cell_row + 1) * oh), cell_col * ow:min(pw, (cell_col + 1) * ow)] = 0.0

                deltas = self._center_shift(self._forward(batch), cy, cx)[1:].reshape(rows, cols)
                worst  = np.unravel_index(int(np.argmax(deltas)), deltas.shape)

                return {
                    "ok"          : True,
                    "center"      : [cy, cx],
                    "patch"       : [ph, pw],
                    "occluder"    : [oh, ow],
                    "grid"        : [rows, cols],
                    "center_cell" : [cy // oh, cx // ow],
                    "worst"       : {"row": int(worst[0]), "col": int(worst[1]), "delta": float(deltas[worst])},
                    "map"         : self._field_payload(deltas, "magma"),
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
                row, col         = int(body["row"]), int(body["col"])
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
        ax.set_title(f"1×1 convolution, {matrix.shape[0]}×{matrix.shape[1]} weight matrix, ±{peak:.2g}", fontsize=8)

        return self._figure_png(fig)

    def kernels_png(self, layer: str, max_out: int = 12, max_in: int = 8) -> bytes | None:
        with self.lock:
            if self.loaded is None or layer not in self.loaded["layers"]:
                return None
            module = dict(self.loaded["run"].model.module.named_modules())[layer]

        w = module.weight.detach().cpu().numpy()
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

        fig.suptitle(f"{out_ch}×{in_ch}×{kh}×{kw} weight, strongest {len(order_out)}×{len(order_in)} kernels, ±{peak:.2g}", fontsize=8)

        return self._figure_png(fig)
