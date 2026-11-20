
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from catalog_roots import CatalogRoots, RunScanner
from model_probe   import ModelProbe
from web_logger    import WebLogger


class SurveyCancelled(Exception):
    pass


class ModelSurvey:

    LOADER_NAME  = "model_survey"
    NOISE_SIGMAS = tuple(float(v) for v in np.linspace(0.0, 2.0, 9))
    SPLITS       = ("train", "val", "test")
    DEVICES      = ("cpu", "cuda")
    ERF_SAMPLES  = 24
    TILE_CAP     = 64
    CPU_THREADS  = 64
    DISTANCE_BIN = 4.0

    def __init__(self, logger: WebLogger) -> None:
        self.logger  = logger
        self.lock    = threading.Lock()
        self.scanner = RunScanner(CatalogRoots())

        self.loaded      = None
        self.result      = None
        self.cancel_flag = False
        self.tile_cap    = self.TILE_CAP
        self.erf_samples = self.ERF_SAMPLES
        self.cpu_threads = self.CPU_THREADS
        self.total_units = 1
        self.done_units  = 0
        self.status      = {"state": "idle", "path": "", "progress": 0.0, "stage": "", "error": ""}

    def runs(self, base: str) -> dict:
        scanned = self.scanner.checkpoint_runs(base, ModelProbe.CHECKPOINT, ModelProbe.CONFIG_NAME)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "runs": []}

        return {"ok": True, "root": scanned["root"], "runs": scanned["entries"]}

    def _reject_config(self, body: dict) -> str:
        split   = body.get("split", "test")
        device  = body.get("device", "cpu")
        tiles   = int(body.get("tiles", self.TILE_CAP))
        erf     = int(body.get("erf_samples", self.ERF_SAMPLES))
        threads = int(body.get("threads", self.CPU_THREADS))

        if split not in self.SPLITS:
            return f"split '{split}' is unknown; pick one of {self.SPLITS}"
        if device not in self.DEVICES:
            return f"device '{device}' is unknown; pick one of {self.DEVICES}"
        if tiles < 0:
            return f"tile cap {tiles} is negative; use 0 to survey every tile"
        if erf < 0:
            return f"{erf} receptive-field samples is negative; use 0 to skip the sampling phase"
        if threads < 1:
            return f"{threads} CPU threads leaves nothing to compute with; use at least 1"

        return ""

    def start(self, body: dict) -> dict:
        run_path = body.get("path", "")
        refusal  = ModelProbe._reject_unloadable(run_path, self.scanner.roots) or self._reject_config(body)
        if refusal:
            return {"ok": False, "error": refusal}

        with self.lock:
            if self.status["state"] == "running":
                return {"ok": False, "error": "a survey is already running; cancel it first"}

            self.result      = None
            self.cancel_flag = False
            self.tile_cap    = int(body.get("tiles", self.TILE_CAP))
            self.erf_samples = int(body.get("erf_samples", self.ERF_SAMPLES))
            self.cpu_threads = int(body.get("threads", self.CPU_THREADS))
            self.status      = {"state": "running", "path": run_path, "progress": 0.0, "stage": "opening run", "error": ""}

        args = (run_path, body.get("split", "test"), body.get("device", "cpu"))
        threading.Thread(target=self._worker, args=args, name="ModelSurvey", daemon=True).start()
        return {"ok": True}

    def cancel(self) -> dict:
        with self.lock:
            if self.status["state"] != "running":
                return {"ok": False, "error": "no survey is running"}
            self.cancel_flag = True
        return {"ok": True}

    def survey_status(self) -> dict:
        with self.lock:
            return dict(self.status)

    def survey_result(self) -> dict:
        with self.lock:
            if self.result is None:
                return {"ok": False, "error": "no finished survey; start one and wait for it to complete"}
            return {"ok": True, "result": self.result}

    def _set(self, progress: float, stage: str) -> None:
        with self.lock:
            self.status["progress"] = progress
            self.status["stage"]    = stage

    def _tick(self, stage: str) -> None:
        with self.lock:
            if self.cancel_flag:
                raise SurveyCancelled()

        self.done_units += 1
        self._set(min(1.0, self.done_units / self.total_units), stage)

    def _load(self, run_path: str, split: str, device: str) -> None:
        from pipelines.backbone.inference.loader import RunLoader
        from tools.monitoring.logger             import Logger

        self._set(0.0, "loading checkpoint and dataset")

        run = RunLoader(Path(run_path), logger=Logger(log_dir="", name=self.LOADER_NAME)).load(
            split           = split,
            batch_size      = 1,
            num_workers     = 0,
            device          = device,
            checkpoint_name = ModelProbe.CHECKPOINT,
        )

        self.loaded = {
            "run"      : run,
            "path"     : run_path,
            "split"    : split,
            "device"   : device,
            "labels"   : run.input_channels,
            "targets"  : run.target_channels,
            "patch"    : tuple(run.dataset_config.patch.size),
            "backbone" : run.backbone_name,
        }

    def _tile_origins(self) -> tuple[list, int, int, int]:
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]
        n_rows   = run.split_region.row_size
        n_cols   = run.split_region.col_size

        if n_rows < ph or n_cols < pw:
            raise ValueError(f"the {n_rows}x{n_cols} region is smaller than the {ph}x{pw} model window; nothing to survey")

        row_starts = list(range(0, n_rows - ph + 1, ph))
        col_starts = list(range(0, n_cols - pw + 1, pw))
        origins    = [(top, left) for top in row_starts for left in col_starts]

        selected = origins
        if 0 < self.tile_cap < len(origins):
            indices  = np.unique(np.linspace(0, len(origins) - 1, self.tile_cap).round().astype(int))
            selected = [origins[index] for index in indices]

        return selected, len(origins), len(row_starts) * ph, len(col_starts) * pw

    def _tile_window(self, top: int, left: int) -> np.ndarray:
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]

        return run.dataset.window(run.inputs[:, top:top + ph, left:left + pw])[None].astype(np.float32)

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        return ModelProbe._outputs(self.loaded["run"], self.loaded["device"], batch)

    @staticmethod
    def _shifts(outputs: np.ndarray) -> np.ndarray:
        return ((outputs - outputs[:1]) ** 2).mean(axis=1)

    def _base_pass(self, acc: dict, window: np.ndarray, top: int, left: int) -> None:
        run         = self.loaded["run"]
        ph, pw      = self.loaded["patch"]
        predictions = run.model(window)[0].astype(np.float64)
        targets     = np.asarray(run.targets[:, top:top + ph, left:left + pw], dtype=np.float64)
        error       = predictions - targets

        acc["sse"]   += (error ** 2).sum(axis=(1, 2))
        acc["sae"]   += np.abs(error).sum(axis=(1, 2))
        acc["sum"]   += targets.sum(axis=(1, 2))
        acc["sum2"]  += (targets ** 2).sum(axis=(1, 2))
        acc["count"] += ph * pw

    def _gradient_pass(self, acc: dict, window: np.ndarray) -> None:
        import torch

        run     = self.loaded["run"]
        targets = self.loaded["targets"]

        x = torch.from_numpy(window).to(self.loaded["device"]).requires_grad_(True)
        y = run.model.module(x)

        for index, name in enumerate(targets):
            keep = index < len(targets) - 1
            grad = torch.autograd.grad(y[0, index].abs().sum(), x, retain_graph=keep)[0].abs()[0].cpu().numpy()

            acc["grad"][name] += grad.sum(axis=(1, 2))

    def _ablation_pass(self, acc: dict, window: np.ndarray) -> None:
        n_channels = window.shape[1]

        batch = np.repeat(window, n_channels + 1, axis=0)
        for channel in range(n_channels):
            batch[channel + 1, channel] = 0.0

        shifts = self._shifts(self._forward(batch))

        for channel in range(n_channels):
            acc["abl_delta"][channel] += float(shifts[channel + 1].sum())
        acc["abl_pixels"] += shifts.shape[1] * shifts.shape[2]

    def _flip_pass(self, acc: dict, window: np.ndarray) -> None:
        batch   = np.concatenate([np.flip(window, axes) if axes else window for _, axes in ModelProbe.FLIPS], axis=0)
        outputs = self._forward(batch)

        restored = np.concatenate([np.flip(outputs[index:index + 1], axes) if axes else outputs[index:index + 1] for index, (_, axes) in enumerate(ModelProbe.FLIPS)], axis=0)
        shifts   = self._shifts(restored)

        for index, (name, _) in enumerate(ModelProbe.FLIPS):
            acc["flip_delta"][name] += float(shifts[index].sum())
        acc["flip_pixels"] += shifts.shape[1] * shifts.shape[2]

    def _noise_pass(self, acc: dict, window: np.ndarray, tile_index: int) -> None:
        variants = [window + np.random.default_rng(tile_index * 100 + index).normal(0.0, sigma, size=window.shape).astype(np.float32) for index, sigma in enumerate(self.NOISE_SIGMAS)]
        shifts   = self._shifts(self._forward(np.concatenate([window] + variants, axis=0)))

        for index in range(len(self.NOISE_SIGMAS)):
            acc["noise_delta"][index] += float(shifts[index + 1].sum())
        acc["noise_pixels"] += shifts.shape[1] * shifts.shape[2]

    def _occlusion_grid(self) -> tuple[int, int, int, int]:
        ph, pw = self.loaded["patch"]
        oh, ow = max(2, ph // 8), max(2, pw // 8)

        return oh, ow, int(np.ceil(ph / oh)), int(np.ceil(pw / ow))

    def _occlusion_distances(self) -> np.ndarray:
        ph, pw               = self.loaded["patch"]
        oh, ow, rows, cols   = self._occlusion_grid()

        yy, xx    = np.mgrid[0:ph, 0:pw]
        distances = np.empty((rows * cols, ph, pw), dtype=np.float32)

        for row in range(rows):
            for col in range(cols):
                cy = (row + 0.5) * oh
                cx = (col + 0.5) * ow
                distances[row * cols + col] = np.hypot(yy + 0.5 - cy, xx + 0.5 - cx)

        return distances

    def _occlusion_pass(self, acc: dict, window: np.ndarray, distances: np.ndarray) -> None:
        ph, pw             = self.loaded["patch"]
        oh, ow, rows, cols = self._occlusion_grid()

        batch = np.repeat(window, rows * cols + 1, axis=0)
        for row in range(rows):
            for col in range(cols):
                cell = 1 + row * cols + col
                batch[cell, :, row * oh:min(ph, (row + 1) * oh), col * ow:min(pw, (col + 1) * ow)] = 0.0

        deltas = self._shifts(self._forward(batch))[1:]
        bins   = (distances / self.DISTANCE_BIN).astype(np.int64)

        np.add.at(acc["occl_sum"], bins.ravel(), deltas.ravel().astype(np.float64))
        np.add.at(acc["occl_count"], bins.ravel(), 1)

    def _vitals_phase(self, tiles: list) -> dict:
        import torch

        from tools.diagnostics.activation_recorder import ActivationRecorder

        model    = self.loaded["run"].model.module
        recorder = ActivationRecorder(model)
        recorder.attach_stats()

        try:
            for index, (top, left) in enumerate(tiles):
                self._tick(f"layer vitals, tile {index + 1}/{len(tiles)}")
                with torch.no_grad():
                    model(torch.from_numpy(self._tile_window(top, left)).to(self.loaded["device"]))
        finally:
            recorder.detach()

        return ModelProbe._vitals_payload(model, recorder.stats())

    def _gradient_map_payload(self, maps: dict, live: dict, center: list | None) -> dict:
        ph, pw     = self.loaded["patch"]
        n_channels = len(self.loaded["labels"])

        targets = []
        for name in self.loaded["targets"]:
            if maps[name] is None:
                targets.append({"name": name, "dead": True, "shares": [0.0] * n_channels, "cells": [None] * n_channels})
                continue

            mean   = maps[name] / live[name]
            shares = mean.sum(axis=(1, 2)) / mean.sum()

            cells = []
            for channel in range(mean.shape[0]):
                peak = float(mean[channel].max())
                cells.append(ModelProbe._cell_png(mean[channel] / peak) if peak > 0.0 else None)

            targets.append({"name": name, "dead": False, "shares": [float(v) for v in shares], "cells": cells})

        return {
            "targets" : targets,
            "center"  : center if center is not None else [ph // 2, pw // 2],
            "patch"   : [ph, pw],
            "samples" : self.erf_samples,
        }

    def _sampling_phase(self, coverage_rows: int, coverage_cols: int) -> tuple[dict, dict]:
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]
        names  = self.loaded["targets"]
        rng    = np.random.default_rng(0)

        row_lo, row_hi = ph // 2, coverage_rows - ph + ph // 2
        col_lo, col_hi = pw // 2, coverage_cols - pw + pw // 2

        cumulative = {name: None for name in names}
        maps       = {name: None for name in names}
        live       = {name: 0 for name in names}
        radii      = None
        center     = None

        for index in range(self.erf_samples):
            self._tick(f"sampled gradients, sample {index + 1}/{self.erf_samples}")

            row = int(rng.integers(row_lo, max(row_lo + 1, row_hi + 1)))
            col = int(rng.integers(col_lo, max(col_lo + 1, col_hi + 1)))

            window, cy, cx = ModelProbe._window_at(run, (ph, pw), row, col)
            center         = [cy, cx]

            for name, grad in ModelProbe._target_gradients_at(run, self.loaded["device"], window, cy, cx):
                if float(grad.sum()) <= 0.0:
                    continue
                profile = ModelProbe._radial_profile(grad, cy, cx)
                radii   = profile["radii"]
                arc     = np.asarray(profile["cumulative"])

                cumulative[name] = arc if cumulative[name] is None else cumulative[name] + arc
                maps[name]       = grad if maps[name] is None else maps[name] + grad
                live[name]      += 1

        targets = []
        for name in names:
            if cumulative[name] is None:
                targets.append({"name": name, "dead": True, "radii": None, "cumulative": None, "r50": None, "r90": None, "samples": 0})
                continue

            mean = cumulative[name] / live[name]
            targets.append({
                "name"       : name,
                "dead"       : False,
                "radii"      : radii,
                "cumulative" : [float(v) for v in mean],
                "r50"        : int(radii[int(np.argmax(mean >= 0.5))]),
                "r90"        : int(radii[int(np.argmax(mean >= 0.9))]),
                "samples"    : live[name],
            })

        return {"targets": targets, "samples": self.erf_samples}, self._gradient_map_payload(maps, live, center)

    def _fit(self, acc: dict) -> list[dict]:
        mean = acc["sum"] / acc["count"]
        sst  = acc["sum2"] - acc["count"] * mean ** 2

        return [
            {
                "name" : name,
                "rmse" : float(np.sqrt(acc["sse"][index] / acc["count"])),
                "mae"  : float(acc["sae"][index] / acc["count"]),
                "r2"   : float(1.0 - acc["sse"][index] / sst[index]) if sst[index] > 0.0 else None,
            }
            for index, name in enumerate(self.loaded["targets"])
        ]

    def _finalize(self, acc: dict, tiles: list, total_tiles: int, vitals: dict, erf: dict, gradients: dict, seconds: float) -> dict:
        run    = self.loaded["run"]
        labels = self.loaded["labels"]
        pixels = len(tiles) * self.loaded["patch"][0] * self.loaded["patch"][1]

        targets = []
        for name in self.loaded["targets"]:
            total = float(acc["grad"][name].sum())
            dead  = total <= 0.0
            targets.append({
                "name"   : name,
                "dead"   : dead,
                "shares" : [0.0] * len(labels) if dead else [float(v) for v in acc["grad"][name] / total],
            })

        channels = [{"channel": channel, "label": labels[channel], "delta": acc["abl_delta"][channel] / acc["abl_pixels"]} for channel in range(len(labels))]
        channels.sort(key=lambda entry: entry["delta"], reverse=True)

        flips = [{"flip": name, "delta": acc["flip_delta"][name] / acc["flip_pixels"]} for name, _ in ModelProbe.FLIPS]
        noise = [{"value": sigma, "delta": acc["noise_delta"][index] / acc["noise_pixels"]} for index, sigma in enumerate(self.NOISE_SIGMAS)]

        oh, ow, _, _ = self._occlusion_grid()
        occupied     = acc["occl_count"] > 0
        occlusion    = {
            "occluder" : [oh, ow],
            "distance" : [float((index + 0.5) * self.DISTANCE_BIN) for index in np.nonzero(occupied)[0]],
            "delta"    : [float(acc["occl_sum"][index] / acc["occl_count"][index]) for index in np.nonzero(occupied)[0]],
        }

        return {
            "run"         : self.loaded["path"],
            "backbone"    : self.loaded["backbone"],
            "split"       : self.loaded["split"],
            "patch"       : list(self.loaded["patch"]),
            "channels"    : labels,
            "targets"     : self.loaded["targets"],
            "region"      : [run.split_region.row_size, run.split_region.col_size],
            "coverage"    : {"tiles": len(tiles), "total_tiles": total_tiles, "pixels": pixels},
            "seconds"     : seconds,
            "fit"         : self._fit(acc),
            "attribution" : {"targets": targets},
            "erf"         : erf,
            "gradients"   : gradients,
            "ablation"    : {"channels": channels},
            "flips"       : flips,
            "noise"       : noise,
            "occlusion"   : occlusion,
            "vitals"      : vitals,
        }

    def _survey(self) -> None:
        started                                         = time.monotonic()
        tiles, total_tiles, coverage_rows, coverage_cols = self._tile_origins()

        n_channels = int(self.loaded["run"].in_channels)
        n_targets  = len(self.loaded["targets"])
        max_bin    = int(np.hypot(*self.loaded["patch"]) / self.DISTANCE_BIN) + 2

        self.done_units  = 0
        self.total_units = len(tiles) * 7 + self.erf_samples

        acc = {
            "sse"           : np.zeros(n_targets, dtype=np.float64),
            "sae"           : np.zeros(n_targets, dtype=np.float64),
            "sum"           : np.zeros(n_targets, dtype=np.float64),
            "sum2"          : np.zeros(n_targets, dtype=np.float64),
            "count"         : 0,
            "grad"          : {name: np.zeros(n_channels, dtype=np.float64) for name in self.loaded["targets"]},
            "abl_delta"     : np.zeros(n_channels, dtype=np.float64),
            "abl_pixels"    : 0,
            "flip_delta"    : {name: 0.0 for name, _ in ModelProbe.FLIPS},
            "flip_pixels"   : 0,
            "noise_delta"   : np.zeros(len(self.NOISE_SIGMAS), dtype=np.float64),
            "noise_pixels"  : 0,
            "occl_sum"      : np.zeros(max_bin, dtype=np.float64),
            "occl_count"    : np.zeros(max_bin, dtype=np.int64),
        }

        distances = self._occlusion_distances()

        try:
            for index, (top, left) in enumerate(tiles):
                label  = f"tile {index + 1}/{len(tiles)}"
                window = self._tile_window(top, left)

                self._tick(f"prediction fit, {label}")
                self._base_pass(acc, window, top, left)

                self._tick(f"gradient attribution, {label}")
                self._gradient_pass(acc, window)

                self._tick(f"channel ablation, {label}")
                self._ablation_pass(acc, window)

                self._tick(f"flip symmetry, {label}")
                self._flip_pass(acc, window)

                self._tick(f"noise robustness, {label}")
                self._noise_pass(acc, window, index)

                self._tick(f"occlusion, {label}")
                self._occlusion_pass(acc, window, distances)

            vitals         = self._vitals_phase(tiles)
            erf, gradients = self._sampling_phase(coverage_rows, coverage_cols)
            result         = self._finalize(acc, tiles, total_tiles, vitals, erf, gradients, time.monotonic() - started)

            with self.lock:
                self.result = result
                self.status = {"state": "done", "path": self.loaded["path"], "progress": 1.0, "stage": "done", "error": ""}

        except SurveyCancelled:
            with self.lock:
                self.result = None
                self.status = {"state": "cancelled", "path": self.loaded["path"], "progress": 0.0, "stage": "", "error": ""}

    def _worker(self, run_path: str, split: str, device: str) -> None:
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(min(self.cpu_threads, previous))

        try:
            self._load(run_path, split, device)
            self._survey()
            self.logger.info(f"model survey finished for {run_path} ({self.status['state']})")

        except Exception as error:
            with self.lock:
                self.result = None
                self.status = {"state": "error", "path": run_path, "progress": 0.0, "stage": "", "error": f"{type(error).__name__}: {error}"}
            self.logger.error(f"model survey failed: {error}")

        finally:
            torch.set_num_threads(previous)
            self.loaded = None
