"""Whole-run model survey: fit quality, attribution, robustness and layer vitals.

Loads a trained checkpoint with its dataset, tiles the split region, and runs
seven channels per tile (prediction fit, gradient attribution, channel ablation,
flip symmetry, noise robustness, occlusion) plus layer vitals and a sampled
effective-receptive-field phase, then reduces everything to one JSON payload for
the web UI. The survey runs on a background thread with cooperative
cancellation and coarse progress reporting.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from catalog_roots import CatalogRoots, RunScanner
from model_probe   import ModelProbe
from web_logger    import WebLogger

from pipelines.shared.inference.run_classifier import RunArtifacts


class SurveyCancelled(Exception):
    """Raised inside the worker thread when the user cancels a running survey."""
    pass


class ModelSurvey:
    """Runs and serves a full diagnostic survey of one trained run.

    Only one survey runs at a time; start spawns the worker thread, cancel sets
    a flag the per-unit tick honours, and the finished payload is held until the
    next survey replaces it.

    Attributes:
        logger      : Console logger for completion and failure messages.
        lock        : Guards the status, result and cancellation flag.
        scanner     : Scanner listing checkpointed runs under the catalog roots.
        loaded      : Loaded run bundle (model, dataset, labels, renderer, patch size) while a survey is in flight, None otherwise.
        result      : Payload of the last finished survey, or None.
        cancel_flag : Set by cancel to stop the worker at the next tick.
        tile_cap    : Maximum tiles to survey; 0 surveys every tile.
        erf_samples : Number of random windows in the sampled-gradient phase.
        cpu_threads : Torch intra-op thread ceiling for the worker.
        total_units : Total progress units for the current survey.
        done_units  : Progress units completed so far.
        status      : Public state with the run path, progress fraction, stage and error.
    """

    LOADER_NAME  = "model_survey"
    NOISE_SIGMAS = tuple(float(v) for v in np.linspace(0.0, 2.0, 9))
    SPLITS       = ("train", "val", "test")
    DEVICES      = ("cpu", "cuda")
    ERF_SAMPLES  = 24
    TILE_CAP     = 64
    CPU_THREADS  = 64
    CHUNK_BUDGET = 4_000_000
    DISTANCE_BIN = 4.0

    def __init__(self, logger: WebLogger) -> None:
        """Prepares an idle survey with default tile, sample and thread budgets.

        Args:
            logger : Console logger for survey completion and failure messages.
        """
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
        """Lists the surveyable runs under a catalog root.

        Args:
            base : Catalog root key to scan.

        Returns:
            Mapping with "ok", the resolved "root" and the "runs" entries that
            carry both a checkpoint and a config, or "ok" False with an "error".
        """
        scanned = self.scanner.checkpoint_runs(base, ModelProbe.CHECKPOINT, ModelProbe.CONFIG_NAMES)
        if not scanned["ok"]:
            return {"ok": False, "error": scanned["error"], "runs": []}

        return {"ok": True, "root": scanned["root"], "runs": scanned["entries"]}

    def _reject_config(self, body: dict) -> str:
        """Returns why the requested survey settings are unusable, or an empty string.

        Args:
            body : Request body with the split, device, tile cap, receptive-field sample count and CPU thread count.
        """
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
        """Validates the request and launches the survey worker thread.

        Args:
            body : Request body with the run "path", "split", "device", "tiles", "erf_samples" and "threads".

        Returns:
            Mapping with "ok" True once the worker started, or "ok" False with
            an "error" when the run cannot be loaded, the settings are invalid,
            or a survey is already running.
        """
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
        """Asks the running survey to stop at its next progress tick.

        Returns:
            Mapping with "ok" True once the flag is set, or "ok" False with an
            "error" when no survey is running.
        """
        with self.lock:
            if self.status["state"] != "running":
                return {"ok": False, "error": "no survey is running"}
            self.cancel_flag = True
        return {"ok": True}

    def survey_status(self) -> dict:
        """Returns a snapshot of the survey state, progress fraction, stage and error."""
        with self.lock:
            return dict(self.status)

    def survey_result(self) -> dict:
        """Returns the payload of the last finished survey, or an error when there is none."""
        with self.lock:
            if self.result is None:
                return {"ok": False, "error": "no finished survey; start one and wait for it to complete"}
            return {"ok": True, "result": self.result}

    def _set(self, progress: float, stage: str) -> None:
        """Publishes a progress fraction in [0, 1] and a human-readable stage label."""
        with self.lock:
            self.status["progress"] = progress
            self.status["stage"]    = stage

    def _tick(self, stage: str) -> None:
        """Counts one completed progress unit and publishes the new stage.

        Args:
            stage : Label describing the work that just completed.

        Raises:
            SurveyCancelled : If cancel was requested since the previous tick.
        """
        with self.lock:
            if self.cancel_flag:
                raise SurveyCancelled()

        self.done_units += 1
        self._set(min(1.0, self.done_units / self.total_units), stage)

    def _load(self, run_path: str, split: str, device: str) -> None:
        """Loads the checkpoint, dataset and rendering helpers into the loaded bundle.

        Args:
            run_path : Directory of the trained run; a dual config in its meta directory selects the dual loader.
            split    : Dataset split to survey ("train", "val" or "test").
            device   : Torch device the model and windows are placed on.
        """
        from pipelines.backbone.inference.analysis.input_attribution import ChannelLabeler
        from pipelines.backbone.inference.probes            import PredictionCurves
        from tools.monitoring.logger                        import Logger

        self._set(0.0, "loading checkpoint and dataset")

        is_dual = (Path(run_path) / "meta" / RunArtifacts.DUAL_CONFIG).is_file()
        run     = ModelProbe._loader_class(is_dual)(Path(run_path), logger=Logger(log_dir="", name=self.LOADER_NAME)).load(
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
            "labels"   : ChannelLabeler.build(run),
            "renderer" : PredictionCurves(run.n_gaussians, run.z_axis),
            "patch"    : tuple(run.dataset_config.patch.size),
            "backbone" : ModelProbe._model_label(run, is_dual),
        }

    def _tile_origins(self) -> tuple[list, int, int, int]:
        """Lays non-overlapping model windows over the split region and subsamples them.

        Returns:
            Tuple of the selected (row top, range left) origins, the total
            number of tiles the region holds, and the row and range extents
            in pixels the full tiling covers.

        Raises:
            ValueError : If the split region is smaller than the model window.
        """
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]
        n_rows   = run.split_region.row_size
        n_cols   = run.split_region.col_size

        if n_rows < ph or n_cols < pw:
            raise ValueError(f"the {n_rows}x{n_cols} region is smaller than the {ph}x{pw} model window; nothing to survey")

        az_starts = list(range(0, n_rows - ph + 1, ph))
        rg_starts = list(range(0, n_cols - pw + 1, pw))
        origins   = [(top, left) for top in az_starts for left in rg_starts]

        selected = origins
        if 0 < self.tile_cap < len(origins):
            indices  = np.unique(np.linspace(0, len(origins) - 1, self.tile_cap).round().astype(int))
            selected = [origins[index] for index in indices]

        return selected, len(origins), len(az_starts) * ph, len(rg_starts) * pw

    def _tile_window(self, top: int, left: int) -> np.ndarray:
        """Assembles the normalised model input for one tile.

        Args:
            top  : Row index of the tile's first row.
            left : Range index of the tile's first column.

        Returns:
            Input stack of shape (1, channels, patch_height, patch_width) as
            float32, built from the complex derived-channel window and the Aux
            channel when the dataset carries one.
        """
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]

        complex_window = run.complex_inputs[:, top:top + ph, left:left + pw]
        aux            = run.dataset.aux
        aux_window     = aux[top:top + ph, left:left + pw] if aux is not None else None

        return run.dataset.assemble_window(complex_window, aux_window)[None].astype(np.float32)

    def _row_chunks(self, batch: int, n_rows: int, width: int):
        """Splits a tile's rows into slices whose rendered curves fit the memory budget.

        Args:
            batch  : Number of parameter maps rendered together.
            n_rows : Row rows in the tile.
            width  : Range columns in the tile.

        Yields:
            Half-open (row_start, row_end) row ranges covering the tile.
        """
        run  = self.loaded["run"]
        cost = batch * run.n_gaussians * len(run.z_axis) * width
        rows = max(1, self.CHUNK_BUDGET // max(1, cost))

        for row0 in range(0, n_rows, rows):
            yield row0, min(n_rows, row0 + rows)

    def _render_rows(self, params: np.ndarray, row0: int, row1: int) -> np.ndarray:
        """Renders z curves for a row band of a Gaussian parameter map.

        Args:
            params : Parameter map of shape (batch, 3 * n_gaussians, rows, width), interleaved as amplitude, mean and spread per slot.
            row0   : First row row of the band.
            row1   : Row past the last one in the band.

        Returns:
            Rendered curves of shape (batch, n_z, row1 - row0, width).
        """
        return self.loaded["renderer"].render(params[:, :, row0:row1, :])

    def _base_pass(self, acc: dict, params: np.ndarray, top: int, left: int) -> None:
        """Accumulates fit, detection and matched-parameter statistics for one tile.

        Compares the predicted curves against the raw reference profiles and,
        when the dataset carries ground-truth parameters, against the GT curves,
        the per-pixel active-slot counts, and the Hungarian-matched per-family
        absolute parameter error.

        Args:
            acc    : Accumulator the statistics are appended to in place.
            params : Predicted parameter map of shape (1, 3 * n_gaussians, patch_height, patch_width).
            top    : Row index of the tile's first row.
            left   : Range index of the tile's first column.
        """
        import torch

        from tools.data.gaussians  import GaussianActivity
        from tools.loss.param_loss import ParamMatcher

        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]
        gt     = run.dataset.gt_parameters

        pred_active = GaussianActivity.is_active(params[0, 0::3], run.activity_threshold).sum(axis=0)
        acc["pred_active"].append(pred_active.ravel())

        gt_tile = None
        if gt is not None:
            gt_tile   = np.asarray(gt[:, top:top + ph, left:left + pw], dtype=np.float32)
            gt_active = GaussianActivity.is_active(gt_tile[0::3], run.activity_threshold).sum(axis=0)
            acc["gt_active"].append(gt_active.ravel())
            acc["count_match"].append((pred_active == gt_active).ravel())

            n_k    = run.n_gaussians
            pred_t = torch.from_numpy(params[0].reshape(n_k, 3, ph, pw)[None].copy())
            gt_t   = torch.from_numpy(gt_tile.reshape(n_k, 3, ph, pw)[None].copy())

            matched, _, sorted_gt, _ = ParamMatcher.match(pred_t, pred_t, gt_t, gt_t, active_thr=run.activity_threshold, method=ParamMatcher.HUNGARIAN)

            mask  = GaussianActivity.is_active(sorted_gt[:, :, 0], run.activity_threshold).float()
            error = (matched - sorted_gt).abs()

            for offset, family in enumerate(ModelProbe.FAMILIES):
                acc["matched_err"][family] += float((error[:, :, offset] * mask).sum())
            acc["matched_count"] += float(mask.sum())

        for row0, row1 in self._row_chunks(1, ph, pw):
            pred_curves = self._render_rows(params, row0, row1)
            acc["power"].append((pred_curves ** 2).mean(axis=1)[0].ravel())

            raw = np.asarray(run.full_curves[:, top + row0:top + row1, left:left + pw], dtype=np.float32)
            acc["mse_raw"].append(((pred_curves[0] - raw) ** 2).mean(axis=0).ravel())

            if gt_tile is not None:
                gt_curves = self._render_rows(gt_tile[None], row0, row1)
                acc["mse_gt"].append(((pred_curves - gt_curves) ** 2).mean(axis=1)[0].ravel())

    def _gradient_pass(self, acc: dict, window: np.ndarray) -> None:
        """Accumulates per-channel input-gradient magnitude for each parameter family.

        Args:
            acc    : Accumulator whose "grad" table is incremented per family.
            window : Model input of shape (1, channels, patch_height, patch_width).
        """
        import torch

        run = self.loaded["run"]

        x = torch.from_numpy(window).to(self.loaded["device"]).requires_grad_(True)
        y = run.model.module(x)

        for offset, family in enumerate(ModelProbe.FAMILIES):
            target = y[0, offset::3].abs().sum()
            keep   = offset < len(ModelProbe.FAMILIES) - 1
            grad   = torch.autograd.grad(target, x, retain_graph=keep)[0].abs()[0].cpu().numpy()

            acc["grad"][family] += grad.sum(axis=(1, 2))

    def _delta_pass(self, params: np.ndarray, base_index: int = 0):
        """Measures how far each batch member's curves move away from a reference member.

        Args:
            params     : Parameter maps of shape (batch, 3 * n_gaussians, patch_height, patch_width).
            base_index : Batch index whose curves serve as the reference.

        Returns:
            Per-pixel mean squared curve difference of shape
            (batch, patch_height, patch_width); the reference row is zero.
        """
        ph, pw = self.loaded["patch"]
        deltas = []

        for row0, row1 in self._row_chunks(params.shape[0], ph, pw):
            curves = self._render_rows(params, row0, row1)
            deltas.append(((curves - curves[base_index:base_index + 1]) ** 2).mean(axis=1))

        return np.concatenate(deltas, axis=1)

    def _ablation_pass(self, acc: dict, window: np.ndarray) -> None:
        """Zeroes each input channel in turn and accumulates the curve and detection damage.

        Args:
            acc    : Accumulator whose per-channel delta, slot-flip and pixel counts are incremented.
            window : Model input of shape (1, channels, patch_height, patch_width).
        """
        from tools.data.gaussians import GaussianActivity

        run        = self.loaded["run"]
        n_channels = window.shape[1]

        batch = np.repeat(window, n_channels + 1, axis=0)
        for channel in range(n_channels):
            batch[channel + 1, channel] = 0.0

        params = run.model(batch)
        deltas = self._delta_pass(params)

        base_active = GaussianActivity.is_active(params[0, 0::3], run.activity_threshold)
        for channel in range(n_channels):
            variant_active = GaussianActivity.is_active(params[channel + 1, 0::3], run.activity_threshold)
            acc["abl_delta"][channel] += float(deltas[channel + 1].sum())
            acc["abl_flips"][channel] += float((variant_active != base_active).any(axis=0).sum())
        acc["abl_pixels"] += deltas.shape[1] * deltas.shape[2]

    def _flip_pass(self, acc: dict, window: np.ndarray) -> None:
        """Measures how far the prediction drifts under each flip of the input window.

        Every flipped prediction is flipped back before comparison, so a model
        equivariant to the flip contributes zero.

        Args:
            acc    : Accumulator whose per-flip delta and pixel counts are incremented.
            window : Model input of shape (1, channels, patch_height, patch_width).
        """
        run = self.loaded["run"]

        batch  = np.concatenate([np.flip(window, axes) if axes else window for _, axes in ModelProbe.FLIPS], axis=0)
        params = run.model(batch)

        restored = np.concatenate([
            np.flip(params[index:index + 1], axes) if axes else params[index:index + 1]
            for index, (_, axes) in enumerate(ModelProbe.FLIPS)
        ], axis=0)

        deltas = self._delta_pass(restored)
        for index, (name, _) in enumerate(ModelProbe.FLIPS):
            acc["flip_delta"][name] += float(deltas[index].sum())
        acc["flip_pixels"] += deltas.shape[1] * deltas.shape[2]

    def _noise_pass(self, acc: dict, window: np.ndarray, tile_index: int) -> None:
        """Measures curve drift under additive input noise at each survey sigma.

        Args:
            acc        : Accumulator whose per-sigma delta and pixel counts are incremented.
            window     : Model input of shape (1, channels, patch_height, patch_width).
            tile_index : Tile position, mixed into the perturbation seeds so different tiles get different noise draws.
        """
        run = self.loaded["run"]

        variants = [ModelProbe._perturb(window, {"kind": "noise", "sigma": sigma, "seed": tile_index * 100 + index}) for index, sigma in enumerate(self.NOISE_SIGMAS)]
        params   = run.model(np.concatenate([window] + variants, axis=0))
        deltas   = self._delta_pass(params)

        for index in range(len(self.NOISE_SIGMAS)):
            acc["noise_delta"][index] += float(deltas[index + 1].sum())
        acc["noise_pixels"] += deltas.shape[1] * deltas.shape[2]

    def _occlusion_grid(self) -> tuple[int, int, int, int]:
        """Returns the occluder height, width, row count and column count for the patch."""
        ph, pw = self.loaded["patch"]
        oh, ow = max(2, ph // 8), max(2, pw // 8)

        return oh, ow, int(np.ceil(ph / oh)), int(np.ceil(pw / ow))

    def _occlusion_distances(self) -> np.ndarray:
        """Precomputes the pixel distance from every patch pixel to each occluder centre.

        Returns:
            Distances in pixels of shape (n_occluders, patch_height, patch_width),
            with the occluders ordered row-major over the occlusion grid.
        """
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
        """Occludes each grid cell in turn and bins the curve damage by distance from it.

        Args:
            acc       : Accumulator whose distance-binned delta sums and counts are incremented.
            window    : Model input of shape (1, channels, patch_height, patch_width).
            distances : Pixel distances of shape (n_occluders, patch_height, patch_width) from _occlusion_distances.
        """
        run                = self.loaded["run"]
        ph, pw             = self.loaded["patch"]
        oh, ow, rows, cols = self._occlusion_grid()

        batch = np.repeat(window, rows * cols + 1, axis=0)
        for row in range(rows):
            for col in range(cols):
                cell = 1 + row * cols + col
                batch[cell, :, row * oh:min(ph, (row + 1) * oh), col * ow:min(pw, (col + 1) * ow)] = 0.0

        deltas = self._delta_pass(run.model(batch))[1:]
        bins   = (distances / self.DISTANCE_BIN).astype(np.int64)

        np.add.at(acc["occl_sum"], bins.ravel(), deltas.ravel().astype(np.float64))
        np.add.at(acc["occl_count"], bins.ravel(), 1)

    def _vitals_phase(self, tiles: list) -> dict:
        """Records per-layer activation statistics over every surveyed tile.

        Args:
            tiles : (row top, range left) origins of the surveyed tiles.

        Returns:
            Layer vitals payload pairing each module with its recorded
            activation statistics.
        """
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
        """Turns accumulated per-family gradient maps into channel shares and PNG cells.

        Args:
            maps   : Per-family summed gradient magnitude of shape (channels, patch_height, patch_width), or None for a family that never produced a gradient.
            live   : Per-family count of samples that contributed a gradient.
            center : (row, column) of the probed pixel inside the patch, or None to fall back to the patch centre.

        Returns:
            Mapping with one entry per family holding its per-channel gradient
            share and peak-normalised map PNGs, plus the centre pixel, the patch
            size and the sample count.
        """
        ph, pw     = self.loaded["patch"]
        n_channels = len(self.loaded["labels"])

        families = []
        for family in ModelProbe.FAMILIES:
            if maps[family] is None:
                families.append({"family": family, "dead": True, "shares": [0.0] * n_channels, "cells": [None] * n_channels})
                continue

            mean   = maps[family] / live[family]
            shares = mean.sum(axis=(1, 2)) / mean.sum()

            cells = []
            for channel in range(mean.shape[0]):
                peak = float(mean[channel].max())
                cells.append(ModelProbe._cell_png(mean[channel] / peak) if peak > 0.0 else None)

            families.append({"family": family, "dead": False, "shares": [float(v) for v in shares], "cells": cells})

        return {
            "families" : families,
            "center"   : center if center is not None else [ph // 2, pw // 2],
            "patch"    : [ph, pw],
            "samples"  : self.erf_samples,
        }

    def _sampling_phase(self, coverage_az: int, coverage_rg: int) -> tuple[dict, dict]:
        """Estimates the effective receptive field from randomly placed centre-pixel gradients.

        Args:
            coverage_az : Row extent in pixels the tiling covers.
            coverage_rg : Range extent in pixels the tiling covers.

        Returns:
            Tuple of the receptive-field payload, holding the mean cumulative
            radial gradient profile and the 50% and 90% radii in pixels per
            parameter family, and the gradient-map payload from
            _gradient_map_payload.
        """
        run    = self.loaded["run"]
        ph, pw = self.loaded["patch"]
        rng    = np.random.default_rng(0)

        row_lo, row_hi = ph // 2, coverage_az - ph + ph // 2
        col_lo, col_hi = pw // 2, coverage_rg - pw + pw // 2

        cumulative = {family: None for family in ModelProbe.FAMILIES}
        maps       = {family: None for family in ModelProbe.FAMILIES}
        live       = {family: 0 for family in ModelProbe.FAMILIES}
        radii      = None
        center     = None

        for index in range(self.erf_samples):
            self._tick(f"sampled gradients, sample {index + 1}/{self.erf_samples}")

            row = int(rng.integers(row_lo, max(row_lo + 1, row_hi + 1)))
            col = int(rng.integers(col_lo, max(col_lo + 1, col_hi + 1)))

            window, cy, cx = ModelProbe._window_at(run, (ph, pw), row, col)
            center         = [cy, cx]

            for family, grad in ModelProbe._family_gradients_at(run, self.loaded["device"], window, cy, cx, -1):
                if float(grad.sum()) <= 0.0:
                    continue
                profile = ModelProbe._radial_profile(grad, cy, cx)
                radii   = profile["radii"]
                arc     = np.asarray(profile["cumulative"])

                cumulative[family] = arc if cumulative[family] is None else cumulative[family] + arc
                maps[family]       = grad if maps[family] is None else maps[family] + grad
                live[family]      += 1

        families = []
        for family in ModelProbe.FAMILIES:
            if cumulative[family] is None:
                families.append({"family": family, "dead": True, "radii": None, "cumulative": None, "r50": None, "r90": None, "samples": 0})
                continue

            mean = cumulative[family] / live[family]
            families.append({
                "family"     : family,
                "dead"       : False,
                "radii"      : radii,
                "cumulative" : [float(v) for v in mean],
                "r50"        : int(radii[int(np.argmax(mean >= 0.5))]),
                "r90"        : int(radii[int(np.argmax(mean >= 0.9))]),
                "samples"    : live[family],
            })

        return {"families": families, "samples": self.erf_samples}, self._gradient_map_payload(maps, live, center)

    def _spread(self, values: list) -> dict:
        """Returns the mean, median and 90th percentile of a list of flat per-pixel arrays."""
        merged = np.concatenate(values)
        return {
            "mean"   : float(merged.mean()),
            "median" : float(np.median(merged)),
            "p90"    : float(np.percentile(merged, 90.0)),
        }

    def _finalize(self, acc: dict, tiles: list, total_tiles: int, vitals: dict, erf: dict, gradients: dict, seconds: float) -> dict:
        """Reduces every accumulator into the survey payload served to the web UI.

        Args:
            acc         : Accumulator filled by the per-tile channels.
            tiles       : Surveyed tile origins.
            total_tiles : Number of tiles the region holds before subsampling.
            vitals      : Layer vitals payload.
            erf         : Effective-receptive-field payload.
            gradients   : Gradient-map payload.
            seconds     : Wall-clock survey duration in seconds.

        Returns:
            Mapping with the run identity and coverage, the curve fit spreads,
            the detection and matched-parameter summaries when ground truth is
            available, the gradient attribution shares, the receptive-field and
            gradient maps, the ablation ranking, the flip and noise deltas, the
            distance-binned occlusion curve, and the layer vitals.
        """
        run    = self.loaded["run"]
        labels = self.loaded["labels"]
        pixels = len(tiles) * self.loaded["patch"][0] * self.loaded["patch"][1]
        has_gt = run.dataset.gt_parameters is not None

        fit = {
            "curve_mse_raw" : self._spread(acc["mse_raw"]),
            "curve_mse_gt"  : self._spread(acc["mse_gt"]) if has_gt else None,
            "base_power"    : float(np.concatenate(acc["power"]).mean()),
        }

        detection = None
        matched   = None
        if has_gt:
            detection = {
                "mean_pred_active" : float(np.concatenate(acc["pred_active"]).mean()),
                "mean_gt_active"   : float(np.concatenate(acc["gt_active"]).mean()),
                "count_match_frac" : float(np.concatenate(acc["count_match"]).mean()),
            }
            matched = {family: acc["matched_err"][family] / max(1.0, acc["matched_count"]) for family in ModelProbe.FAMILIES}
            matched["n_matched"] = int(acc["matched_count"])

        families = []
        for family in ModelProbe.FAMILIES:
            total = float(acc["grad"][family].sum())
            dead  = total <= 0.0
            families.append({
                "family" : family,
                "dead"   : dead,
                "shares" : [0.0] * len(labels) if dead else [float(v) for v in acc["grad"][family] / total],
            })

        channels = [
            {
                "channel"    : channel,
                "label"      : labels[channel],
                "delta_mse"  : acc["abl_delta"][channel] / acc["abl_pixels"],
                "flips_frac" : acc["abl_flips"][channel] / acc["abl_pixels"],
            }
            for channel in range(len(labels))
        ]
        channels.sort(key=lambda entry: entry["delta_mse"], reverse=True)

        flips = [{"flip": name, "delta_mse": acc["flip_delta"][name] / acc["flip_pixels"]} for name, _ in ModelProbe.FLIPS]
        noise = [{"value": sigma, "delta_mse": acc["noise_delta"][index] / acc["noise_pixels"]} for index, sigma in enumerate(self.NOISE_SIGMAS)]

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
            "n_gaussians" : run.n_gaussians,
            "patch"       : list(self.loaded["patch"]),
            "channels"    : labels,
            "region"      : [run.split_region.row_size, run.split_region.col_size],
            "coverage"    : {"tiles": len(tiles), "total_tiles": total_tiles, "pixels": pixels},
            "seconds"     : seconds,
            "fit"         : fit,
            "detection"   : detection,
            "matched"     : matched,
            "attribution" : {"families": families},
            "erf"         : erf,
            "gradients"   : gradients,
            "ablation"    : {"channels": channels},
            "flips"       : flips,
            "noise"       : noise,
            "occlusion"   : occlusion,
            "vitals"      : vitals,
        }

    def _survey(self) -> None:
        """Runs every survey phase over the loaded run and publishes the result.

        Sequences the six per-tile channels, then the layer vitals and sampled
        gradient phases, then the reduction. A cancellation raised at any tick
        leaves the status "cancelled" and the result unset.
        """
        started                                       = time.monotonic()
        tiles, total_tiles, coverage_az, coverage_rg  = self._tile_origins()

        n_channels = int(self.loaded["run"].in_channels)
        max_bin    = int(np.hypot(*self.loaded["patch"]) / self.DISTANCE_BIN) + 2

        self.done_units  = 0
        self.total_units = len(tiles) * 7 + self.erf_samples

        acc = {
            "power"         : [],
            "mse_raw"       : [],
            "mse_gt"        : [],
            "pred_active"   : [],
            "gt_active"     : [],
            "count_match"   : [],
            "matched_err"   : {family: 0.0 for family in ModelProbe.FAMILIES},
            "matched_count" : 0.0,
            "grad"          : {family: np.zeros(n_channels, dtype=np.float64) for family in ModelProbe.FAMILIES},
            "abl_delta"     : np.zeros(n_channels, dtype=np.float64),
            "abl_flips"     : np.zeros(n_channels, dtype=np.float64),
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
                self._base_pass(acc, self.loaded["run"].model(window), top, left)

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
            erf, gradients = self._sampling_phase(coverage_az, coverage_rg)
            result         = self._finalize(acc, tiles, total_tiles, vitals, erf, gradients, time.monotonic() - started)

            with self.lock:
                self.result = result
                self.status = {"state": "done", "path": self.loaded["path"], "progress": 1.0, "stage": "done", "error": ""}

        except SurveyCancelled:
            with self.lock:
                self.result = None
                self.status = {"state": "cancelled", "path": self.loaded["path"], "progress": 0.0, "stage": "", "error": ""}

    def _worker(self, run_path: str, split: str, device: str) -> None:
        """Runs one survey on the background thread under a capped torch thread count.

        Any failure is recorded in the status as an error state; the thread
        count and the loaded bundle are always restored on the way out.

        Args:
            run_path : Directory of the trained run to survey.
            split    : Dataset split to survey.
            device   : Torch device the model runs on.
        """
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
