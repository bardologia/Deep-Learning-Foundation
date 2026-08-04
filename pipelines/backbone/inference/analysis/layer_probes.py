from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import ModelDevice
from tools.data.io                                   import FileIO
from tools.diagnostics.activation_recorder           import ActivationRecorder
from tools.loss.param_loss                           import ParamMatcher
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class RidgeProbe:

    def __init__(self, ridge_lambda: float = 1.0, test_fraction: float = 0.5, seed: int = 0) -> None:
        self.ridge_lambda  = float(ridge_lambda)
        self.test_fraction = float(test_fraction)
        self.seed          = int(seed)

    def score(self, features: np.ndarray, targets: np.ndarray) -> float:
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
            raise ValueError(f"Probe needs X (n, d) and y (n,), got {X.shape} and {y.shape}")
        if X.shape[0] < 20:
            raise ValueError(f"Probe needs at least 20 samples, got {X.shape[0]}")

        order   = np.random.default_rng(self.seed).permutation(X.shape[0])
        n_test  = max(1, int(round(self.test_fraction * X.shape[0])))
        test    = order[:n_test]
        train   = order[n_test:]

        mean = X[train].mean(axis=0)
        std  = X[train].std(axis=0) + 1e-8
        Xn   = (X - mean) / std

        y_mean  = y[train].mean()
        y_train = y[train] - y_mean

        A = Xn[train]
        w = np.linalg.solve(A.T @ A + self.ridge_lambda * A.shape[0] * np.eye(A.shape[1]), A.T @ y_train)

        pred     = Xn[test] @ w + y_mean
        ss_res   = float(((y[test] - pred) ** 2).sum())
        ss_total = float(((y[test] - y[test].mean()) ** 2).sum())

        if ss_total <= 0.0:
            return 0.0

        return 1.0 - ss_res / ss_total


class FeatureSampler:

    MAX_FEATURES = 512

    def __init__(self, samples_per_batch: int, seed: int = 0) -> None:
        self.samples_per_batch = int(samples_per_batch)
        self.rng               = np.random.default_rng(seed)

    def sample_coords(self, B: int, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n  = min(self.samples_per_batch, B * H * W)
        ix = self.rng.choice(B * H * W, size=n, replace=False)

        return ix // (H * W), (ix // W) % H, ix % W

    @staticmethod
    def is_feature_map(stored: torch.Tensor, B: int, H: int, W: int) -> bool:
        if stored.ndim != 4:
            return False

        n, _C, h, w = stored.shape

        if n != B or h > H or w > W:
            return False

        return H % h == 0 and W % w == 0 and H // h == W // w

    def features_at(self, stored: torch.Tensor, b: np.ndarray, i: np.ndarray, j: np.ndarray, B: int, H: int, W: int) -> np.ndarray:
        if not self.is_feature_map(stored, B, H, W):
            raise ValueError(f"Activation of shape {tuple(stored.shape)} is not a (B, C, H, W) map of a {B}x{H}x{W} input; its leading dimension is not the batch or its grid is not a uniform downsample of the input")

        maps        = stored.numpy()
        _B, C, h, w = maps.shape

        fi = np.clip((i * h) // H, 0, h - 1)
        fj = np.clip((j * w) // W, 0, w - 1)

        features = maps[b, :, fi, fj]

        if C > self.MAX_FEATURES:
            keep     = np.linspace(0, C - 1, self.MAX_FEATURES).astype(int)
            features = features[:, keep]

        return features


class LayerProbeCore:

    def __init__(self, model, layers: list[str], ppg: int, amp_thr: float, samples_per_batch: int, probe: RidgeProbe, seed: int = 0) -> None:
        self.model   = model
        self.layers  = list(layers)
        self.ppg     = ppg
        self.amp_thr = amp_thr
        self.probe   = probe
        self.sampler = FeatureSampler(samples_per_batch, seed)

    def _targets(self, gt_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        B, C, H, W = gt_phys.shape
        p          = gt_phys.reshape(B, C // self.ppg, self.ppg, H, W)
        amps       = p[:, :, 0]
        mus        = p[:, :, 1]

        counts = (amps > self.amp_thr).sum(axis=1).astype(np.float64)

        dominant = np.take_along_axis(mus, amps.argmax(axis=1)[:, None], axis=1)[:, 0]
        dominant = np.where(counts > 0, dominant, np.nan)

        return counts, dominant

    def collect(self, batches: list[tuple[torch.Tensor, np.ndarray]]) -> dict[str, dict]:
        store: dict[str, list] = {layer: [] for layer in self.layers}
        counts_all             = []
        mus_all                = []

        recorder = ActivationRecorder(self.model)
        recorder.attach_store(self.layers)

        device = ModelDevice.of(self.model)

        for images, gt_phys in batches:
            with torch.no_grad():
                self.model(images.to(device))

            stored     = recorder.stored()
            B, _, H, W = images.shape
            b, i, j    = self.sampler.sample_coords(B, H, W)

            counts, dominant = self._targets(gt_phys)
            counts_all.append(counts[b, i, j])
            mus_all.append(dominant[b, i, j])

            for layer in self.layers:
                tensor = stored.get(layer)
                if tensor is None or not self.sampler.is_feature_map(tensor, B, H, W):
                    store.pop(layer, None)
                    continue
                if layer in store:
                    store[layer].append(self.sampler.features_at(tensor, b, i, j, B, H, W))

        recorder.detach()

        if not store:
            raise ValueError("No probed layer produced a (B, C, H, W) feature map on the input grid; nothing to probe")

        counts = np.concatenate(counts_all)
        mus    = np.concatenate(mus_all)

        results = {}
        for layer, chunks in store.items():
            X       = np.concatenate(chunks, axis=0)
            with_mu = np.isfinite(mus)

            results[layer] = {
                "n_channels" : int(X.shape[1]),
                "k_r2"       : self.probe.score(X, counts),
                "mu_r2"      : self.probe.score(X[with_mu], mus[with_mu]) if with_mu.sum() >= 20 else None,
            }

        return results


class LayerProbePlots(PlotBase):

    def decodability_by_depth(self, layers: list[str], scores: list[float | None], target: str, path: Path) -> Path:
        self._apply_style()

        values = [score if score is not None else np.nan for score in scores]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(range(len(layers)), values, marker="o", linewidth=1.4, color="#0072B2")
        ax.set_xlabel("Probed layer index (forward order)")
        ax.set_ylabel("Held-out linear-probe R²")
        ax.set_title(f"Where '{target}' becomes linearly decodable")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, path)


class LayerProbeRun(AnalysisRun):

    SUMMARY_FILENAME = "layer_probes.json"
    REPORT_FILENAME  = "layer_probes.md"

    def _select_layers(self, run) -> list[str]:
        names = ActivationRecorder(run.model.module).leaf_names()

        if len(names) <= self.config.max_layers:
            return names

        keep = np.linspace(0, len(names) - 1, self.config.max_layers).astype(int)
        return [names[index] for index in sorted(set(keep))]

    def _batches(self, run) -> list[tuple[torch.Tensor, np.ndarray]]:
        batches = []
        for index, batch in enumerate(run.loader):
            if index >= self.config.max_batches:
                break
            gt_phys = run.dataset.normalizer.denormalize_output(batch[1].float()).numpy()
            batches.append((batch[0], gt_phys))

        if not batches:
            raise ValueError("The loader yielded no batches to probe")

        return batches

    def _write_report(self, run, layers: list[str], results: dict, figures: dict[str, Path]) -> Path:
        doc = MarkdownDoc(title=f"Layer-wise linear probes: {run.backbone_name}")
        doc.paragraph(
            f"Ridge probes (λ={self.config.ridge_lambda}) on {len(layers)} layers over {self.config.max_batches} '{self.config.split}' batches: "
            "held-out R² of a linear readout predicting the GT active Gaussian count and the dominant scatterer elevation from each layer's features. "
            "Rising curves show where the quantity becomes linearly decodable inside the network."
        )

        table = MarkdownTable(("#", "Layer", "Channels", "K R²", "mu R²"))
        for index, layer in enumerate(layers):
            entry = results[layer]
            mu_cell = f"{entry['mu_r2']:.3f}" if entry["mu_r2"] is not None else "n/a"
            table.add_row(str(index), f"`{layer}`", str(entry["n_channels"]), f"{entry['k_r2']:.3f}", mu_cell)
        doc.table(table)

        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run    = self._load_run()
        layers = self._select_layers(run)

        core = LayerProbeCore(
            model             = run.model.module,
            layers            = layers,
            ppg               = 3,
            amp_thr           = ParamMatcher.ACTIVE_AMP_THR,
            samples_per_batch = self.config.samples_per_batch,
            probe             = RidgeProbe(self.config.ridge_lambda, self.config.test_fraction, self.config.seed),
            seed              = self.config.seed,
        )

        results = core.collect(self._batches(run))
        layers  = [layer for layer in layers if layer in results]

        plots   = LayerProbePlots()
        figures = {
            "k_by_depth"  : plots.decodability_by_depth(layers, [results[l]["k_r2"] for l in layers], "active count K", self.output_dir / "k_by_depth.png"),
            "mu_by_depth" : plots.decodability_by_depth(layers, [results[l]["mu_r2"] for l in layers], "dominant mu", self.output_dir / "mu_by_depth.png"),
        }

        payload = {"backbone": run.backbone_name, "split": self.config.split, "layers": layers, "results": results}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, layers, results, figures)

        best = max(layers, key=lambda layer: results[layer]["k_r2"])
        self.logger.ok(f"{self.run_dir.name}: K most decodable at '{best}' (R² {results[best]['k_r2']:.3f}) -> {report_path}")

        return payload


class LayerProbeBatch(RunBatch):

    SELECTOR_ACTION = "probe"
    SECTION_TITLE   = "Layer-wise linear probes"
    RUN_CLASS       = LayerProbeRun
