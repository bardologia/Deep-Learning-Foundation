"""Layer-wise linear probes measuring where scene quantities become decodable.

Ridge probes are fitted on the features of each layer to predict five ground-truth
quantities of the volumetric scene, against an input-pixel baseline and a
shuffled-target control, alongside the effective dimensionality of each layer.
"""

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
    """Cross-validated ridge regression scoring how linearly decodable a target is.

    Attributes:
        ridge_lambda : Ridge penalty, scaled by the number of training samples.
        n_folds      : Number of cross-validation folds.
        seed         : Seed of the fold permutation.
    """

    def __init__(self, ridge_lambda: float = 1.0, n_folds: int = 4, seed: int = 0) -> None:
        """Stores the probe hyper-parameters.

        Args:
            ridge_lambda : Ridge penalty per sample.
            n_folds      : Number of cross-validation folds.
            seed         : Seed of the fold permutation.

        Raises:
            ValueError : If fewer than two folds are requested.
        """
        self.ridge_lambda = float(ridge_lambda)
        self.n_folds      = int(n_folds)
        self.seed         = int(seed)

        if self.n_folds < 2:
            raise ValueError(f"Cross-validated probe needs at least 2 folds, got {self.n_folds}")

    def _fold_score(self, X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
        """Fits the ridge readout on one fold's train indices and scores it on its test indices.

        Features are standardized and the target centred using the train statistics only.

        Args:
            X     : Feature matrix of shape (samples, features).
            y     : Target vector of shape (samples,).
            train : Sample indices of the training part.
            test  : Sample indices of the held-out part.

        Returns:
            The held-out R², or 0.0 when the held-out target has no variance.
        """
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

    def score(self, features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        """Cross-validates the ridge readout over the configured folds.

        Args:
            features : Feature matrix of shape (samples, features).
            targets  : Target vector of shape (samples,).

        Returns:
            Dictionary with the "mean" and "std" held-out R² across folds.

        Raises:
            ValueError : If the shapes do not line up or fewer than 20 samples are given.
        """
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
            raise ValueError(f"Probe needs X (n, d) and y (n,), got {X.shape} and {y.shape}")
        if X.shape[0] < 20:
            raise ValueError(f"Probe needs at least 20 samples, got {X.shape[0]}")

        order = np.random.default_rng(self.seed).permutation(X.shape[0])
        folds = np.array_split(order, self.n_folds)

        scores = []
        for f, test in enumerate(folds):
            train = np.concatenate([fold for g, fold in enumerate(folds) if g != f])
            scores.append(self._fold_score(X, y, train, test))

        return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


class FeatureSampler:
    """Samples pixel positions and reads the matching features off activation maps.

    Attributes:
        samples_per_batch : Number of pixel positions drawn per batch.
        rng               : Generator drawing the positions.
    """

    MAX_FEATURES = 512

    def __init__(self, samples_per_batch: int, seed: int = 0) -> None:
        """Stores the sampling budget and seeds the generator.

        Args:
            samples_per_batch : Number of pixel positions to draw per batch.
            seed              : Seed of the position generator.
        """
        self.samples_per_batch = int(samples_per_batch)
        self.rng               = np.random.default_rng(seed)

    def sample_coords(self, B: int, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draws pixel positions uniformly over a batch of patches, without replacement.

        Args:
            B : Batch size.
            H : Patch height in row pixels.
            W : Patch width in range pixels.

        Returns:
            Three index arrays of equal length holding the batch, row and column
            coordinate of each drawn pixel.
        """
        n  = min(self.samples_per_batch, B * H * W)
        ix = self.rng.choice(B * H * W, size=n, replace=False)

        return ix // (H * W), (ix // W) % H, ix % W

    @staticmethod
    def is_feature_map(stored: torch.Tensor, B: int, H: int, W: int) -> bool:
        """Returns whether a stored activation is a spatial map alignable to the input grid.

        Args:
            stored : Recorded activation tensor.
            B      : Batch size of the forward pass.
            H      : Input patch height in pixels.
            W      : Input patch width in pixels.

        Returns:
            True when the tensor is (B, C, h, w) with h and w a uniform integer
            downsample of the input grid.
        """
        if stored.ndim != 4:
            return False

        n, _C, h, w = stored.shape

        if n != B or h > H or w > W:
            return False

        return H % h == 0 and W % w == 0 and H // h == W // w

    def features_at(self, stored: torch.Tensor, b: np.ndarray, i: np.ndarray, j: np.ndarray, B: int, H: int, W: int) -> np.ndarray:
        """Reads the feature vector of each sampled pixel off an activation map.

        Input coordinates are mapped onto the layer's own grid, and layers wider than
        MAX_FEATURES channels are subsampled to that many evenly spaced channels.

        Args:
            stored : Activation of shape (B, C, h, w).
            b      : Batch index of each sampled pixel.
            i      : Azimuth index of each sampled pixel on the input grid.
            j      : Range index of each sampled pixel on the input grid.
            B      : Batch size of the forward pass.
            H      : Input patch height in pixels.
            W      : Input patch width in pixels.

        Returns:
            Feature matrix of shape (samples, min(C, MAX_FEATURES)).

        Raises:
            ValueError : If the activation is not a feature map on the input grid.
        """
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


class FeatureGeometry:
    """Geometric descriptors of a layer's feature cloud."""

    @staticmethod
    def participation_ratio(features: np.ndarray) -> float:
        """Returns the effective number of dimensions the features spread their variance over.

        Args:
            features : Feature matrix of shape (samples, features).

        Returns:
            The participation ratio of the centred singular spectrum, 0.0 when the cloud
            carries no variance.
        """
        centered = features - features.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        spectrum = singular ** 2

        total = spectrum.sum()
        if total <= 0.0:
            return 0.0

        return float(total ** 2 / (spectrum ** 2).sum())


class LayerProbeCore:
    """Collects layer features and scores the linear decodability of the scene targets.

    Attributes:
        model   : Model whose layers are probed.
        layers  : Qualified names of the probed layers, in forward order.
        ppg     : Number of parameters per Gaussian slot in the ground truth.
        amp_thr : Amplitude above which a Gaussian slot counts as active.
        probe   : Ridge probe scoring each layer.
        sampler : Sampler drawing the pixel positions and reading their features.
    """

    TARGETS = (
        ("count",           "active Gaussian count K"),
        ("dominant_mu",     "dominant component z position"),
        ("dominant_sigma",  "dominant component width"),
        ("total_amp",       "total active amplitude"),
        ("second_presence", "second component present"),
    )

    def __init__(self, model, layers: list[str], ppg: int, amp_thr: float, samples_per_batch: int, probe: RidgeProbe, seed: int = 0) -> None:
        """Stores the model, the probed layers and the sampling and scoring components.

        Args:
            model             : Model to probe.
            layers            : Qualified names of the layers to record.
            ppg               : Number of parameters per Gaussian slot, three for amplitude, mu and sigma.
            amp_thr           : Amplitude threshold marking a Gaussian slot as active.
            samples_per_batch : Number of pixel positions drawn per batch.
            probe             : Ridge probe used for every layer and target.
            seed              : Seed of the sampler and of the shuffled-target control.
        """
        self.model   = model
        self.layers  = list(layers)
        self.ppg     = ppg
        self.amp_thr = amp_thr
        self.probe   = probe
        self.sampler = FeatureSampler(samples_per_batch, seed)
        self.rng     = np.random.default_rng(seed)

    def _targets(self, gt_phys: np.ndarray) -> dict[str, np.ndarray]:
        """Derives the five per-pixel scene quantities from the ground-truth parameters.

        Args:
            gt_phys : Ground-truth parameters in physical units of shape (batch, 3 * slots, rows, cols), with mu and sigma in metres.

        Returns:
            Per-pixel maps of shape (batch, rows, cols) keyed by target: the active
            Gaussian "count", the "dominant_mu" and "dominant_sigma" of the strongest slot
            (NaN where no slot is active), the "total_amp" over active slots and the
            binary "second_presence".
        """
        B, C, H, W = gt_phys.shape
        p          = gt_phys.reshape(B, C // self.ppg, self.ppg, H, W)
        amps       = p[:, :, 0]
        mus        = p[:, :, 1]
        sigmas     = p[:, :, 2]

        active = amps > self.amp_thr
        counts = active.sum(axis=1).astype(np.float64)

        dominant_index = amps.argmax(axis=1)[:, None]
        dominant_mu    = np.take_along_axis(mus, dominant_index, axis=1)[:, 0]
        dominant_sigma = np.take_along_axis(sigmas, dominant_index, axis=1)[:, 0]

        with_dominant  = counts > 0
        dominant_mu    = np.where(with_dominant, dominant_mu, np.nan)
        dominant_sigma = np.where(with_dominant, dominant_sigma, np.nan)

        return {
            "count"           : counts,
            "dominant_mu"     : dominant_mu,
            "dominant_sigma"  : dominant_sigma,
            "total_amp"       : np.where(active, amps, 0.0).sum(axis=1),
            "second_presence" : (counts >= 2).astype(np.float64),
        }

    def _score_targets(self, features: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, dict | None]:
        """Scores every target from one feature matrix, skipping the undersampled ones.

        Args:
            features : Feature matrix of shape (samples, features).
            targets  : Target vectors of shape (samples,) keyed by target name.

        Returns:
            The probe score of each target, or None when fewer than 20 samples carry a
            finite value for it.
        """
        scores = {}
        for target, _label in self.TARGETS:
            y      = targets[target]
            usable = np.isfinite(y)

            scores[target] = self.probe.score(features[usable], y[usable]) if usable.sum() >= 20 else None

        return scores

    def collect(self, batches: list[tuple[torch.Tensor, np.ndarray]]) -> dict:
        """Records the layer features over the batches and probes every target on them.

        Layers whose activation is not a feature map on the input grid are dropped. The
        raw input channels are probed as a baseline, and the best layer of each target is
        re-probed on permuted targets as a leakage control.

        Args:
            batches : Pairs of an input batch of shape (batch, channels, rows, cols) and its ground-truth parameters in physical units of shape (batch, 3 * slots, rows, cols).

        Returns:
            Dictionary with "layers", mapping each surviving layer to its channel count,
            effective dimensionality and per-target scores, the same block for the
            "input_baseline", and the "shuffled_controls" naming the best layer and its
            permuted-target score per target.

        Raises:
            ValueError : If no probed layer produced a usable feature map.
        """
        store: dict[str, list] = {layer: [] for layer in self.layers}
        input_chunks           = []
        target_chunks          = {target: [] for target, _label in self.TARGETS}

        recorder = ActivationRecorder(self.model)
        recorder.attach_store(self.layers)

        device = ModelDevice.of(self.model)

        for images, gt_phys in batches:
            with torch.no_grad():
                self.model(images.to(device))

            stored     = recorder.stored()
            B, _, H, W = images.shape
            b, i, j    = self.sampler.sample_coords(B, H, W)

            targets = self._targets(gt_phys)
            for target, _label in self.TARGETS:
                target_chunks[target].append(targets[target][b, i, j])

            input_chunks.append(self.sampler.features_at(images.cpu(), b, i, j, B, H, W))

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

        targets = {target: np.concatenate(chunks) for target, chunks in target_chunks.items()}

        layers = {}
        for layer, chunks in store.items():
            X             = np.concatenate(chunks, axis=0)
            layers[layer] = {
                "n_channels"    : int(X.shape[1]),
                "effective_dim" : FeatureGeometry.participation_ratio(X),
                "scores"        : self._score_targets(X, targets),
            }

        X_input  = np.concatenate(input_chunks, axis=0)
        baseline = {
            "n_channels"    : int(X_input.shape[1]),
            "effective_dim" : FeatureGeometry.participation_ratio(X_input),
            "scores"        : self._score_targets(X_input, targets),
        }

        controls = {}
        for target, _label in self.TARGETS:
            scored = {layer: entry["scores"][target]["mean"] for layer, entry in layers.items() if entry["scores"][target] is not None}
            if not scored:
                controls[target] = None
                continue

            best     = max(scored, key=scored.get)
            X_best   = np.concatenate(store[best], axis=0)
            y        = targets[target]
            usable   = np.isfinite(y)
            shuffled = self.rng.permutation(y[usable])

            controls[target] = {"layer": best, "score": self.probe.score(X_best[usable], shuffled)["mean"]}

        return {"layers": layers, "input_baseline": baseline, "shuffled_controls": controls}


class LayerProbePlots(PlotBase):
    """Depth profiles of probe decodability and of feature dimensionality."""

    def decodability_by_depth(self, layers: list[str], results: dict, target: str, label: str, baseline: dict | None, control: dict | None, path: Path) -> Path:
        """Draws the held-out probe R² of one target across the probed depth.

        Args:
            layers   : Probed layer names in forward order.
            results  : Per-layer probe results keyed by layer name.
            target   : Target key to plot.
            label    : Human-readable target name for the title.
            baseline : Input-pixel baseline block, drawn as a reference line when scored.
            control  : Shuffled-target control block, drawn as a reference line when present.
            path     : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        depth  = np.arange(len(layers))
        means  = np.array([results[layer]["scores"][target]["mean"] if results[layer]["scores"][target] is not None else np.nan for layer in layers])
        stds   = np.array([results[layer]["scores"][target]["std"] if results[layer]["scores"][target] is not None else np.nan for layer in layers])

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))

        band = np.isfinite(means) & np.isfinite(stds)
        ax.fill_between(depth[band], (means - stds)[band], (means + stds)[band], color=self.OKABE_ITO[0], alpha=0.25, label="± std across folds")
        ax.plot(depth, means, marker="o", linewidth=1.4, color=self.OKABE_ITO[0], label="layer probe")

        if baseline is not None and baseline["scores"][target] is not None:
            ax.axhline(baseline["scores"][target]["mean"], color=self.OKABE_ITO[1], linestyle="--", linewidth=1.1, label="input pixels")
        if control is not None:
            ax.axhline(control["score"], color="0.45", linestyle=":", linewidth=1.1, label="shuffled targets")
        ax.axhline(0.0, color="0.75", linewidth=0.8)

        if np.isfinite(means).any():
            best = int(np.nanargmax(means))
            ax.scatter([depth[best]], [means[best]], s=70, facecolors="none", edgecolors=self.OKABE_ITO[2], linewidths=1.4, zorder=4)
            ax.annotate(f"best: layer {best} ({means[best]:.2f})", xy=(depth[best], means[best]), xytext=(6, -12), textcoords="offset points", fontsize=7, color="0.25")

        ax.set_xlabel("Probed layer index (forward order)")
        ax.set_ylabel("Held-out linear-probe R²")
        ax.set_title(f"Where '{label}' becomes linearly decodable")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)

    def effective_dim_by_depth(self, layers: list[str], results: dict, input_dim: float, path: Path) -> Path:
        """Draws the participation ratio of each layer against the input-pixel reference.

        Args:
            layers    : Probed layer names in forward order.
            results   : Per-layer probe results holding the effective dimensionality.
            input_dim : Participation ratio of the raw input channels.
            path      : Output path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        depth  = np.arange(len(layers))
        values = np.array([results[layer]["effective_dim"] for layer in layers])

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(depth, values, marker="o", linewidth=1.4, color=self.OKABE_ITO[0], label="layer features")
        ax.axhline(input_dim, color=self.OKABE_ITO[1], linestyle="--", linewidth=1.1, label="input pixels")

        peak = int(np.argmax(values))
        ax.annotate(f"max {values[peak]:.1f}", xy=(depth[peak], values[peak]), xytext=(4, 6), textcoords="offset points", fontsize=7, color="0.25")

        ax.set_xlabel("Probed layer index (forward order)")
        ax.set_ylabel("Participation ratio [effective dims]")
        ax.set_title("Effective feature dimensionality by depth")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)


class LayerProbeRun(AnalysisRun):
    """Runs the layer-wise linear probes on a single training run."""

    SUMMARY_FILENAME = "layer_probes.json"
    REPORT_FILENAME  = "layer_probes.md"

    def _select_layers(self, run) -> list[str]:
        """Picks the layers to probe, thinned evenly over depth when the model has too many.

        Args:
            run : Loaded run exposing the model.

        Returns:
            Qualified leaf-module names in forward order, at most max_layers of them.
        """
        names = ActivationRecorder(run.model.module).leaf_names()

        if len(names) <= self.config.max_layers:
            return names

        keep = np.linspace(0, len(names) - 1, self.config.max_layers).astype(int)
        return [names[index] for index in sorted(set(keep))]

    def _batches(self, run) -> list[tuple[torch.Tensor, np.ndarray]]:
        """Materializes the probe batches with their targets denormalized to physical units.

        Args:
            run : Loaded run bundling the dataloader and the dataset normalizer.

        Returns:
            Pairs of a normalized input batch and its ground-truth parameters in physical
            units of shape (batch, 3 * slots, rows, cols).

        Raises:
            ValueError : If the loader yields no batch.
        """
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
        """Writes the markdown report of the probes.

        Args:
            run     : Loaded run, used for the backbone name in the title.
            layers  : Probed layer names in forward order.
            results : Probe results with their layers, baseline and controls.
            figures : Figure paths keyed by figure name.

        Returns:
            Path of the written report.
        """
        doc = MarkdownDoc(title=f"Layer-wise linear probes: {run.backbone_name}")
        doc.paragraph(
            f"Ridge probes (λ={self.config.ridge_lambda}, {self.config.n_folds}-fold cross-validation) on {len(layers)} layers over {self.config.max_batches} '{self.config.split}' batches: "
            "held-out R² of a linear readout predicting five GT quantities from each layer's features. Rising curves show where a quantity becomes linearly decodable inside the network. "
            "The input-pixels baseline probes the raw input channels at the same sampled pixels: layers below it destroy linearly available information. "
            "The shuffled-targets control reruns the best layer's probe on permuted targets and should sit near zero; anything higher marks probe leakage. "
            "The participation ratio is the effective number of feature dimensions the layer spreads its variance over."
        )

        entries = results["layers"]

        doc.heading("Decodability", level=2)
        table = MarkdownTable(("#", "Layer", "Channels", "Eff. dims") + tuple(label for _target, label in LayerProbeCore.TARGETS))
        for index, layer in enumerate(layers):
            entry = entries[layer]
            row   = [str(index), f"`{layer}`", str(entry["n_channels"]), f"{entry['effective_dim']:.1f}"]
            for target, _label in LayerProbeCore.TARGETS:
                score = entry["scores"][target]
                row.append(f"{score['mean']:.3f} ± {score['std']:.3f}" if score is not None else "n/a")
            table.add_row(*row)
        doc.table(table)

        doc.heading("Baselines and controls", level=2)
        baseline = results["input_baseline"]
        control_table = MarkdownTable(("Target", "Input-pixel R²", "Shuffled control", "Control layer"))
        for target, label in LayerProbeCore.TARGETS:
            base    = baseline["scores"][target]
            control = results["shuffled_controls"][target]
            control_table.add_row(label, f"{base['mean']:.3f}" if base is not None else "n/a", f"{control['score']:.3f}" if control is not None else "n/a", f"`{control['layer']}`" if control is not None else "—")
        doc.table(control_table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def run(self) -> dict:
        """Probes the layers of this run, plots the depth profiles and writes the report.

        Returns:
            Payload with the backbone name, the split, the surviving layer order and the
            probe results, baseline and shuffled controls.
        """
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
            probe             = RidgeProbe(self.config.ridge_lambda, self.config.n_folds, self.config.seed),
            seed              = self.config.seed,
        )

        results = core.collect(self._batches(run))
        layers  = [layer for layer in layers if layer in results["layers"]]

        plots   = LayerProbePlots()
        figures = {}
        for target, label in LayerProbeCore.TARGETS:
            figures[f"{target}_by_depth"] = plots.decodability_by_depth(
                layers, results["layers"], target, label,
                baseline = results["input_baseline"],
                control  = results["shuffled_controls"][target],
                path     = self.output_dir / f"{target}_by_depth.png",
            )
        figures["effective_dim_by_depth"] = plots.effective_dim_by_depth(layers, results["layers"], results["input_baseline"]["effective_dim"], self.output_dir / "effective_dim_by_depth.png")

        payload = {"backbone": run.backbone_name, "split": self.config.split, "layer_order": layers, **results}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, layers, results, figures)

        count_scores = {layer: results["layers"][layer]["scores"]["count"] for layer in layers}
        best         = max((layer for layer in layers if count_scores[layer] is not None), key=lambda layer: count_scores[layer]["mean"])
        self.logger.ok(f"{self.run_dir.name}: K most decodable at '{best}' (R² {count_scores[best]['mean']:.3f}) -> {report_path}")

        return payload


class LayerProbeBatch(RunBatch):
    """Runs the layer probes over every run matching the configured filter."""

    SELECTOR_ACTION = "probe"
    SECTION_TITLE   = "Layer-wise linear probes"
    RUN_CLASS       = LayerProbeRun
