
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.backbone.inference.analysis.run_batch import AnalysisOutcome, AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import ModelDevice
from tools.data.io                                   import FileIO
from tools.diagnostics.activation_recorder           import ActivationRecorder
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class RidgeProbe:

    MIN_SAMPLES = 20

    def __init__(self, ridge_lambda: float = 1.0, n_folds: int = 4, seed: int = 0) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.n_folds      = int(n_folds)
        self.seed         = int(seed)

    def _fold_score(self, X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
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
            return float("nan")

        return 1.0 - ss_res / ss_total

    def _folds(self, n_samples: int) -> list[np.ndarray]:
        order = np.random.default_rng(self.seed).permutation(n_samples)
        return np.array_split(order, self.n_folds)

    def degenerate_folds(self, targets: np.ndarray) -> int:
        y = np.asarray(targets, dtype=np.float64)
        return sum(1 for test in self._folds(y.size) if float(((y[test] - y[test].mean()) ** 2).sum()) <= 0.0)

    def score(self, features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        folds = self._folds(X.shape[0])

        scores = []
        for f, test in enumerate(folds):
            train = np.concatenate([fold for g, fold in enumerate(folds) if g != f])
            scores.append(self._fold_score(X, y, train, test))

        defined = np.array([score for score in scores if np.isfinite(score)])

        return {"mean": float(defined.mean()), "std": float(defined.std()), "degenerate_folds": len(scores) - int(defined.size)}


class FeatureSampler:

    MAX_FEATURES = 512

    def __init__(self, samples_per_batch: int, seed: int = 0) -> None:
        self.samples_per_batch = int(samples_per_batch)
        self.rng               = np.random.default_rng(seed)

    def sample_coords(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B, H, W = weights.shape
        valid   = np.flatnonzero(weights.reshape(-1) > 0.0)
        ix      = self.rng.choice(valid, size=min(self.samples_per_batch, valid.size), replace=False)

        return ix // (H * W), (ix // W) % H, ix % W

    @staticmethod
    def is_feature_map(stored: torch.Tensor, B: int, H: int, W: int) -> bool:
        if stored.ndim != 4:
            return False

        n, _C, h, w = stored.shape

        if n != B or h > H or w > W:
            return False

        return H % h == 0 and W % w == 0 and H // h == W // w

    @classmethod
    def drop_reason(cls, stored: torch.Tensor | None, B: int, H: int, W: int) -> str | None:
        if stored is None:
            return "produced no tensor output"
        if not cls.is_feature_map(stored, B, H, W):
            return f"output of shape {tuple(stored.shape)} is not a feature map on the {B}x{H}x{W} input grid"

        return None

    def features_at(self, stored: torch.Tensor, b: np.ndarray, i: np.ndarray, j: np.ndarray, B: int, H: int, W: int) -> np.ndarray:
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

    @staticmethod
    def participation_ratio(features: np.ndarray) -> float:
        centered = features - features.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        spectrum = singular ** 2

        total = spectrum.sum()
        if total <= 0.0:
            return 0.0

        return float(total ** 2 / (spectrum ** 2).sum())


class LayerProbeCore:

    def __init__(self, model, layers: list[str], target_names: list[str], samples_per_batch: int, probe: RidgeProbe, seed: int = 0) -> None:
        self.model        = model
        self.layers       = list(layers)
        self.target_names = list(target_names)
        self.probe        = probe
        self.sampler      = FeatureSampler(samples_per_batch, seed)
        self.rng          = np.random.default_rng(seed)

    def _target_notes(self, targets: dict[str, np.ndarray]) -> tuple[dict[str, str], set[str]]:
        notes    = {}
        unscored = set()

        for target in self.target_names:
            usable = targets[target][np.isfinite(targets[target])]

            if usable.size < RidgeProbe.MIN_SAMPLES:
                notes[target] = f"{usable.size} finite samples < {RidgeProbe.MIN_SAMPLES}; not scored"
                unscored.add(target)
                continue

            degenerate = self.probe.degenerate_folds(usable)
            if degenerate == self.probe.n_folds:
                notes[target] = f"zero variance in {degenerate}/{self.probe.n_folds} held-out folds; R² undefined, not scored"
                unscored.add(target)
            elif degenerate > 0:
                notes[target] = f"zero variance in {degenerate}/{self.probe.n_folds} held-out folds; R² averaged over the other folds"

        return notes, unscored

    def _score_targets(self, features: np.ndarray, targets: dict[str, np.ndarray], unscored: set[str]) -> dict[str, dict | None]:
        scores = {}
        for target in self.target_names:
            y      = targets[target]
            usable = np.isfinite(y)

            scores[target] = None if target in unscored else self.probe.score(features[usable], y[usable])

        return scores

    def collect(self, batches: list[tuple[torch.Tensor, np.ndarray, np.ndarray]]) -> dict:
        store: dict[str, list]  = {layer: [] for layer in self.layers}
        widths: dict[str, int]  = {}
        dropped: dict[str, str] = {}
        input_chunks            = []
        target_chunks           = {target: [] for target in self.target_names}

        recorder = ActivationRecorder(self.model)
        recorder.attach_store(self.layers)

        device = ModelDevice.of(self.model)

        for images, target_maps, weights in batches:
            with torch.no_grad():
                self.model(images.to(device))

            stored     = recorder.stored()
            B, _, H, W = images.shape
            b, i, j    = self.sampler.sample_coords(weights[:, 0])

            for index, target in enumerate(self.target_names):
                target_chunks[target].append(target_maps[b, index, i, j])

            input_chunks.append(self.sampler.features_at(images.cpu(), b, i, j, B, H, W))

            for layer in self.layers:
                if layer in dropped:
                    continue

                tensor = stored.get(layer)
                reason = FeatureSampler.drop_reason(tensor, B, H, W)
                if reason is not None:
                    dropped[layer] = reason
                    store.pop(layer)
                    continue

                widths[layer] = int(tensor.shape[1])
                store[layer].append(self.sampler.features_at(tensor, b, i, j, B, H, W))

        recorder.detach()

        targets         = {target: np.concatenate(chunks) for target, chunks in target_chunks.items()}
        notes, unscored = self._target_notes(targets)

        layers = {}
        for layer, chunks in store.items():
            X             = np.concatenate(chunks, axis=0)
            layers[layer] = {
                "n_channels"         : widths[layer],
                "n_channels_sampled" : int(X.shape[1]),
                "effective_dim"      : FeatureGeometry.participation_ratio(X),
                "scores"             : self._score_targets(X, targets, unscored),
            }

        X_input  = np.concatenate(input_chunks, axis=0)
        baseline = {
            "n_channels"         : int(batches[0][0].shape[1]),
            "n_channels_sampled" : int(X_input.shape[1]),
            "effective_dim"      : FeatureGeometry.participation_ratio(X_input),
            "scores"             : self._score_targets(X_input, targets, unscored),
        }

        controls = {}
        for target in self.target_names:
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

        return {"layers": layers, "input_baseline": baseline, "shuffled_controls": controls, "dropped_layers": dropped, "target_notes": notes}


class LayerProbePlots(PlotBase):

    def decodability_by_depth(self, layers: list[str], results: dict, target: str, baseline: dict | None, control: dict | None, path: Path) -> Path:
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
        ax.set_title(f"Where target '{target}' becomes linearly decodable")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)

    def effective_dim_by_depth(self, layers: list[str], results: dict, input_dim: float, path: Path) -> Path:
        self._apply_style()

        depth  = np.arange(len(layers))
        values = np.array([results[layer]["effective_dim"] for layer in layers])

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(depth, values, marker="o", linewidth=1.4, color=self.OKABE_ITO[0], label="layer features")
        ax.axhline(input_dim, color=self.OKABE_ITO[1], linestyle="--", linewidth=1.1, label="input pixels")

        peak = int(np.argmax(values))
        ax.annotate(f"max {values[peak]:.1f}", xy=(depth[peak], values[peak]), xytext=(4, 6), textcoords="offset points", fontsize=7, color="0.25")

        ax.set_xlabel("Probed layer index (forward order)")
        ax.set_ylabel("Participation ratio (effective dimensions)")
        ax.set_title("Effective feature dimensionality by depth")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()

        return self._save(fig, path)


class LayerProbeRun(AnalysisRun):

    SUMMARY_FILENAME = "layer_probes.json"
    REPORT_FILENAME  = "layer_probes.md"

    def _select_layers(self, run) -> list[str]:
        names = ActivationRecorder(run.model.module).leaf_names()

        if len(names) <= self.config.max_layers:
            selected = names
        else:
            keep     = np.linspace(0, len(names) - 1, self.config.max_layers).astype(int)
            selected = [names[index] for index in sorted(set(keep))]

        self.logger.info(f"{self.run_dir.name}: probing {len(selected)} of {len(names)} leaf layers (max_layers={self.config.max_layers})")

        return selected

    def _batches(self, run) -> list[tuple[torch.Tensor, np.ndarray, np.ndarray]]:
        batches = []
        for index, (inputs, targets, weights) in enumerate(run.loader):
            if index >= self.config.max_batches:
                break
            batches.append((inputs, run.model.denormalize_target(targets.float()).numpy(), weights.numpy()))

        return batches

    def _write_report(self, run, layers: list[str], results: dict, figures: dict[str, Path], n_batches: int) -> Path:
        doc = MarkdownDoc(title=f"Layer-wise linear probes: {run.backbone_name}")
        doc.paragraph(
            f"Ridge probes (λ={self.config.ridge_lambda}, {self.config.n_folds}-fold cross-validation) on {len(layers)} layers over {n_batches} '{self.config.split}' batches: "
            "held-out R² of a linear readout predicting each target channel from each layer's features at sampled pixels. Rising curves show where a target becomes linearly decodable inside the network. "
            "The input-pixels baseline probes the raw input channels at the same sampled pixels: layers below it destroy linearly available information. "
            "The shuffled-targets control reruns the best layer's probe on permuted targets and should sit near zero; anything higher marks probe leakage. "
            "The participation ratio is the effective number of feature dimensions the layer spreads its variance over."
        )

        entries = results["layers"]

        doc.heading("Decodability", level=2)
        doc.paragraph(f"Layers wider than {FeatureSampler.MAX_FEATURES} channels are probed on {FeatureSampler.MAX_FEATURES} evenly spaced channels; their Channels cell reads 'sampled of width', and their effective dimensionality is that of the sample.")
        table = MarkdownTable(("#", "Layer", "Channels", "Eff. dims") + tuple(run.target_channels))
        for index, layer in enumerate(layers):
            entry    = entries[layer]
            channels = f"{entry['n_channels_sampled']} of {entry['n_channels']}" if entry["n_channels_sampled"] < entry["n_channels"] else str(entry["n_channels"])
            row      = [str(index), f"`{layer}`", channels, f"{entry['effective_dim']:.1f}"]
            for target in run.target_channels:
                score = entry["scores"][target]
                row.append(f"{score['mean']:.3f} ± {score['std']:.3f}" if score is not None else "n/a")
            table.add_row(*row)
        doc.table(table)

        if results["target_notes"]:
            doc.heading("Target notes", level=2)
            notes_table = MarkdownTable(("Target", "Note"))
            for target in run.target_channels:
                if target in results["target_notes"]:
                    notes_table.add_row(target, results["target_notes"][target])
            doc.table(notes_table)

        if results["dropped_layers"]:
            doc.heading("Dropped layers", level=2)
            dropped_table = MarkdownTable(("Layer", "Reason"))
            for layer, reason in results["dropped_layers"].items():
                dropped_table.add_row(f"`{layer}`", reason)
            doc.table(dropped_table)

        doc.heading("Baselines and controls", level=2)
        baseline = results["input_baseline"]
        control_table = MarkdownTable(("Target", "Input-pixel R²", "Shuffled control", "Control layer"))
        for target in run.target_channels:
            base    = baseline["scores"][target]
            control = results["shuffled_controls"][target]
            control_table.add_row(target, f"{base['mean']:.3f}" if base is not None else "n/a", f"{control['score']:.3f}" if control is not None else "n/a", f"`{control['layer']}`" if control is not None else "—")
        doc.table(control_table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / self.REPORT_FILENAME)

    def _analyse(self, run) -> AnalysisOutcome:
        layers = self._select_layers(run)

        core = LayerProbeCore(
            model             = run.model.module,
            layers            = layers,
            target_names      = run.target_channels,
            samples_per_batch = self.config.samples_per_batch,
            probe             = RidgeProbe(self.config.ridge_lambda, self.config.n_folds, self.config.seed),
            seed              = self.config.seed,
        )

        batches = self._batches(run)
        results = core.collect(batches)
        probed  = len(layers)
        layers  = [layer for layer in layers if layer in results["layers"]]

        if results["dropped_layers"]:
            self.logger.warning(f"{self.run_dir.name}: {len(results['dropped_layers'])}/{probed} probed layers dropped: " + "; ".join(f"{layer} ({reason})" for layer, reason in results["dropped_layers"].items()))
        for target, note in results["target_notes"].items():
            self.logger.warning(f"{self.run_dir.name}: probe target {target}: {note}")

        subsampled = [f"{layer} ({entry['n_channels']})" for layer, entry in results["layers"].items() if entry["n_channels_sampled"] < entry["n_channels"]]
        if subsampled:
            self.logger.info(f"{self.run_dir.name}: {len(subsampled)} layer(s) wider than {FeatureSampler.MAX_FEATURES} channels probed on {FeatureSampler.MAX_FEATURES} evenly spaced channels: {', '.join(subsampled)}")

        plots   = LayerProbePlots()
        figures = {}
        for target in run.target_channels:
            figures[f"{target}_by_depth"] = plots.decodability_by_depth(
                layers, results["layers"], target,
                baseline = results["input_baseline"],
                control  = results["shuffled_controls"][target],
                path     = self.output_dir / f"{target}_by_depth.png",
            )
        figures["effective_dim_by_depth"] = plots.effective_dim_by_depth(layers, results["layers"], results["input_baseline"]["effective_dim"], self.output_dir / "effective_dim_by_depth.png")

        payload = {"backbone": run.backbone_name, "split": self.config.split, "n_batches": len(batches), "layer_order": layers, **results}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        report_path = self._write_report(run, layers, results, figures, len(batches))

        first        = run.target_channels[0]
        first_scores = {layer: results["layers"][layer]["scores"][first] for layer in layers}
        scored       = [layer for layer in layers if first_scores[layer] is not None]

        if scored:
            best     = max(scored, key=lambda layer: first_scores[layer]["mean"])
            headline = f"'{first}' most decodable at '{best}' (R² {first_scores[best]['mean']:.3f})"
        else:
            headline = f"'{first}' not scored: {results['target_notes'][first]}"

        return AnalysisOutcome(payload, headline, report_path)


class LayerProbeBatch(RunBatch):

    SELECTOR_ACTION = "probe"
    SECTION_TITLE   = "Layer-wise linear probes"
    RUN_CLASS       = LayerProbeRun
