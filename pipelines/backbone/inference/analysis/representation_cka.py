from __future__ import annotations

from itertools import combinations
from pathlib   import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from configuration.diagnostics                          import CkaConfig
from pipelines.backbone.inference.analysis.layer_probes import FeatureSampler
from pipelines.backbone.inference.loader                import RunLoader
from pipelines.backbone.inference.probes                import ModelDevice
from tools.data.io                                      import FileIO
from tools.diagnostics.activation_recorder              import ActivationRecorder
from tools.reporting.markdown                           import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                           import PlotBase
from tools.runtime.run_selector                         import RunSelector
from tools.runtime.run_tag                              import RunTag


class CkaComputation:

    @staticmethod
    def linear_cka(features_a: np.ndarray, features_b: np.ndarray) -> float:
        X = np.asarray(features_a, dtype=np.float64)
        Y = np.asarray(features_b, dtype=np.float64)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"CKA needs matched samples, got {X.shape[0]} and {Y.shape[0]}")
        if X.shape[0] < 4:
            raise ValueError(f"CKA needs at least 4 samples, got {X.shape[0]}")

        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        cross  = float(np.linalg.norm(Y.T @ X, ord="fro") ** 2)
        norm_x = float(np.linalg.norm(X.T @ X, ord="fro"))
        norm_y = float(np.linalg.norm(Y.T @ Y, ord="fro"))

        if norm_x <= 0.0 or norm_y <= 0.0:
            return 0.0

        return cross / (norm_x * norm_y)

    @classmethod
    def cross_matrix(cls, layers_a: dict[str, np.ndarray], layers_b: dict[str, np.ndarray]) -> np.ndarray:
        names_a = list(layers_a)
        names_b = list(layers_b)

        matrix = np.zeros((len(names_a), len(names_b)))
        for i, name_a in enumerate(names_a):
            for j, name_b in enumerate(names_b):
                matrix[i, j] = cls.linear_cka(layers_a[name_a], layers_b[name_b])

        return matrix

    @staticmethod
    def alignment_score(matrix: np.ndarray) -> float:
        return float((matrix.max(axis=1).mean() + matrix.max(axis=0).mean()) / 2.0)


class CkaPlots(PlotBase):

    def pair_heatmap(self, matrix: np.ndarray, name_a: str, name_b: str, path: Path) -> Path:
        return self._imshow_figure(
            matrix,
            x_label        = f"{name_b} layers (forward order)",
            y_label        = f"{name_a} layers (forward order)",
            title          = f"Linear CKA: {name_a} vs {name_b}",
            cmap           = self._cmap_with_bad("magma"),
            vmin           = 0.0,
            vmax           = 1.0,
            colorbar_label = "CKA",
            path           = path,
        )

    def summary_heatmap(self, matrix: np.ndarray, names: list[str], path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.9))
        im      = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Representation alignment across runs")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("Mean best-match CKA")

        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] < 0.6 else "black")

        fig.tight_layout()
        return self._save(fig, path)


class CkaComparison:

    def __init__(self, config: CkaConfig, logger) -> None:
        self.config = config
        self.logger = logger

        self.output_dir = Path(config.output_dir) / RunTag.now()

    def _select_runs(self) -> list[Path]:
        selector = RunSelector(self.config.runs_dir, self.config.checkpoint_name, self.logger, action="compare")
        run_dirs = selector.resolve(self.config.run_filter)

        if len(run_dirs) < 2:
            raise ValueError(f"CKA needs at least two runs, got {len(run_dirs)}")

        return run_dirs

    def _load(self, run_dir: Path):
        return RunLoader(run_dir, logger=self.logger).load(
            split           = self.config.split,
            batch_size      = self.config.batch_size,
            num_workers     = 0,
            device          = self.config.device,
            checkpoint_name = self.config.checkpoint_name,
            load_tomogram   = False,
        )

    @staticmethod
    def _sample_grid(run) -> tuple:
        return (run.split_region.as_tuple(), tuple(run.dataset_config.patch.size), tuple(run.dataset_config.patch.stride))

    def _validate_alignment(self, grids: list[tuple]) -> None:
        regions = {grid[0] for grid in grids}
        patches = {grid[1] for grid in grids}
        strides = {grid[2] for grid in grids}

        if len(regions) != 1 or len(patches) != 1 or len(strides) != 1:
            raise ValueError(f"Runs are not sample-aligned: regions {sorted(regions)}, patches {sorted(patches)}, strides {sorted(strides)}; CKA needs identical split regions and patch grids")

    def _probe_layers(self, run) -> list[str]:
        names = ActivationRecorder(run.model.module).leaf_names()

        if len(names) <= self.config.max_layers:
            return names

        keep = np.linspace(0, len(names) - 1, self.config.max_layers).astype(int)
        return [names[index] for index in sorted(set(keep))]

    def _collect(self, run) -> dict[str, np.ndarray]:
        layers   = self._probe_layers(run)
        sampler  = FeatureSampler(self.config.samples_per_batch, seed=self.config.sample_seed)
        recorder = ActivationRecorder(run.model.module)
        recorder.attach_store(layers)

        chunks: dict[str, list] = {layer: [] for layer in layers}

        for index, batch in enumerate(run.loader):
            if index >= self.config.max_batches:
                break

            images = batch[0]
            with torch.no_grad():
                run.model.module(images.to(ModelDevice.of(run.model.module)))

            stored     = recorder.stored()
            B, _, H, W = images.shape
            b, i, j    = sampler.sample_coords(B, H, W)

            for layer in layers:
                tensor = stored.get(layer)
                if tensor is None or not sampler.is_feature_map(tensor, B, H, W):
                    chunks.pop(layer, None)
                    continue
                if layer in chunks:
                    chunks[layer].append(sampler.features_at(tensor, b, i, j, B, H, W))

        recorder.detach()

        if not chunks:
            raise ValueError(f"Run {run.backbone_name} produced no (B, C, H, W) feature map on the input grid to compare")

        return {layer: np.concatenate(parts, axis=0) for layer, parts in chunks.items()}

    def _write_report(self, names: list[str], summary: np.ndarray, pair_figures: dict, summary_figure: Path) -> Path:
        doc = MarkdownDoc(title="Representation similarity (linear CKA)")
        doc.paragraph(
            f"Linear CKA between {len(names)} runs on identical sampled pixels of the '{self.config.split}' split. "
            "The summary cell is the mean best-match CKA across layers (1 = every layer of one run has a near-identical "
            "counterpart in the other); per-pair heatmaps show the full cross-layer structure."
        )

        table = MarkdownTable(("Run", *[name for name in names]))
        for i, name in enumerate(names):
            table.add_row(f"`{name}`", *[f"{summary[i, j]:.3f}" for j in range(len(names))])
        doc.table(table)

        doc.image("alignment", str(summary_figure.relative_to(self.output_dir)))
        for (name_a, name_b), path in pair_figures.items():
            doc.image(f"{name_a} vs {name_b}", str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / "cka_report.md")

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run_dirs = self._select_runs()
        names    = ["/".join(run_dir.relative_to(self.config.runs_dir).parts) if run_dir.is_relative_to(self.config.runs_dir) else run_dir.name for run_dir in run_dirs]

        grids    = []
        features = []
        for run_dir in run_dirs:
            run = self._load(run_dir)

            grids.append(self._sample_grid(run))
            self._validate_alignment(grids)

            features.append(self._collect(run))
            del run

        plots        = CkaPlots()
        summary      = np.eye(len(names))
        pair_figures = {}
        pair_scores  = {}

        for (i, j) in combinations(range(len(names)), 2):
            matrix = CkaComputation.cross_matrix(features[i], features[j])
            score  = CkaComputation.alignment_score(matrix)

            summary[i, j] = summary[j, i] = score
            pair_scores[f"{names[i]}|{names[j]}"] = score
            pair_figures[(names[i], names[j])]    = plots.pair_heatmap(matrix, names[i], names[j], self.output_dir / "pairs" / f"{i}_{j}.png")

        summary_figure = plots.summary_heatmap(summary, names, self.output_dir / "alignment.png")

        payload = {"runs": names, "split": self.config.split, "alignment": summary.tolist(), "pair_scores": pair_scores}
        FileIO.save_json(payload, self.output_dir / "cka.json")

        report_path = self._write_report(names, summary, pair_figures, summary_figure)
        self.logger.ok(f"CKA over {len(names)} runs -> {report_path}")

        return payload
