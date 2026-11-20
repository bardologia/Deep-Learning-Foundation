from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pipelines.backbone.inference.analysis.run_batch import AnalysisOutcome, AnalysisRun, RunBatch
from pipelines.backbone.inference.metrics            import Metrics
from pipelines.backbone.inference.predictor          import Predictor
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                        import PlotBase


class EvolutionFrames(PlotBase):

    def frame(self, prediction: np.ndarray, name: str, epoch: int, vmin: float, vmax: float) -> Image.Image:
        self._apply_style()

        fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=110)
        image   = ax.imshow(prediction, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")

        ax.set_xlabel("col (pixel)")
        ax.set_ylabel("row (pixel)")
        ax.set_title(f"{name}: prediction at epoch {epoch}")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02).set_label(name)
        fig.tight_layout()

        fig.canvas.draw()
        frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)

        return frame


class TrainingEvolutionRun(AnalysisRun):

    SUMMARY_FILENAME = "training_evolution.json"
    REPORT_FILENAME  = "training_evolution.md"

    def _snapshots(self) -> list[Path]:
        return sorted((self.run_dir / "checkpoints").glob("epoch_*.pt"))

    def _replay(self, run, snapshots: list[Path]) -> tuple[list[int], list[np.ndarray]]:
        module     = run.model.module
        best_state = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
        epochs     = []
        maps       = []

        with self.logger.timer(f"Training evolution: snapshot replay ({len(snapshots)} snapshots)"):
            try:
                for snapshot_path in snapshots:
                    payload = torch.load(snapshot_path, map_location=self.config.device, weights_only=False)
                    module.load_state_dict(payload["params"])

                    epochs.append(int(payload["epoch"]) + 1)
                    maps.append(Predictor(run, self.logger).predict())
            finally:
                module.load_state_dict(best_state)

        return epochs, maps

    def _render_gifs(self, run, epochs: list[int], maps: list[np.ndarray]) -> dict[str, Path]:
        frames = EvolutionFrames()
        gifs   = {}

        with self.logger.timer(f"Training evolution: frame rendering ({len(epochs) * run.out_channels} frames)"):
            for index, name in enumerate(run.target_channels):
                vmin, vmax = PlotBase._shared_clim(run.targets[index], *[prediction[index] for prediction in maps])
                images     = [frames.frame(prediction[index], name, epoch, vmin, vmax) for epoch, prediction in zip(epochs, maps)]
                gif_path   = self.output_dir / f"evolution_{name}.gif"

                images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int(1000 / self.config.fps), loop=0)
                gifs[name] = gif_path

        return gifs

    def _analyse(self, run) -> AnalysisOutcome:
        snapshots = self._snapshots()

        self.logger.info(f"{self.run_dir.name}: replaying {len(snapshots)} snapshots ({snapshots[0].stem} .. {snapshots[-1].stem}) over the '{self.config.split}' split")

        epochs, maps = self._replay(run, snapshots)
        gifs         = self._render_gifs(run, epochs, maps)
        rmse         = {name: [Metrics.channel(prediction[index], run.targets[index])["rmse"] for prediction in maps] for index, name in enumerate(run.target_channels)}

        payload = {"backbone": run.backbone_name, "split": self.config.split, "epochs": epochs, "rmse": rmse}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        table = MarkdownTable(("Epoch", *run.target_channels))
        for row, epoch in enumerate(epochs):
            table.add_row(str(epoch), *[f"{rmse[name][row]:.4g}" for name in run.target_channels])

        doc = MarkdownDoc(title=f"Training evolution: {run.backbone_name}")
        doc.paragraph(f"Prediction of the '{self.config.split}' split re-evaluated from {len(snapshots)} epoch snapshots ({epochs[0]}..{epochs[-1]}), one GIF per target channel on the colour scale of its targets, with the RMSE of every snapshot in target units.")
        doc.table(table)
        for name, path in gifs.items():
            doc.image(name, path.name)
        report_path = doc.save(self.output_dir / self.REPORT_FILENAME)

        return AnalysisOutcome(payload, f"{len(gifs)} evolution GIFs over {len(snapshots)} snapshots", report_path)


class TrainingEvolutionBatch(RunBatch):

    SELECTOR_ACTION = "animate"
    SECTION_TITLE   = "Training evolution animation"
    RUN_CLASS       = TrainingEvolutionRun
