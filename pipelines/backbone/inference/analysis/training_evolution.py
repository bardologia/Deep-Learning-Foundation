"""Animation of how a run's prediction at probe pixels evolves during training.

Each saved epoch snapshot is reloaded into the trained model, evaluated at a set
of probe pixels, and rendered as one GIF frame against the fixed ground-truth
profile; the best weights are restored afterwards.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pipelines.backbone.inference.analysis.run_batch import AnalysisOutcome, AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import PredictionCurves, ProbeWindows
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc
from tools.reporting.plotting                        import PlotBase


class EvolutionFrames(PlotBase):
    """Renders single GIF frames comparing a predicted profile to ground truth."""

    def frame(self, z_axis: np.ndarray, gt_curve: np.ndarray, pred_curve: np.ndarray, epoch: int, ylim: float) -> Image.Image:
        """Renders one animation frame for a probe pixel at one epoch.

        Args:
            z_axis     : Z axis in metres, shape (n_z,).
            gt_curve   : Ground-truth amplitude profile, shape (n_z,).
            pred_curve : Predicted amplitude profile, shape (n_z,).
            epoch      : One-based epoch number of the snapshot.
            ylim       : Upper amplitude limit held fixed across the animation.

        Returns:
            The rendered frame as an RGB image.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=110)
        ax.plot(z_axis, gt_curve,   color="black",   linestyle="--", linewidth=1.2, label="Ground truth")
        ax.plot(z_axis, pred_curve, color="#1f77b4", linestyle="-",  linewidth=1.5, label=f"Epoch {epoch}")

        ax.set_xlabel("Z")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(0.0, ylim)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()

        fig.canvas.draw()
        image = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)

        return image


class TrainingEvolutionRun(AnalysisRun):
    """Builds per-pixel training-evolution GIFs from one run's epoch snapshots."""

    SUMMARY_FILENAME = "training_evolution.json"
    REPORT_FILENAME  = "training_evolution.md"

    def _snapshots(self) -> list[Path]:
        """Returns the epoch snapshot checkpoints in epoch order.

        Raises:
            FileNotFoundError : If the run saved no epoch snapshots.
        """
        snapshots = sorted((self.run_dir / "checkpoints").glob("epoch_*.pt"))
        if not snapshots:
            raise FileNotFoundError(f"No epoch snapshots under {self.run_dir / 'checkpoints'}; train with snapshot_every_n_epochs > 0 to record the trajectory")

        return snapshots

    def _batch_size(self) -> int:
        """Returns the loader batch size, fixed at one probe window per forward pass."""
        return 1

    def _gt_curve(self, run, renderer: PredictionCurves, row: int, col: int) -> np.ndarray:
        """Returns the ground-truth profile at one pixel, shape (n_z,).

        Args:
            run      : Loaded run providing the ground-truth Gaussian parameters.
            renderer : Renderer producing profiles from parameters.
            row       : Row index of the probe pixel.
            col       : Range index of the probe pixel.
        """
        params = np.asarray(run.dataset.gt_parameters[:, row, col], dtype=np.float64)
        return renderer.render(params.reshape(1, -1, 1, 1))[0, :, 0, 0]

    def _predict_centers(self, run, renderer: PredictionCurves, windows: torch.Tensor, centers_local: list[tuple[int, int]]) -> list[np.ndarray]:
        """Predicts the profile at the centre of each probe window.

        Args:
            run           : Loaded run providing the model.
            renderer      : Renderer producing profiles from parameters.
            windows       : Probe windows of shape (probes, channels, window, window).
            centers_local : Window-local (row, column) of each probe centre.

        Returns:
            One profile of shape (n_z,) per probe window.
        """
        params = run.model(windows)
        curves = renderer.render(params)

        return [curves[index, :, cy, cx] for index, (cy, cx) in enumerate(centers_local)]

    def _analyse(self, run) -> AnalysisOutcome:
        """Replays every epoch snapshot at the probe pixels and writes GIFs and report.

        Args:
            run : Loaded run bundling the model and the dataset.

        Returns:
            The outcome whose payload holds the backbone name, split, replayed epochs
            and the probe pixel coordinates.
        """
        snapshots = self._snapshots()
        renderer  = PredictionCurves(run.n_gaussians, run.z_axis)

        probes  = ProbeWindows(run, self.config.window)
        centers = probes.centers(self.config.n_row_probes, self.config.n_col_probes)
        windows = probes.assemble(centers)
        half    = self.config.window // 2

        centers_local = [(half, half)] * len(centers)
        gt_curves     = [self._gt_curve(run, renderer, row, col) for row, col in centers]

        module     = run.model.module
        best_state = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}

        epochs         = []
        frames_by_pix  = [[] for _ in centers]
        frame_renderer = EvolutionFrames()
        ylims          = [max(float(curve.max()) * 1.5, 1e-6) for curve in gt_curves]

        try:
            for snapshot_path in snapshots:
                payload = torch.load(snapshot_path, map_location=self.config.device, weights_only=False)
                module.load_state_dict(payload["params"])
                epoch = int(payload["epoch"]) + 1
                epochs.append(epoch)

                predictions = self._predict_centers(run, renderer, windows, centers_local)
                for index, prediction in enumerate(predictions):
                    ylims[index] = max(ylims[index], float(prediction.max()) * 1.05)
                    frames_by_pix[index].append((epoch, prediction))
        finally:
            module.load_state_dict(best_state)

        gifs = {}
        for index, (row, col) in enumerate(centers):
            images = [frame_renderer.frame(run.z_axis, gt_curves[index], prediction, epoch, ylims[index]) for epoch, prediction in frames_by_pix[index]]

            gif_path = self.output_dir / f"evolution_row{row:04d}_col{col:04d}.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int(1000 / self.config.fps), loop=0)
            gifs[f"row {row}, col {col}"] = gif_path

        payload = {"backbone": run.backbone_name, "split": self.config.split, "epochs": epochs, "pixels": [[row, col] for row, col in centers]}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        doc = MarkdownDoc(title=f"Training evolution: {run.backbone_name}")
        doc.paragraph(f"Prediction at {len(centers)} probe pixels re-evaluated from {len(snapshots)} epoch snapshots ({epochs[0]}..{epochs[-1]}), one GIF per pixel against the fixed ground truth.")
        for label, path in gifs.items():
            doc.image(label, path.name)
        report_path = doc.save(self.output_dir / self.REPORT_FILENAME)

        return AnalysisOutcome(payload, f"{len(gifs)} evolution GIFs over {len(snapshots)} snapshots", report_path)


class TrainingEvolutionBatch(RunBatch):
    """Builds training-evolution animations for every selected run."""

    SELECTOR_ACTION = "animate"
    SECTION_TITLE   = "Training evolution animation"
    RUN_CLASS       = TrainingEvolutionRun
