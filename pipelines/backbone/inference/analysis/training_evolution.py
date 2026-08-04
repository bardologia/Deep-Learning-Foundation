from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pipelines.backbone.inference.analysis.run_batch import AnalysisRun, RunBatch
from pipelines.backbone.inference.probes             import PredictionCurves, ProbeWindows
from tools.data.io                                   import FileIO
from tools.reporting.markdown                        import MarkdownDoc
from tools.reporting.plotting                        import PlotBase


class EvolutionFrames(PlotBase):

    def frame(self, x_axis: np.ndarray, gt_curve: np.ndarray, pred_curve: np.ndarray, epoch: int, ylim: float) -> Image.Image:
        self._apply_style()

        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=110)
        ax.plot(x_axis, gt_curve,   color="black",   linestyle="--", linewidth=1.2, label="Ground truth")
        ax.plot(x_axis, pred_curve, color="#1f77b4", linestyle="-",  linewidth=1.5, label=f"Epoch {epoch}")

        ax.set_xlabel("Elevation [m]")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(0.0, ylim)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()

        fig.canvas.draw()
        image = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)

        return image


class TrainingEvolutionRun(AnalysisRun):

    SUMMARY_FILENAME = "training_evolution.json"
    REPORT_FILENAME  = "training_evolution.md"

    def _snapshots(self) -> list[Path]:
        snapshots = sorted((self.run_dir / "checkpoints").glob("epoch_*.pt"))
        if not snapshots:
            raise FileNotFoundError(f"No epoch snapshots under {self.run_dir / 'checkpoints'}; train with snapshot_every_n_epochs > 0 to record the trajectory")

        return snapshots

    def _batch_size(self) -> int:
        return 1

    def _gt_curve(self, run, renderer: PredictionCurves, az: int, rg: int) -> np.ndarray:
        params = np.asarray(run.dataset.gt_parameters[:, az, rg], dtype=np.float64)
        return renderer.render(params.reshape(1, -1, 1, 1))[0, :, 0, 0]

    def _predict_centers(self, run, renderer: PredictionCurves, windows: torch.Tensor, centers_local: list[tuple[int, int]]) -> list[np.ndarray]:
        params = run.model(windows)
        curves = renderer.render(params)

        return [curves[index, :, cy, cx] for index, (cy, cx) in enumerate(centers_local)]

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run       = self._load_run()
        snapshots = self._snapshots()
        renderer  = PredictionCurves(run.n_gaussians, run.x_axis)

        probes  = ProbeWindows(run, self.config.window)
        centers = probes.centers(self.config.n_azimuth_probes, self.config.n_range_probes)
        windows = probes.assemble(centers)
        half    = self.config.window // 2

        centers_local = [(half, half)] * len(centers)
        gt_curves     = [self._gt_curve(run, renderer, az, rg) for az, rg in centers]

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
        for index, (az, rg) in enumerate(centers):
            images = [frame_renderer.frame(run.x_axis, gt_curves[index], prediction, epoch, ylims[index]) for epoch, prediction in frames_by_pix[index]]

            gif_path = self.output_dir / f"evolution_az{az:04d}_rg{rg:04d}.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int(1000 / self.config.fps), loop=0)
            gifs[f"az {az}, rg {rg}"] = gif_path

        payload = {"backbone": run.backbone_name, "split": self.config.split, "epochs": epochs, "pixels": [[az, rg] for az, rg in centers]}
        FileIO.save_json(payload, self.output_dir / self.SUMMARY_FILENAME)

        doc = MarkdownDoc(title=f"Training evolution: {run.backbone_name}")
        doc.paragraph(f"Prediction at {len(centers)} probe pixels re-evaluated from {len(snapshots)} epoch snapshots ({epochs[0]}..{epochs[-1]}), one GIF per pixel against the fixed ground truth.")
        for label, path in gifs.items():
            doc.image(label, path.name)
        report_path = doc.save(self.output_dir / self.REPORT_FILENAME)

        self.logger.ok(f"{self.run_dir.name}: {len(gifs)} evolution GIFs over {len(snapshots)} snapshots -> {report_path}")

        return payload


class TrainingEvolutionBatch(RunBatch):

    SELECTOR_ACTION = "animate"
    SECTION_TITLE   = "Training evolution animation"
    RUN_CLASS       = TrainingEvolutionRun
