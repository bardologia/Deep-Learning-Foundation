"""Exports the per-slice raw-intensity SSIM stored in an inference stamp's metrics.json as one .npy curve per image axis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


class SliceSsimExporter:
    """Gathers the per-slice SSIM of the prediction against the labels in raw intensity into one array per image axis.

    Inference scores SSIM on every column and row slice of the region it ran on and
    stores each value under its own metrics.json key, ssim_gt_<axis>_<i>, where i is the
    slice position inside that region. The exported array keeps that order, so entry i is
    column i or row i of the region; the region's global position is the split_region
    entry of the same metrics.json, as (row_start, row_end, col_start, col_end).
    Non-finite slices are kept as NaN.

    Attributes:
        AXES         : Exported axis labels mapped to the (start, end) positions of that axis in split_region.
        metrics_path : metrics.json of one inference stamp.
        logger       : Logger reporting each written file.
    """

    AXES = {"col": (2, 3), "row": (0, 1)}

    def __init__(self, metrics_path: Path, logger: Logger) -> None:
        """Binds the exporter to one metrics file.

        Args:
            metrics_path : metrics.json of one inference stamp; the curves are written beside it.
            logger       : Logger reporting each written file.
        """
        self.metrics_path = Path(metrics_path)
        self.logger       = logger

    def _curve(self, metrics: dict, axis: str) -> np.ndarray:
        """Returns the SSIM of every slice along one axis, in slice order.

        Args:
            metrics : Loaded metrics.json payload.
            axis    : Axis label, "col" or "row".

        Returns:
            SSIM per slice, of shape (n_slices,), where n_slices is the region's extent along the axis.

        Raises:
            KeyError   : If a slice inside the region has no stored SSIM.
            ValueError : If the file stores SSIM for more slices than the region holds along the axis.
        """
        start, end = self.AXES[axis]
        region     = metrics["split_region"]
        n_slices   = region[end] - region[start]

        if f"ssim_gt_{axis}_{n_slices}" in metrics:
            raise ValueError(f"{self.metrics_path} stores ssim_gt_{axis}_{n_slices}, past the {n_slices} {axis} slices of split_region {region}; the per-slice keys do not match the scored region")

        return np.array([metrics[f"ssim_gt_{axis}_{i}"] for i in range(n_slices)], dtype=np.float64)

    def run(self) -> None:
        """Loads the metrics file and writes ssim_gt_col.npy and ssim_gt_row.npy beside it."""
        metrics = FileIO.load_json(self.metrics_path)

        for axis in self.AXES:
            curve = self._curve(metrics, axis)
            path  = self.metrics_path.parent / f"ssim_gt_{axis}.npy"

            np.save(path, curve)
            self.logger.info(f"{path}: {curve.size} {axis} slices, mean SSIM {np.nanmean(curve):.4f} ({int(np.isnan(curve).sum())} non-finite)")


def main() -> None:
    """Parses the command line and exports the curves of every given metrics file."""
    parser = argparse.ArgumentParser(description="Save the per-slice raw-intensity SSIM (prediction vs labels, ssim_gt_<axis>_<i>) of inference stamps as ssim_gt_col.npy and ssim_gt_row.npy beside each metrics.json.")
    parser.add_argument("paths", type=str, nargs="+", help="metrics.json files of inference stamps.")
    args = parser.parse_args()

    with Logger(log_dir="logs", name="export_slice_ssim") as logger:
        for path in args.paths:
            SliceSsimExporter(Path(path), logger).run()


if __name__ == "__main__":
    main()
