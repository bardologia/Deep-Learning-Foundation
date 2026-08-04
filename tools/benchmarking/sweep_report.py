from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas             as pd

matplotlib.use("Agg")


class SweepReport:
    def __init__(self, results: list[dict], wait_threshold: float = 0.05) -> None:
        self.wait_threshold = float(wait_threshold)
        self.frame          = pd.DataFrame(results)
        self.ok_frame       = self.frame[self.frame["status"] == "ok"].copy()

    @property
    def dataframe(self):
        return self.frame

    def _saturated(self):
        if self.ok_frame.empty:
            return self.ok_frame

        return self.ok_frame[(self.ok_frame["data_wait_fraction"] <= self.wait_threshold) | (self.ok_frame["feed_ratio"] >= 1.0)]

    @property
    def recommendation(self) -> dict:
        if self.ok_frame.empty:
            return {"found": False, "reason": "no successful configurations"}

        saturated = self._saturated()
        cpu_bound = saturated.empty

        pool = self.ok_frame if cpu_bound else saturated
        pool = pool.sort_values(by=["end_to_end_samples_per_s", "num_workers", "batch_size"], ascending=[False, True, True])
        best = pool.iloc[0]

        return {
            "found"                    : True,
            "cpu_bound"                : bool(cpu_bound),
            "batch_size"               : int(best["batch_size"]),
            "num_workers"              : int(best["num_workers"]),
            "prefetch_factor"          : int(best["prefetch_factor"]),
            "pin_memory"               : bool(best["pin_memory"]),
            "persistent_workers"       : bool(best["persistent_workers"]),
            "end_to_end_samples_per_s" : float(best["end_to_end_samples_per_s"]),
            "data_wait_fraction"       : float(best["data_wait_fraction"]),
            "gpu_util_mean"            : float(best["gpu_util_mean"]),
            "feed_ratio"               : float(best["feed_ratio"]),
        }

    def _new_axes(self):
        figure, axes = plt.subplots(figsize=(7.0, 4.5))
        return figure, axes

    def _save(self, figure, axes, fig_dir: Path, name: str) -> Path:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        out_path = fig_dir / name
        figure.savefig(out_path)
        plt.close(figure)

        return out_path

    def plot_throughput_vs_batch(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        for workers, group in self.ok_frame.groupby("num_workers"):
            ordered = group.sort_values("batch_size")
            axes.plot(ordered["batch_size"], ordered["end_to_end_samples_per_s"], marker="o", label=f"{workers} workers")

        axes.set_xscale("log", base=2)
        axes.set_xlabel("Batch size (samples)")
        axes.set_ylabel("End-to-end throughput (samples/s)")
        axes.set_title("End-to-end training throughput vs batch size")
        axes.legend(title="DataLoader workers", bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)

        return self._save(figure, axes, fig_dir, "throughput_vs_batch_size.png")

    def plot_data_wait_vs_workers(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        for batch_size, group in self.ok_frame.groupby("batch_size"):
            ordered = group.sort_values("num_workers")
            axes.plot(ordered["num_workers"], 100.0 * ordered["data_wait_fraction"], marker="o", label=f"batch {batch_size}")

        axes.axhline(100.0 * self.wait_threshold, color="black", linestyle="--", linewidth=1.0, label=f"{100.0 * self.wait_threshold:.0f}% target")
        axes.set_xlabel("DataLoader workers")
        axes.set_ylabel("GPU idle waiting for data (%)")
        axes.set_title("Data-starvation share of step time vs worker count")
        axes.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)

        return self._save(figure, axes, fig_dir, "data_wait_vs_workers.png")

    def plot_gpu_util_vs_throughput(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        scatter = axes.scatter(
            self.ok_frame["end_to_end_samples_per_s"],
            self.ok_frame["gpu_util_mean"],
            c=self.ok_frame["num_workers"],
            cmap="viridis",
            s=55,
        )

        axes.set_xlabel("End-to-end throughput (samples/s)")
        axes.set_ylabel("Mean GPU utilization (%)")
        axes.set_title("Achieved GPU utilization vs throughput")

        colorbar = figure.colorbar(scatter, ax=axes)
        colorbar.set_label("DataLoader workers")

        return self._save(figure, axes, fig_dir, "gpu_util_vs_throughput.png")

    def plot_feed_ratio_vs_workers(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        for batch_size, group in self.ok_frame.groupby("batch_size"):
            ordered = group.sort_values("num_workers")
            axes.plot(ordered["num_workers"], ordered["feed_ratio"], marker="o", label=f"batch {batch_size}")

        axes.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="GPU-bound threshold")
        axes.set_xlabel("DataLoader workers")
        axes.set_ylabel("Feed ratio (loader / GPU ceiling)")
        axes.set_title("CPU feed capacity relative to the GPU compute ceiling")
        axes.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)

        return self._save(figure, axes, fig_dir, "feed_ratio_vs_workers.png")

    def save_all(self, fig_dir: Path) -> list[Path]:
        return [
            self.plot_throughput_vs_batch(fig_dir),
            self.plot_data_wait_vs_workers(fig_dir),
            self.plot_gpu_util_vs_throughput(fig_dir),
            self.plot_feed_ratio_vs_workers(fig_dir),
        ]
