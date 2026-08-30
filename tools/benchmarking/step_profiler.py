"""Kernel-level profiling of full training steps with torch.profiler.

`StepProfiler` drives forward, backward and optimizer steps of a feed target
under the profiler schedule and writes three artifacts: a chrome trace for
chrome://tracing or Perfetto, a plain-text operator table sorted by self time,
and a machine-readable summary of the top operators.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader

from tools.data.io import FileIO


class StepProfiler:
    """Profiles training steps of one feed target and writes the trace artifacts.

    Attributes:
        TOP_OPS         : Number of operators kept in the machine-readable summary.
        TABLE_ROWS      : Number of rows rendered in the plain-text operator table.
        feed            : Feed target providing the dataset, model, input mover and loss.
        device          : Device the profiled steps run on.
        batch_size      : Samples per profiled step.
        num_workers     : Loader worker processes.
        prefetch_factor : Batches prefetched per worker.
        use_amp         : Whether the forward runs under bfloat16 autocast.
        warmup_steps    : Steps the profiler observes without recording.
        active_steps    : Steps recorded into the trace.
        record_shapes   : Whether operator input shapes are recorded.
        profile_memory  : Whether allocator events are recorded.
        with_stack      : Whether Python stacks are recorded (large traces).
        logger          : Logger receiving the progress and artifact lines.
    """

    TOP_OPS    = 25
    TABLE_ROWS = 50

    def __init__(self, feed, device: torch.device, batch_size: int, num_workers: int, prefetch_factor: int, use_amp: bool, warmup_steps: int, active_steps: int, record_shapes: bool, profile_memory: bool, with_stack: bool, logger) -> None:
        """Stores the feed target and every profiling knob."""
        self.feed            = feed
        self.device          = device
        self.batch_size      = int(batch_size)
        self.num_workers     = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.use_amp         = bool(use_amp)
        self.warmup_steps    = int(warmup_steps)
        self.active_steps    = int(active_steps)
        self.record_shapes   = bool(record_shapes)
        self.profile_memory  = bool(profile_memory)
        self.with_stack      = bool(with_stack)
        self.logger          = logger

    def _loader(self) -> DataLoader:
        """Returns the loader feeding the profiled steps."""
        return DataLoader(
            self.feed.dataset,
            batch_size      = self.batch_size,
            shuffle         = True,
            num_workers     = self.num_workers,
            pin_memory      = self.device.type == "cuda",
            prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None,
            drop_last       = True,
        )

    def _batches(self, loader: DataLoader, total: int):
        """Yields exactly `total` batches, restarting the loader between epochs.

        Raises:
            ValueError : If the loader yields no batch at all.
        """
        if len(loader) == 0:
            raise ValueError(f"The profiling loader yields zero batches: {len(self.feed.dataset)} samples at batch size {self.batch_size} with drop_last; lower the batch size")

        produced = 0
        while produced < total:
            for batch in loader:
                yield batch
                produced += 1
                if produced == total:
                    return

    def _activities(self) -> list:
        """Returns the profiler activities: CPU always, CUDA when profiling on a GPU."""
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        return activities

    def _summary(self, prof, trace_path: Path, table_path: Path) -> dict:
        """Returns the machine-readable summary of the profiled steps.

        Args:
            prof       : Finished profiler.
            trace_path : Written chrome trace path.
            table_path : Written operator table path.
        """
        events = prof.key_averages()
        cuda   = self.device.type == "cuda"

        ranked = sorted(events, key=lambda e: (e.self_device_time_total if cuda else e.self_cpu_time_total), reverse=True)

        top = []
        for event in ranked[: self.TOP_OPS]:
            top.append({
                "name"                    : event.key,
                "count"                   : int(event.count),
                "self_cpu_time_us"        : float(event.self_cpu_time_total),
                "cpu_time_us"             : float(event.cpu_time_total),
                "self_device_time_us"     : float(event.self_device_time_total),
                "device_time_us"          : float(event.device_time_total),
            })

        return {
            "device"         : str(self.device),
            "model"          : self.feed.model_name,
            "batch_size"     : self.batch_size,
            "num_workers"    : self.num_workers,
            "use_amp"        : self.use_amp,
            "warmup_steps"   : self.warmup_steps,
            "active_steps"   : self.active_steps,
            "top_operators"  : top,
            "trace"          : str(trace_path),
            "operator_table" : str(table_path),
        }

    def run(self, output_dir: Path) -> dict:
        """Profiles the configured steps and writes trace, table and summary.

        Args:
            output_dir : Directory receiving trace.json, op_table.txt and summary.json.

        Returns:
            The summary payload, also written as summary.json.
        """
        output_dir = FileIO.ensure_dir(Path(output_dir))

        model     = self.feed.model.to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        loader = self._loader()
        total  = self.warmup_steps + self.active_steps

        self.logger.section("[Step Profiler]")
        self.logger.kv_table({
            "Model"       : self.feed.model_name,
            "Device"      : self.device,
            "Batch size"  : self.batch_size,
            "Steps"       : f"{self.warmup_steps} warmup + {self.active_steps} recorded",
            "AMP"         : self.use_amp,
            "Sample feed" : self.feed.sample_text,
        })

        with profile(
            activities     = self._activities(),
            schedule       = schedule(wait=0, warmup=self.warmup_steps, active=self.active_steps, repeat=1),
            record_shapes  = self.record_shapes,
            profile_memory = self.profile_memory,
            with_stack     = self.with_stack,
        ) as prof:
            for batch in self._batches(loader, total):
                model_input = self.feed.to_model_input(batch, self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    loss = self.feed.forward_loss(model, model_input)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                prof.step()

        trace_path = output_dir / "trace.json"
        table_path = output_dir / "op_table.txt"

        prof.export_chrome_trace(str(trace_path))

        sort_key = "self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
        table_path.write_text(prof.key_averages().table(sort_by=sort_key, row_limit=self.TABLE_ROWS), encoding="utf-8")

        summary = self._summary(prof, trace_path, table_path)
        FileIO.save_json(summary, output_dir / "summary.json")

        self.logger.kv_table({
            "Chrome trace"   : trace_path,
            "Operator table" : table_path,
            "Summary"        : output_dir / "summary.json",
        })

        return summary
