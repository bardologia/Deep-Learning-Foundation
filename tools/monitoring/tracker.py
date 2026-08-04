from __future__ import annotations

from contextlib import contextmanager
from typing     import Any, Mapping, Optional

import numpy as np
import torch


class Tracker:
    def __init__(self, writer=None, debug: bool = False) -> None:
        self.writer  = writer
        self.debug   = debug
        self._step   = 0
        self._scopes = []

    @property
    def active(self) -> bool:
        return self.writer is not None

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def advance(self, n: int = 1) -> int:
        self._step += n
        return self._step

    @contextmanager
    def scope(self, name: str):
        self._scopes.append(str(name))
        try:
            yield self
        finally:
            self._scopes.pop()

    def _tag(self, tag: str) -> str:
        return "/".join([*self._scopes, str(tag)])

    def _resolve(self, step: Optional[int]) -> int:
        return self._step if step is None else int(step)

    def _emit(self, method: str, tag: str, payload: Any, step: Optional[int], **kwargs: Any) -> None:
        if self.writer is None:
            return
        getattr(self.writer, method)(self._tag(tag), payload, self._resolve(step), **kwargs)

    def log_scalar(self, tag, value, step=None) -> None:
        self._emit("add_scalar", tag, float(value), step)

    def log_metrics(self, prefix, values: Mapping[str, Any], step=None) -> None:
        for k, v in values.items():
            try:
                self._emit("add_scalar", f"{prefix}/{k}", float(v), step)
            except (TypeError, ValueError):
                continue

    def log_staged(self, group, values: Mapping[str, Any], stage, step=None) -> None:
        for k, v in values.items():
            try:
                self._emit("add_scalar", f"{group}/{k}/{stage}", float(v), step)
            except (TypeError, ValueError):
                continue

    def log_figure(self, tag, figure, step=None) -> None:
        self._emit("add_figure", tag, figure, step, close=True)

    def log_histogram(self, tag, values, step=None, bins="auto") -> None:
        v = np.asarray(values).ravel().astype(np.float32)
        try:
            self._emit("add_histogram", tag, v, step, bins=bins)
        except (ValueError, RuntimeError):
            pass

    def log_memory(self, step=None, device=None) -> None:
        if self.writer is None or not torch.cuda.is_available():
            return

        dev = device if device is not None else torch.cuda.current_device()
        self.log_scalar("system/gpu_peak_alloc_gb", torch.cuda.max_memory_allocated(dev) / 1024 ** 3, step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class NullTracker(Tracker):
    def __init__(self, debug: bool = False) -> None:
        super().__init__(writer=None, debug=debug)
