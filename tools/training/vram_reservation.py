from __future__ import annotations

import torch

from tools.monitoring.logger import Logger


class VramReservation:

    GRANULARITY_BYTES = 2 * 1024 ** 2
    CHUNK_FLOOR_BYTES = 16 * 1024 ** 2

    def __init__(self, enabled: bool, keep_free_gb: float, device: torch.device, logger: Logger) -> None:
        self.enabled         = bool(enabled) and device.type == "cuda"
        self.keep_free_bytes = int(float(keep_free_gb) * 1024 ** 3)
        self.device          = device
        self.logger          = logger
        self.filled          = False

        if bool(enabled) and not self.enabled:
            self.logger.subsection("VRAM reservation requested (memory.reserve_vram=True) but the device is not CUDA; no memory is parked.")

    def _free_bytes(self) -> int:
        free, _total = torch.cuda.mem_get_info(self.device)
        return free

    def _claim(self) -> int:
        chunks  = []
        claimed = 0
        shrink  = 1

        while True:
            spare = self._free_bytes() - self.keep_free_bytes
            if spare < self.CHUNK_FLOOR_BYTES:
                break

            chunk = (spare // shrink // self.GRANULARITY_BYTES) * self.GRANULARITY_BYTES
            if chunk < self.CHUNK_FLOOR_BYTES:
                break

            try:
                chunks.append(torch.empty(chunk, dtype=torch.uint8, device=self.device))
                claimed += chunk
                shrink   = 1
            except torch.cuda.OutOfMemoryError:
                shrink *= 2

        del chunks
        return claimed

    def fill(self) -> None:
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)

        claimed     = self._claim()
        self.filled = True

        free, total = torch.cuda.mem_get_info(self.device)
        self.logger.subsection(f"VRAM reservation: parked {claimed / 1024 ** 3:.2f} GB in the allocator cache, {free / 1024 ** 3:.2f} GB of {total / 1024 ** 3:.2f} GB left free (keep-free target {self.keep_free_bytes / 1024 ** 3:.2f} GB)")

    def refill(self) -> None:
        if not (self.enabled and self.filled):
            return

        self._claim()
