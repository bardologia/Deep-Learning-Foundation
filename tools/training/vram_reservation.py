"""Parks spare GPU memory in the PyTorch allocator cache to reserve it for a run.

Allocating and freeing large uint8 blocks leaves them in the caching allocator, so
the memory stays claimed from the driver's point of view and other processes cannot
take it while the run is in flight.
"""

from __future__ import annotations

import torch

from tools.monitoring.logger import Logger


class VramReservation:
    """Claims free VRAM into the allocator cache, leaving a configured headroom.

    Attributes:
        enabled         : True only when reservation was requested and the device is CUDA.
        keep_free_bytes : Amount of device memory deliberately left unclaimed.
        device          : Device whose memory is reserved.
        logger          : Logger used for the request and outcome messages.
    """

    GRANULARITY_BYTES = 2 * 1024 ** 2
    CHUNK_FLOOR_BYTES = 16 * 1024 ** 2

    def __init__(self, enabled: bool, keep_free_gb: float, device: torch.device, logger: Logger) -> None:
        """Configures the reservation, disabling it on non-CUDA devices.

        Args:
            enabled      : Whether reservation was requested by the config.
            keep_free_gb : Device memory to leave free, in gibibytes.
            device       : Device to reserve memory on.
            logger       : Logger used to report that a non-CUDA device parks nothing.
        """
        self.enabled         = bool(enabled) and device.type == "cuda"
        self.keep_free_bytes = int(float(keep_free_gb) * 1024 ** 3)
        self.device          = device
        self.logger          = logger

        if bool(enabled) and not self.enabled:
            self.logger.subsection("VRAM reservation requested (memory.reserve_vram=True) but the device is not CUDA; no memory is parked.")

    def _free_bytes(self) -> int:
        """Returns the driver-reported free memory on the device, in bytes."""
        free, _total = torch.cuda.mem_get_info(self.device)
        return free

    def _claim(self) -> int:
        """Allocates and releases uint8 blocks until only the headroom is left free.

        Each block is sized to the spare memory above the keep-free target, rounded
        down to the allocator granularity; an out-of-memory error halves the request
        until it fits or drops below the chunk floor.

        Returns:
            Total bytes claimed into the allocator cache.
        """
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
        """Parks the spare VRAM in the allocator cache and logs the resulting budget.

        Does nothing when reservation is disabled or the device is not CUDA.
        """
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)

        claimed     = self._claim()
        free, total = torch.cuda.mem_get_info(self.device)

        self.logger.subsection(f"VRAM reservation: parked {claimed / 1024 ** 3:.2f} GB in the allocator cache, {free / 1024 ** 3:.2f} GB of {total / 1024 ** 3:.2f} GB left free (keep-free target {self.keep_free_bytes / 1024 ** 3:.2f} GB); the blocks stay in the allocator cache until the process exits or something calls torch.cuda.empty_cache()")
