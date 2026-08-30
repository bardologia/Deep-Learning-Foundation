"""Context masking of JEPA input patches.

Samples per-batch pixel masks and writes the configured fill value into the
hidden pixels, so the backbone predicts profile embeddings from partial context
and the embedding loss can be restricted to the hidden region.
"""

from __future__ import annotations

import torch


class MaskSampler:
    """Samples boolean pixel masks according to the masking configuration.

    Attributes:
        cfg : Masking configuration holding strategy, ratio and block count.
    """

    STRATEGIES = ("block", "random")

    def __init__(self, cfg) -> None:
        """Validates and stores the masking configuration.

        Args:
            cfg : JepaMaskingConfig of the run.

        Raises:
            ValueError : If the strategy is unknown, the mask ratio is outside (0, 1), or the block count is below one.
        """
        if cfg.strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown masking strategy '{cfg.strategy}'. Available: {', '.join(self.STRATEGIES)}")

        if not 0.0 < cfg.mask_ratio < 1.0:
            raise ValueError(f"masking.mask_ratio must lie in (0, 1); got {cfg.mask_ratio}")

        if cfg.strategy == "block" and cfg.n_blocks < 1:
            raise ValueError(f"masking.n_blocks must be at least 1; got {cfg.n_blocks}")

        self.cfg = cfg

    def _random_mask(self, batch: int, height: int, width: int, device) -> torch.Tensor:
        """Returns independent Bernoulli pixel masks of shape (batch, height, width)."""
        return torch.rand(batch, height, width, device=device) < self.cfg.mask_ratio

    def _block_mask(self, batch: int, height: int, width: int, device) -> torch.Tensor:
        """Returns rectangle-union masks of shape (batch, height, width).

        Each sample hides ``n_blocks`` rectangles whose combined target area is
        ``mask_ratio`` of the patch, with per-block aspect ratios drawn from
        [0.5, 2]; overlapping rectangles make the realised ratio a lower bound.
        Every block parameter is drawn as one batched tensor on ``device`` and
        the rectangles are rasterised by broadcast comparison, so no host-side
        loop runs on the forward path.

        Args:
            batch  : Number of samples.
            height : Patch height in pixels.
            width  : Patch width in pixels.
            device : Device the mask lives on.

        Returns:
            Boolean mask tensor, True on hidden pixels.
        """
        blocks = self.cfg.n_blocks
        area   = self.cfg.mask_ratio * height * width / blocks

        aspect = torch.empty(batch, blocks, device=device).uniform_(0.5, 2.0)
        bh     = (area * aspect).sqrt().round().clamp(1, height)
        bw     = (area / aspect).sqrt().round().clamp(1, width)
        top    = (torch.rand(batch, blocks, device=device) * (height - bh + 1)).floor()
        left   = (torch.rand(batch, blocks, device=device) * (width  - bw + 1)).floor()

        rows = torch.arange(height, device=device, dtype=aspect.dtype).view(1, 1, height, 1)
        cols = torch.arange(width,  device=device, dtype=aspect.dtype).view(1, 1, 1, width)

        inside_rows = (rows >= top.view(batch, blocks, 1, 1))  & (rows < (top + bh).view(batch, blocks, 1, 1))
        inside_cols = (cols >= left.view(batch, blocks, 1, 1)) & (cols < (left + bw).view(batch, blocks, 1, 1))

        return (inside_rows & inside_cols).any(dim=1)

    def sample(self, batch: int, height: int, width: int, device) -> torch.Tensor:
        """Samples one pixel mask per batch element.

        At least one pixel per sample is guaranteed hidden, so a loss
        restricted to the masked region never sees an empty selection.

        Args:
            batch  : Number of samples.
            height : Patch height in pixels.
            width  : Patch width in pixels.
            device : Device the mask lives on.

        Returns:
            Boolean tensor of shape (batch, height, width), True on hidden pixels.
        """
        if self.cfg.strategy == "block":
            mask = self._block_mask(batch, height, width, device)
        else:
            mask = self._random_mask(batch, height, width, device)

        empty = ~mask.flatten(1).any(dim=1)
        if empty.any():
            mask[empty, 0, 0] = True

        return mask


class InputMasker:
    """Writes the configured fill value into the hidden pixels of a patch batch.

    Attributes:
        cfg : Masking configuration holding the fill mode and noise scale.
    """

    FILLS = ("zeros", "noise")

    def __init__(self, cfg) -> None:
        """Validates and stores the masking configuration.

        Args:
            cfg : JepaMaskingConfig of the run.

        Raises:
            ValueError : If the fill mode is unknown.
        """
        if cfg.fill not in self.FILLS:
            raise ValueError(f"Unknown masking fill '{cfg.fill}'. Available: {', '.join(self.FILLS)}")

        self.cfg = cfg

    def apply(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns the batch with the hidden pixels replaced by the fill value.

        Args:
            images : Input patch batch of shape (batch, channels, height, width).
            mask   : Boolean mask of shape (batch, height, width), True on hidden pixels.

        Returns:
            Masked copy of the batch with the same shape.
        """
        fill = torch.zeros_like(images) if self.cfg.fill == "zeros" else torch.randn_like(images) * self.cfg.fill_noise_std
        return torch.where(mask.unsqueeze(1), fill, images)
