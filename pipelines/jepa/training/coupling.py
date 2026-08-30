"""Coupling policies binding a pretrained autoencoder into JEPA training.

Provides the frozen/fine-tune mode applied to an imported autoencoder and the
stop-gradient or live provider that produces the embedding target.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CouplingMode:
    """Frozen or fine-tuned coupling of a pretrained autoencoder.

    Attributes:
        kind : Either "frozen" or "finetune".
        name : Human-readable label of the coupled autoencoder, used in error messages and as the optimizer parameter-group name.
    """

    VALID = ("frozen", "finetune")

    def __init__(self, kind: str, name: str) -> None:
        """Validates the requested coupling mode.

        Args:
            kind : Coupling mode, one of "frozen" or "finetune".
            name : Label of the coupled autoencoder ("profile autoencoder", "image autoencoder").

        Raises:
            ValueError : If kind is not one of the supported modes.
        """
        if kind not in self.VALID:
            raise ValueError(f"Unknown {name} mode '{kind}'. Available: {list(self.VALID)}. The autoencoder must be imported from a pretrained {name} run and either frozen or fine-tuned; joint training from scratch is not supported.")

        self.kind = kind
        self.name = name

    @property
    def trainable(self) -> bool:
        """Whether the coupled autoencoder receives gradients."""
        return self.kind == "finetune"

    def apply(self, autoencoder: nn.Module) -> None:
        """Sets requires_grad on the autoencoder and pins it to eval when frozen.

        Args:
            autoencoder : Imported autoencoder module to couple.
        """
        autoencoder.requires_grad_(self.trainable)
        if not self.trainable:
            autoencoder.eval()

    def param_groups(self, autoencoder: nn.Module, lr: float, wd: float) -> list[dict]:
        """Builds the optimizer parameter group for a fine-tuned autoencoder.

        Args:
            autoencoder : Coupled autoencoder module.
            lr          : Learning rate for the autoencoder group.
            wd          : Weight decay for the autoencoder group.

        Returns:
            A single-element list with the parameter group, or an empty list when
            the autoencoder is frozen or exposes no trainable parameters.
        """
        if not self.trainable:
            return []

        params = [p for p in autoencoder.parameters() if p.requires_grad]
        if not params:
            return []
        return [{"params": params, "lr": lr, "weight_decay": wd, "name": self.name}]


class TargetProvider:
    """Producer of the embedding target branch of the JEPA loss.

    Attributes:
        kind : Either "stopgrad" (detached target) or "live" (differentiable target).
    """

    VALID = ("stopgrad", "live")

    def __init__(self, kind: str) -> None:
        """Validates the requested target-provider kind.

        Args:
            kind : One of "stopgrad" or "live".

        Raises:
            ValueError : If kind is not one of the supported providers.
        """
        if kind not in self.VALID:
            raise ValueError(f"Unknown target_provider '{kind}'. Available: {list(self.VALID)}. 'stopgrad' detaches the encoder target; 'live' keeps it differentiable so the encoder trains through the embedding loss. The former 'ema' provider was removed: the encoder receives no gradient under a detached target, so its moving average never left the pretrained weights and behaved exactly like 'stopgrad'.")

        self.kind = kind

    def target(self, encoder: nn.Module, curve: torch.Tensor) -> torch.Tensor:
        """Encodes the ground-truth curve into the target embedding.

        Args:
            encoder : Profile-autoencoder encoder producing the embedding.
            curve   : Normalized ground-truth z curves of shape (batch, z, rows, cols).

        Returns:
            Target embedding tensor of shape (batch, embedding_dim, rows, cols),
            detached under "stopgrad" and differentiable under "live".
        """
        if self.kind == "live":
            return encoder(curve)

        with torch.no_grad():
            return encoder(curve)
