from __future__ import annotations

import torch
import torch.nn as nn


class CouplingMode:
    VALID = ("frozen", "finetune")

    def __init__(self, kind: str, name: str) -> None:
        if kind not in self.VALID:
            raise ValueError(f"Unknown {name} mode '{kind}'. Available: {list(self.VALID)}. The autoencoder must be imported from a pretrained {name} run and either frozen or fine-tuned; joint training from scratch is not supported.")

        self.kind = kind
        self.name = name

    @property
    def trainable(self) -> bool:
        return self.kind == "finetune"

    def apply(self, autoencoder: nn.Module) -> None:
        autoencoder.requires_grad_(self.trainable)
        if not self.trainable:
            autoencoder.eval()

    def param_groups(self, autoencoder: nn.Module, lr: float, wd: float) -> list[dict]:
        if not self.trainable:
            return []

        params = [p for p in autoencoder.parameters() if p.requires_grad]
        if not params:
            return []
        return [{"params": params, "lr": lr, "weight_decay": wd, "name": self.name}]


class TargetProvider:
    VALID = ("stopgrad", "live")

    def __init__(self, kind: str) -> None:
        if kind not in self.VALID:
            raise ValueError(f"Unknown target_provider '{kind}'. Available: {list(self.VALID)}. 'stopgrad' detaches the encoder target; 'live' keeps it differentiable so the encoder trains through the embedding loss. The former 'ema' provider was removed: the encoder receives no gradient under a detached target, so its moving average never left the pretrained weights and behaved exactly like 'stopgrad'.")

        self.kind = kind

    def target(self, encoder: nn.Module, curve: torch.Tensor) -> torch.Tensor:
        if self.kind == "live":
            return encoder(curve)

        with torch.no_grad():
            return encoder(curve)
