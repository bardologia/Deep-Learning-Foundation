"""Adapter exposing a trainer's loss computation to the pre-training probes."""

from __future__ import annotations


class TrainerFeed:
    """Feeds probe batches through a trainer's own loss path.

    Attributes:
        trainer : Trainer whose loss computation is reused by the probes.
    """

    def __init__(self, trainer) -> None:
        """Stores the trainer the probes borrow their loss path from."""
        self.trainer = trainer

    @staticmethod
    def to_model_input(batch, device):
        """Returns the batch unchanged, since the trainer moves tensors to the device itself."""
        return batch

    def forward_loss(self, model, batch):
        """Returns the trainer's total loss for one batch, as a 0-d tensor.

        Args:
            model : Model argument required by the probe interface and unused here, because the trainer holds its own model.
            batch : Batch drawn from the probe data loader.

        Returns:
            Total loss of the batch as a 0-d tensor carrying the graph.
        """
        return self.trainer._compute_loss(batch)["total_loss"]
