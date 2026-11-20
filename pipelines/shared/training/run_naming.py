from __future__ import annotations

from configuration.dataset  import AugmentationConfig
from configuration.training import LossConfig


class RunNaming:

    AUGMENTATION_FLAGS = (("h", "p_flip_h"), ("v", "p_flip_v"), ("r", "p_rot90"), ("n", "p_noise"))

    @staticmethod
    def loss_tag(loss: LossConfig) -> str:
        return loss.kind.value

    @classmethod
    def augmentation_tag(cls, augmentation: AugmentationConfig) -> str:
        letters = "".join(letter for letter, probability in cls.AUGMENTATION_FLAGS if getattr(augmentation, probability) > 0.0)
        return letters or "noaug"

    @classmethod
    def tag(cls, model_name: str, loss: LossConfig, augmentation: AugmentationConfig) -> str:
        return "-".join((model_name, cls.loss_tag(loss), cls.augmentation_tag(augmentation)))

    @staticmethod
    def compose(tag: str, suffix: str | None) -> str:
        return tag if not suffix else f"{tag}_{suffix}"
