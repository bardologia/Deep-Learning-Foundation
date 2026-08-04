from __future__ import annotations

from configuration.dataset                  import AugmentationConfig
from configuration.training                 import LossConfig, LossCurriculumConfig
from pipelines.backbone.training.loss_terms import LOSS_TERMS
from pipelines.shared.model.model_builder   import ModelBuilder


class RunNaming:

    NAMING_ORDER = tuple(reversed(LOSS_TERMS))

    AUGMENTATION_FLAGS = (("h", "p_flip_h"), ("v", "p_flip_v"), ("r", "p_rot90"), ("n", "p_noise"))

    @classmethod
    def loss_tag(cls, loss: LossConfig) -> str:
        parts = [f"{term.name}_{getattr(loss, term.weight_key):g}" for term in cls.NAMING_ORDER if getattr(loss, term.use_flag)]
        if not parts:
            raise ValueError("Cannot name a run from a loss config with no enabled loss terms")

        return "-".join(parts)

    @staticmethod
    def matching_tag(loss: LossConfig) -> str:
        return loss.param_matching.value

    @staticmethod
    def gaussians_tag(n_gaussians: int) -> str:
        return f"K_{n_gaussians}"

    @classmethod
    def augmentation_tag(cls, augmentation: AugmentationConfig) -> str:
        letters = "".join(letter for letter, probability in cls.AUGMENTATION_FLAGS if getattr(augmentation, probability) > 0.0)
        return letters or "noaug"

    @staticmethod
    def presence_tag(loss: LossConfig) -> str:
        letters = ("A" if loss.use_active_normalization else "") + ("B" if loss.presence_balance else "")
        return letters or "none"

    @staticmethod
    def dual_model_tag(params_backbone: str, existence_backbone: str) -> str:
        trunks = params_backbone if params_backbone == existence_backbone else f"{params_backbone}.{existence_backbone}"
        return f"dual_{trunks}"

    @classmethod
    def stem(cls, model_name: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig, extras: tuple = ()) -> str:
        return "-".join((model_name, head, cls.matching_tag(loss), cls.gaussians_tag(n_gaussians), cls.augmentation_tag(augmentation), cls.presence_tag(loss), *extras))

    @classmethod
    def tag(cls, model_name: str, head: str, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig, extras: tuple = ()) -> str:
        return f"{cls.stem(model_name, head, loss, n_gaussians, augmentation, extras)}-{cls.loss_tag(loss)}"

    @classmethod
    def training_tag(cls, model_name: str, head: str, curriculum: LossCurriculumConfig, n_gaussians: int, augmentation: AugmentationConfig, extras: tuple = ()) -> str:
        return cls.tag(model_name, head, curriculum.active_stages()[-1], n_gaussians, augmentation, extras)

    @classmethod
    def benchmark_unit(cls, model_key: str, component: str | None, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        name, head = ModelBuilder.split_key(model_key)
        if component is None:
            return cls.tag(name, head, loss, n_gaussians, augmentation)

        return f"{cls.stem(name, head, loss, n_gaussians, augmentation)}__{component}"

    @staticmethod
    def compose(tag: str, suffix: str | None) -> str:
        return tag if not suffix else f"{tag}_{suffix}"
