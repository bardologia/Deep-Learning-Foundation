
from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.dataset                  import AugmentationConfig
from configuration.training                 import EmbeddingLossConfig, LossConfig, LossCurriculumConfig
from configuration.training.jepa            import JepaMaskingConfig
from pipelines.backbone.training.loss_terms import LOSS_TERMS
from pipelines.shared.model.model_builder   import ModelBuilder


@dataclass
class JepaNamingSpec:

    profile_ae      : str | None
    profile_mode    : str
    target_provider : str
    image_ae        : str | None
    image_mode      : str
    embedding_loss  : EmbeddingLossConfig
    param_loss      : LossConfig
    masking         : JepaMaskingConfig

    @classmethod
    def resolve(cls, config) -> JepaNamingSpec:
        from pipelines.shared.config.config_persistence import ImageAutoencoderConfigIO, ProfileAutoencoderConfigIO

        if not config.profile_autoencoder_run and not config.image_autoencoder_run:
            raise ValueError("JEPA run naming requires at least one of profile_autoencoder_run or image_autoencoder_run; with neither, the run is a plain backbone run and is named by RunNaming.tag.")

        profile_ae = None
        if config.profile_autoencoder_run:
            _, profile_ae = ProfileAutoencoderConfigIO.load(Path(config.profile_autoencoder_logdir) / config.profile_autoencoder_run / "meta")

        image_ae = None
        if config.image_autoencoder_run:
            _, image_ae = ImageAutoencoderConfigIO.load(Path(config.image_autoencoder_logdir) / config.image_autoencoder_run / "meta")

        return cls(
            profile_ae      = profile_ae,
            profile_mode    = config.profile_autoencoder_mode,
            target_provider = config.target_provider,
            image_ae        = image_ae,
            image_mode      = config.image_autoencoder_mode,
            embedding_loss  = config.embedding_loss,
            param_loss      = config.param_loss,
            masking         = config.masking,
        )


class RunNaming:

    NAMING_ORDER = tuple(reversed(LOSS_TERMS))

    AUGMENTATION_FLAGS = (("h", "p_flip_h"), ("v", "p_flip_v"), ("r", "p_rot90"), ("n", "p_noise"))

    EMBEDDING_TERMS = (("emb_mse", "use_embedding_mse", "weight_embedding_mse"), ("emb_cos", "use_embedding_cosine", "weight_embedding_cosine"), ("emb_sl1", "use_embedding_smoothl1", "weight_embedding_smoothl1"), ("emb_hub", "use_embedding_huber", "weight_embedding_huber"), ("emb_cha", "use_embedding_charbonnier", "weight_embedding_charbonnier"), ("emb_var", "use_embedding_variance", "weight_embedding_variance"), ("emb_cov", "use_embedding_covariance", "weight_embedding_covariance"))

    @classmethod
    def loss_tag(cls, loss: LossConfig) -> str:
        parts = [f"{term.name}_{getattr(loss, term.weight_key):g}" for term in cls.NAMING_ORDER if getattr(loss, term.use_flag)]
        if not parts:
            raise ValueError("Cannot name a run from a loss config with no enabled loss terms")

        return "-".join(parts)

    @classmethod
    def embedding_loss_tag(cls, embedding: EmbeddingLossConfig) -> str:
        parts = [f"{name}_{getattr(embedding, weight_key):g}" for name, use_flag, weight_key in cls.EMBEDDING_TERMS if getattr(embedding, use_flag)]
        if embedding.use_curve_recon:
            parts.append(f"curve_{embedding.curve_kind}_{embedding.weight_curve_recon:g}")

        if embedding.use_curve_sobolev:
            parts.append(f"curve_sobolev_{embedding.weight_curve_sobolev:g}")

        if not parts:
            raise ValueError("Cannot name a JEPA run from an embedding loss config with no enabled loss terms")

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
    def masking_tag(masking) -> str:
        return f"mask_{masking.strategy}{masking.mask_ratio:g}"

    @staticmethod
    def profile_ae_tag(ae_model_name: str, mode: str) -> str:
        return f"pae_{ae_model_name.removesuffix('_ae')}.{mode}"

    @staticmethod
    def image_ae_tag(ae_model_name: str, mode: str) -> str:
        return f"iae_{ae_model_name.removesuffix('_ae')}.{mode}"

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
    def jepa_tag(cls, model_name: str, head: str, naming: JepaNamingSpec, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        if naming.profile_ae is None and naming.image_ae is None:
            raise ValueError("Cannot build a JEPA tag without a coupled autoencoder; with neither a profile nor an image autoencoder the run is a plain backbone run and is named by RunNaming.tag.")

        image_extras = (cls.image_ae_tag(naming.image_ae, naming.image_mode),) if naming.image_ae else ()

        if naming.profile_ae is None:
            return cls.tag(model_name, head, naming.param_loss, n_gaussians, augmentation, extras=image_extras)

        mask_extras = (cls.masking_tag(naming.masking),) if naming.masking.enabled else ()
        extras      = (cls.profile_ae_tag(naming.profile_ae, naming.profile_mode), *image_extras, *mask_extras)
        stem        = "-".join((model_name, head, naming.target_provider, cls.gaussians_tag(n_gaussians), cls.augmentation_tag(augmentation), "none", *extras))

        return f"{stem}-{cls.embedding_loss_tag(naming.embedding_loss)}"

    @classmethod
    def benchmark_unit(cls, model_key: str, component: str | None, loss: LossConfig, n_gaussians: int, augmentation: AugmentationConfig) -> str:
        name, head = ModelBuilder.split_key(model_key)
        if component is None:
            return cls.tag(name, head, loss, n_gaussians, augmentation)

        return f"{cls.stem(name, head, loss, n_gaussians, augmentation)}__{component}"

    @staticmethod
    def compose(tag: str, suffix: str | None) -> str:
        return tag if not suffix else f"{tag}_{suffix}"
