from __future__ import annotations

import torch
import torch.nn as nn

from models                                              import get_backbone
from models.profile_autoencoder                          import get_profile_autoencoder
from models.image_autoencoder                            import get_image_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.backbone.inference.loader                 import RunLoader
from pipelines.backbone.inference.model_wrapper          import ModelWrapper
from pipelines.jepa.training.trainer                     import JepaModule
from pipelines.shared.config.config_persistence          import ProfileAutoencoderConfigIO, ImageAutoencoderConfigIO, BackboneModelConfigIO
from pipelines.shared.model.model_builder                import ModelBuilder



class JepaInferenceModel(nn.Module):
    def __init__(self, jepa_module: JepaModule, profile_normalizer: ProfileNormalizer) -> None:
        super().__init__()
        self.jepa               = jepa_module
        self.profile_normalizer = profile_normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_hat   = self.jepa(x)
        curve_n = self.jepa.profile_autoencoder.decode(self.jepa.profile_autoencoder.normalize_embedding(z_hat))
        return self.profile_normalizer.denormalize(curve_n)


class JepaRunLoader(RunLoader):
    def _validate_out_channels(self, out_channels: int, n_gaussians: int) -> None:
        ae_cfg, _ = ProfileAutoencoderConfigIO.load(self.meta_directory)

        if out_channels != ae_cfg.embedding_dim:
            raise ValueError(f"JEPA run '{self.run_directory}' persists out_channels={out_channels} but its profile autoencoder embeds into {ae_cfg.embedding_dim} channels; the backbone width in the run summary does not describe this run.")

    def _image_frontend(self, dataset_in_channels: int):
        if not ImageAutoencoderConfigIO.exists(self.meta_directory):
            return None, dataset_in_channels

        image_cfg, image_name = ImageAutoencoderConfigIO.load(self.meta_directory)
        image_autoencoder, _  = get_image_autoencoder(image_name, image_cfg)
        return image_autoencoder, image_cfg.embedding_dim

    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size):
        ae_cfg, ae_name = ProfileAutoencoderConfigIO.load(self.meta_directory)
        model_config, _ = BackboneModelConfigIO.load(self.meta_directory)
        self.model_head = model_config.head

        image_autoencoder, backbone_in = self._image_frontend(in_channels)

        overrides = {"in_channels": backbone_in, "out_channels": ae_cfg.embedding_dim}
        overrides.update(ModelBuilder.image_size_override(backbone_name, patch_size))
        backbone, _ = get_backbone(backbone_name, config=model_config, **overrides)

        profile_autoencoder, _  = get_profile_autoencoder(ae_name, ae_cfg)
        self.profile_normalizer = ProfileNormalizer(ProfileStats.load(self.meta_directory))

        return JepaModule(backbone, profile_autoencoder=profile_autoencoder, image_autoencoder=image_autoencoder)

    def _wrap_model(self, model, device: str, norm_stats, x_axis, amp_max: float) -> ModelWrapper:
        adapter = JepaInferenceModel(model, self.profile_normalizer).to(device)
        adapter.eval()
        return ModelWrapper(model=adapter, device=device, params_per_gaussian=3, normalizer=None, x_axis=None, amp_max=None)


class JepaParamRunLoader(RunLoader):
    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size):
        model_config, _       = BackboneModelConfigIO.load(self.meta_directory)
        self.model_head       = model_config.head
        image_cfg, image_name = ImageAutoencoderConfigIO.load(self.meta_directory)
        image_autoencoder, _  = get_image_autoencoder(image_name, image_cfg)

        overrides = {"in_channels": image_cfg.embedding_dim, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(backbone_name, patch_size))
        backbone, _ = get_backbone(backbone_name, config=model_config, **overrides)

        return JepaModule(backbone, profile_autoencoder=None, image_autoencoder=image_autoencoder)
