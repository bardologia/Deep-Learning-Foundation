"""Reassembly of trained JEPA runs for inference, in curve space and param space.

A JEPA run predicts a profile-autoencoder embedding from the channel stack, so
inference has to rebuild the backbone, the frozen profile autoencoder and, when
the run used one, the image-autoencoder frontend from their persisted
configurations. The curve loader wraps the chain into a module that decodes and
denormalizes the profile; the param loader restores runs whose backbone predicts
Gaussian parameters directly behind an image-autoencoder frontend.
"""

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
    """Chains the JEPA predictor, the profile decoder and the profile denormalization.

    Attributes:
        jepa               : Trained JEPA module holding the backbone and the profile autoencoder.
        profile_normalizer : Normalizer carrying the training profile statistics.
    """

    def __init__(self, jepa_module: JepaModule, profile_normalizer: ProfileNormalizer) -> None:
        """Stores the trained JEPA module and the profile normalizer.

        Args:
            jepa_module        : Trained JEPA module.
            profile_normalizer : Normalizer of the training profile statistics.
        """
        super().__init__()
        self.jepa               = jepa_module
        self.profile_normalizer = profile_normalizer

    def forward_with_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts the profile of every pixel and exposes the intermediate latents.

        Args:
            x : Input channel stack of shape (batch, channels, rows, cols).

        Returns:
            The denormalized profile of shape (batch, height_bins, row,
            range), the predicted normalized embedding of shape
            (batch, embedding_dim, rows, cols), and the decoded profile still
            in normalized units.
        """
        autoencoder = self.jepa.profile_autoencoder

        z_hat_n = autoencoder.normalize_embedding(self.jepa(x))
        curve_n = autoencoder.decode(z_hat_n)

        return self.profile_normalizer.denormalize(curve_n), z_hat_n, curve_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the denormalized profile of shape (batch, height_bins, rows, cols)."""
        return self.forward_with_embedding(x)[0]


class JepaRunLoader(RunLoader):
    """Restores a JEPA run as a curve predictor over the channel stack."""

    def _validate_out_channels(self, out_channels: int, n_gaussians: int) -> None:
        """Checks that the persisted backbone width equals the profile embedding width.

        Raises:
            ValueError : If the run summary's output width differs from the profile autoencoder's embedding dimension.
        """
        ae_cfg, _ = ProfileAutoencoderConfigIO.load(self.meta_directory)

        if out_channels != ae_cfg.embedding_dim:
            raise ValueError(f"JEPA run '{self.run_directory}' persists out_channels={out_channels} but its profile autoencoder embeds into {ae_cfg.embedding_dim} channels; the backbone width in the run summary does not describe this run.")

    def _image_frontend(self, dataset_in_channels: int):
        """Restores the optional image-autoencoder frontend of the run.

        Args:
            dataset_in_channels : Input width the dataset yields.

        Returns:
            The image autoencoder, the width the backbone then sees and a
            description for the reassembly table; None, the dataset width and
            a line naming the absent config file when the run has no frontend.
        """
        if not ImageAutoencoderConfigIO.exists(self.meta_directory):
            return None, dataset_in_channels, f"none (no {ImageAutoencoderConfigIO.FILENAME} in meta; the backbone consumes the {dataset_in_channels} dataset channels)"

        image_cfg, image_name = ImageAutoencoderConfigIO.load(self.meta_directory)
        image_autoencoder, _  = get_image_autoencoder(image_name, image_cfg)
        return image_autoencoder, image_cfg.embedding_dim, f"{image_name} (embedding_dim={image_cfg.embedding_dim}; the backbone consumes its feature map)"

    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size):
        """Rebuilds the JEPA module: optional image frontend, backbone and profile autoencoder.

        Args:
            backbone_name : Registered backbone architecture name.
            in_channels   : Input width the dataset yields.
            out_channels  : Output width recorded in the run summary.
            patch_size    : Patch extent as (rows, cols) in pixels.

        Returns:
            The assembled JEPA module, its backbone widened to the profile
            embedding dimension; the reassembly is logged as a table.
        """
        ae_cfg, ae_name = ProfileAutoencoderConfigIO.load(self.meta_directory)
        model_config, _ = BackboneModelConfigIO.load(self.meta_directory)
        self.model_head = model_config.head

        image_autoencoder, backbone_in, frontend = self._image_frontend(in_channels)

        overrides = {"in_channels": backbone_in, "out_channels": ae_cfg.embedding_dim}
        overrides.update(ModelBuilder.image_size_override(backbone_name, patch_size))
        backbone, _ = get_backbone(backbone_name, config=model_config, **overrides)

        profile_autoencoder, _  = get_profile_autoencoder(ae_name, ae_cfg)
        self.profile_normalizer = ProfileNormalizer(ProfileStats.load(self.meta_directory, logger=self.logger))

        self.logger.section("[JEPA Run Reassembled]")
        self.logger.kv_table({
            "Image frontend" : frontend,
            "Profile AE"     : f"{ae_name} (embedding_dim={ae_cfg.embedding_dim}, profile_length={ae_cfg.profile_length}; decodes the predicted embedding into curves)",
            "Backbone"       : f"{backbone_name} (head {model_config.head}, {backbone_in} -> {ae_cfg.embedding_dim} channels)",
        })

        return JepaModule(backbone, profile_autoencoder=profile_autoencoder, image_autoencoder=image_autoencoder)

    def _wrap_model(self, model, device: str, norm_stats, z_axis, amp_max: float) -> ModelWrapper:
        """Wraps the JEPA module into the curve-space inference adapter.

        The adapter already denormalizes its own profiles, so the wrapper is
        given no normalizer, z axis or amplitude ceiling.

        Args:
            model      : Assembled JEPA module.
            device     : Torch device string the adapter is moved to.
            norm_stats : Input normalization statistics, unused by this family.
            z_axis     : Profile z axis in metres, unused by this family.
            amp_max    : Amplitude ceiling, unused by this family.

        Returns:
            The model wrapper used by the predictor.
        """
        adapter = JepaInferenceModel(model, self.profile_normalizer).to(device)
        adapter.eval()
        return ModelWrapper(model=adapter, device=device, params_per_gaussian=3, normalizer=None, z_axis=None, amp_max=None)


class JepaParamRunLoader(RunLoader):
    """Restores a JEPA-frontend run whose backbone predicts Gaussian parameters directly."""

    def _build_model(self, backbone_name: str, in_channels: int, out_channels: int, patch_size):
        """Rebuilds the image-autoencoder frontend and the parameter-predicting backbone.

        Args:
            backbone_name : Registered backbone architecture name.
            in_channels   : Input width the dataset yields, consumed by the frontend.
            out_channels  : Width of the Gaussian-parameter output.
            patch_size    : Patch extent as (rows, cols) in pixels.

        Returns:
            The JEPA module carrying the frontend and the backbone, with no
            profile autoencoder; the reassembly is logged as a table.
        """
        model_config, _       = BackboneModelConfigIO.load(self.meta_directory)
        self.model_head       = model_config.head
        image_cfg, image_name = ImageAutoencoderConfigIO.load(self.meta_directory)
        image_autoencoder, _  = get_image_autoencoder(image_name, image_cfg)

        overrides = {"in_channels": image_cfg.embedding_dim, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(backbone_name, patch_size))
        backbone, _ = get_backbone(backbone_name, config=model_config, **overrides)

        self.logger.section("[JEPA Run Reassembled]")
        self.logger.kv_table({
            "Image frontend" : f"{image_name} (embedding_dim={image_cfg.embedding_dim}; the backbone consumes its feature map)",
            "Profile AE"     : "none (the backbone predicts Gaussian parameters directly)",
            "Backbone"       : f"{backbone_name} (head {model_config.head}, {image_cfg.embedding_dim} -> {out_channels} channels)",
        })

        return JepaModule(backbone, profile_autoencoder=None, image_autoencoder=image_autoencoder)
