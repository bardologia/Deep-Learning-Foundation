"""Training pipeline and single-run entry runner for JEPA predictor runs.

Imports one or both pretrained autoencoders from their run directories, checks
that their input representation and z axis match the JEPA dataset, builds
the coupled module, and trains it through the shared training-run machinery.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib     import Path

import numpy as np
import torch

from configuration.training                              import JepaTrainerConfig
from models                                              import get_backbone
from models.profile_autoencoder                          import get_profile_autoencoder
from models.image_autoencoder                            import get_image_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.shared.config.config_factory              import ConfigFactory
from pipelines.shared.config.run_metadata                import TrainingRunMetadata
from pipelines.backbone.dataset.pipeline                 import BackboneDatasetPreparation
from pipelines.shared.model.model_builder                import ModelBuilder
from pipelines.shared.training.overfit_check             import OverfitCheck
from pipelines.shared.training.run_naming                import JepaNamingSpec, RunNaming
from pipelines.shared.training.training_runner           import EntryConfigTrainRunner
from pipelines.jepa.training.trainer                     import JepaModule, Trainer
from pipelines.shared.config.config_persistence          import ProfileAutoencoderConfigIO, ImageAutoencoderConfigIO
from tools.runtime.reproducibility                       import Reproducibility
from tools.runtime.run_tag                               import RunTag


class TrainingPipeline:
    """Builds and trains a JEPA module from an entry configuration.

    Attributes:
        entry                    : JEPA entry configuration driving the run.
        factory                  : ConfigFactory deriving trainer and dataset configurations.
        profile_autoencoder_meta : Metadata directory of the imported profile autoencoder run, or None.
        autoencoder_cfg          : Loaded profile autoencoder model configuration, or None.
        ae_model_name            : Registry name of the profile autoencoder, or None.
        image_autoencoder_meta   : Metadata directory of the imported image autoencoder run, or None.
        image_ae_cfg             : Loaded image autoencoder model configuration, or None.
        image_ae_model_name      : Registry name of the image autoencoder, or None.
        trainer_config           : Assembled JepaTrainerConfig for this run.
        dataset_config           : Dataset configuration, with patch geometry squared where the backbone family requires it.
        backbone_name            : Registry name of the backbone.
        profile_ae_state         : Cached profile autoencoder state dict, loaded on first use.
        image_ae_state           : Cached image autoencoder state dict, loaded on first use.
    """

    def __init__(self, entry_config, split_regions=None) -> None:
        """Resolves the imported autoencoder runs and assembles the trainer configuration.

        Args:
            entry_config  : JEPA entry configuration.
            split_regions : Optional explicit train/val/test regions overriding the dataset configuration, used by cross-validation folds.

        Raises:
            ValueError        : If neither a profile nor an image autoencoder run is given, or if the backbone head is incompatible with a coupled profile autoencoder.
            FileNotFoundError : If a named autoencoder run directory does not exist.
        """
        self.entry   = entry_config
        self.factory = ConfigFactory(entry_config)
        Reproducibility.seed_everything(entry_config.seed)

        base = self.factory.training_trainer_config(logdir=entry_config.logdir)

        profile_dir = self._resolve_ae_run(entry_config.profile_autoencoder_logdir, entry_config.profile_autoencoder_run, "profile")
        image_dir   = self._resolve_ae_run(entry_config.image_autoencoder_logdir,   entry_config.image_autoencoder_run,   "image")

        if profile_dir is None and image_dir is None:
            raise ValueError("JEPA requires at least one of profile_autoencoder_run or image_autoencoder_run; with neither, train the plain backbone via main/training/train_backbone.py.")

        self._validate_head(entry_config.backbone_head, profile_coupled=profile_dir is not None)

        self.profile_autoencoder_meta = None
        self.autoencoder_cfg          = None
        self.ae_model_name            = None
        profile_checkpoint            = None
        if profile_dir is not None:
            self.profile_autoencoder_meta            = profile_dir / "meta"
            self.autoencoder_cfg, self.ae_model_name = ProfileAutoencoderConfigIO.load(self.profile_autoencoder_meta)
            profile_checkpoint                       = str(profile_dir / "best_model.pt")

        self.image_autoencoder_meta = None
        self.image_ae_cfg           = None
        self.image_ae_model_name    = None
        image_checkpoint            = None
        if image_dir is not None:
            self.image_autoencoder_meta                 = image_dir / "meta"
            self.image_ae_cfg, self.image_ae_model_name = ImageAutoencoderConfigIO.load(self.image_autoencoder_meta)
            image_checkpoint                            = str(image_dir / "best_model.pt")

        self.trainer_config = JepaTrainerConfig(
            gaussian                       = base.gaussian,
            autoencoder                    = self.autoencoder_cfg,
            embedding_loss                 = entry_config.embedding_loss,
            profile_autoencoder_mode       = entry_config.profile_autoencoder_mode,
            target_provider                = entry_config.target_provider,
            profile_autoencoder_checkpoint = profile_checkpoint,
            ae_finetune_lr                 = entry_config.ae_finetune_lr,
            ae_finetune_wd                 = entry_config.ae_finetune_wd,
            image_autoencoder              = self.image_ae_cfg,
            image_autoencoder_mode         = entry_config.image_autoencoder_mode,
            image_autoencoder_checkpoint   = image_checkpoint,
            image_ae_finetune_lr           = entry_config.image_ae_finetune_lr,
            image_ae_finetune_wd           = entry_config.image_ae_finetune_wd,
            param_loss                     = entry_config.param_loss,
            curriculum                     = entry_config.curriculum,
            masking                        = entry_config.masking,
        )
        self.trainer_config.inherit_shared_from(base)

        self.dataset_config              = self.factory.training_dataset_config()
        self.dataset_config.input_config = entry_config.input
        if split_regions is not None:
            self.dataset_config.split_regions = split_regions

        self.backbone_name    = entry_config.backbone_name
        self.profile_ae_state = None
        self.image_ae_state   = None

        patch                    = self.dataset_config.patch
        patch.size, patch.stride = ModelBuilder.square_patch([self.backbone_name], patch.size, patch.stride)

    @staticmethod
    def _resolve_ae_run(logdir, run, label):
        """Returns the autoencoder run directory, or None when no run is named.

        Args:
            logdir : Root directory holding autoencoder runs.
            run    : Run directory name, empty to disable this coupling.
            label  : Label used in the error message ("profile", "image").

        Returns:
            The resolved run directory, or None.

        Raises:
            FileNotFoundError : If the named run directory does not exist.
        """
        if not run:
            return None
        directory = Path(logdir) / run
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} autoencoder run '{run}' not found under {logdir}")
        return directory

    @staticmethod
    def _validate_head(backbone_head: str, profile_coupled: bool) -> None:
        """Requires the 'conv' head whenever the backbone must emit an embedding.

        Args:
            backbone_head   : Configured backbone head name.
            profile_coupled : Whether a profile autoencoder is coupled.

        Raises:
            ValueError : If a profile autoencoder is coupled to a non-conv head.
        """
        if profile_coupled and backbone_head != "conv":
            raise ValueError(f"backbone_head '{backbone_head}' arranges the backbone output as Gaussian parameter slots, but a coupled profile autoencoder makes the backbone predict its embedding; only the 'conv' head projects to the embedding. Set backbone_head='conv' or drop profile_autoencoder_run.")

    def _prepare_datasets(self, run_directory, logger, seed):
        """Builds the backbone datasets and loaders for this run.

        Args:
            run_directory : Directory receiving dataset metadata.
            logger        : Logger for the preparation banners.
            seed          : Seed driving the split and augmentation randomness.

        Returns:
            Tuple of the (train, val, test) loaders, the dataset dict keyed by split,
            the z axis, and the number of z bins.
        """
        return BackboneDatasetPreparation(self.dataset_config, self.trainer_config, run_directory, logger, seed).run()

    def _gaussian_out_channels(self) -> int:
        """Returns the backbone output channel count for direct Gaussian prediction."""
        g = self.trainer_config.gaussian
        return g.params_per_gaussian * g.n_default_gaussians

    def _run_overfit_check(self, run_meta, logger, datasets, z_len: int, z_axis, profile_normalizer) -> None:
        """Runs the pre-training overfit sanity gate on a sanitized copy of the configs.

        Args:
            run_meta           : Training run metadata holding the run directory.
            logger             : Logger for the gate banners.
            datasets           : Dataset dict keyed by split.
            z_len              : Number of z bins.
            z_axis             : Z axis in metres.
            profile_normalizer : Curve normalizer for the coupled profile autoencoder, or None.
        """
        check = OverfitCheck(self.entry.overfit_check, run_meta.run_directory, logger)
        if not check.enabled:
            return

        check.announce()

        gate_trainer_config = check.sanitized_trainer_config(self.trainer_config)

        if self.autoencoder_cfg is None:
            gate_trainer_config.param_loss.use_active_normalization = False
            check.record("param_loss.use_active_normalization", False)

        base_backbone_config = ModelBuilder.config_from_registry(self.backbone_name, self.entry.model_overrides, head=self.entry.backbone_head)
        gate_backbone_config = check.sanitized_model_config(base_backbone_config)

        gate_module, gate_backbone_cfg = self._build_module(datasets, z_len, z_axis, logger, backbone_config=gate_backbone_config)

        gate_trainer = Trainer(gate_module, gate_backbone_cfg, z_axis, gate_trainer_config, check.work_directory, logger, datasets["train"].normalizer, profile_normalizer)

        check.run(gate_trainer, datasets["train"])

    def _build_module(self, datasets, z_len: int, z_axis, logger, backbone_config=None):
        """Builds the JepaModule and resolves the backbone input and output widths.

        Args:
            datasets        : Dataset dict keyed by split; the train split supplies the input channel count.
            z_len           : Number of z bins produced by the dataset.
            z_axis          : Z axis in metres, checked against the profile autoencoder.
            logger          : Logger receiving the module summary table.
            backbone_config : Optional prebuilt backbone configuration, used by the overfit gate; when None the registry configuration plus entry overrides are used.

        Returns:
            Tuple of the assembled JepaModule and the resolved backbone configuration.

        Raises:
            ValueError : If the image autoencoder's input channels or the profile autoencoder's profile length disagree with the JEPA dataset.
        """
        dataset_in_channels = datasets["train"].input_channels

        image_autoencoder = None
        backbone_in       = dataset_in_channels
        if self.image_ae_cfg is not None:
            if self.image_ae_cfg.in_channels != dataset_in_channels:
                raise ValueError(f"Image autoencoder was trained with in_channels={self.image_ae_cfg.in_channels} but the JEPA dataset produces {dataset_in_channels} input channels; the input representation and secondaries must match the image autoencoder run.")
            image_autoencoder = self._load_image_autoencoder()
            backbone_in       = self.image_ae_cfg.embedding_dim

        profile_autoencoder = None
        if self.autoencoder_cfg is not None:
            if self.autoencoder_cfg.profile_length != z_len:
                raise ValueError(f"Profile autoencoder was trained with profile_length={self.autoencoder_cfg.profile_length} but the JEPA dataset produces curves of length {z_len}; retrain the profile autoencoder on this dataset's z axis before coupling it.")

            backbone_out        = self.autoencoder_cfg.embedding_dim
            profile_autoencoder = self._load_profile_autoencoder(z_axis)
        else:
            backbone_out = self._gaussian_out_channels()

        self.backbone_in_channels  = backbone_in
        self.backbone_out_channels = backbone_out

        backbone, backbone_cfg = self._build_backbone(backbone_in, backbone_out, self.dataset_config.patch.size, config=backbone_config)
        module                 = JepaModule(backbone, profile_autoencoder=profile_autoencoder, image_autoencoder=image_autoencoder)

        self._log_module(logger, module, backbone, dataset_in_channels, backbone_in, backbone_out)
        return module, backbone_cfg

    def _log_module(self, logger, module, backbone, dataset_in: int, backbone_in: int, backbone_out: int) -> None:
        """Prints the built-module table with widths, parameter counts and coupling state.

        Args:
            logger       : Logger receiving the section and table.
            module       : Assembled JepaModule.
            backbone     : Backbone submodule, counted separately.
            dataset_in   : Input channel count produced by the dataset.
            backbone_in  : Input channel count seen by the backbone.
            backbone_out : Output channel count emitted by the backbone.
        """
        backbone_params  = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        info = {
            "Backbone"             : self.backbone_name,
            "Dataset Channels"     : dataset_in,
            "Backbone In"          : backbone_in,
            "Backbone Out"         : backbone_out,
            "Backbone Parameters"  : f"{backbone_params:,}",
            "Trainable Parameters" : f"{trainable_params:,}",
        }

        if self.autoencoder_cfg is not None:
            info["Target Provider"] = self.trainer_config.target_provider
            info["Embedding Loss"]  = self._embedding_loss_label(self.trainer_config.embedding_loss)
            info["Param Loss"]      = "inactive (profile AE coupled; the run name carries the embedding loss tokens)"
        else:
            info["Target Provider"] = "inactive (no profile AE)"
            info["Embedding Loss"]  = "inactive (no profile AE; the backbone trains on param_loss, banner below)"

        if self.image_ae_cfg is not None:
            info["Image AE"]            = f"{self.image_ae_model_name}  (embedding_dim={self.image_ae_cfg.embedding_dim}, mode={self.trainer_config.image_autoencoder_mode})"
            info["Image AE Checkpoint"] = self.trainer_config.image_autoencoder_checkpoint
        else:
            info["Image AE"] = "disabled (backbone consumes dataset channels directly)"

        if self.autoencoder_cfg is not None:
            info["Profile AE"]            = f"{self.ae_model_name}  (embedding_dim={self.autoencoder_cfg.embedding_dim}, mode={self.trainer_config.profile_autoencoder_mode})"
            info["Profile AE Checkpoint"] = self.trainer_config.profile_autoencoder_checkpoint
        else:
            info["Profile AE"] = "disabled (backbone predicts Gaussian parameters directly)"

        logger.section("[JEPA Module Built]")
        logger.kv_table(info)

    @staticmethod
    def _embedding_loss_label(emb) -> str:
        """Formats the enabled embedding-loss terms and weights as one banner string.

        Args:
            emb : Embedding-loss configuration.

        Returns:
            Pipe-separated term description, or "no terms enabled".
        """
        terms = []
        if emb.use_embedding_mse:
            terms.append(f"mse w={emb.weight_embedding_mse:g}")
        if emb.use_embedding_cosine:
            terms.append(f"cosine w={emb.weight_embedding_cosine:g}")
        if emb.use_embedding_smoothl1:
            terms.append(f"smoothl1 w={emb.weight_embedding_smoothl1:g} beta={emb.smoothl1_beta:g}")
        if emb.use_embedding_huber:
            terms.append(f"huber w={emb.weight_embedding_huber:g} delta={emb.embedding_huber_delta:g}")
        if emb.use_embedding_charbonnier:
            terms.append(f"charbonnier w={emb.weight_embedding_charbonnier:g} eps={emb.embedding_charbonnier_eps:g}")
        if emb.use_embedding_variance:
            terms.append(f"variance w={emb.weight_embedding_variance:g} gamma={emb.variance_gamma:g}")
        if emb.use_embedding_covariance:
            terms.append(f"covariance w={emb.weight_embedding_covariance:g}")
        if emb.use_curve_recon:
            terms.append(f"curve_recon({emb.curve_kind}) w={emb.weight_curve_recon:g}")

        return "  |  ".join(terms) if terms else "no terms enabled"

    def _build_backbone(self, in_channels: int, out_channels: int, patch_size, config=None):
        """Instantiates the backbone at the resolved input and output widths.

        Args:
            in_channels  : Backbone input channel count.
            out_channels : Backbone output channel count.
            patch_size   : Patch shape as (rows, cols) in pixels, used for the image-size override of ViT-family backbones.
            config       : Optional prebuilt backbone configuration; when None the head and entry overrides are applied on top of the registry defaults.

        Returns:
            Tuple of the backbone module and its configuration.
        """
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(self.backbone_name, patch_size))
        if config is None:
            overrides["head"] = self.entry.backbone_head
            for k, v in self.entry.model_overrides.items():
                overrides[k] = v
        return get_backbone(self.backbone_name, config=config, **overrides)

    def _load_profile_autoencoder(self, z_axis):
        """Instantiates the profile autoencoder and loads its pretrained weights.

        Args:
            z_axis : Z axis in metres of the JEPA dataset, compared against the axis stored in the checkpoint.

        Returns:
            The profile autoencoder with the checkpoint state loaded.
        """
        autoencoder, _ = get_profile_autoencoder(self.ae_model_name, self.autoencoder_cfg)
        ckpt_path      = self.trainer_config.profile_autoencoder_checkpoint

        if self.profile_ae_state is None:
            self.validate_checkpoint(ckpt_path, "profile")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self._validate_profile_axis(np.asarray(ckpt["z_axis"], dtype=np.float32), np.asarray(z_axis, dtype=np.float32), ckpt_path)

            self.profile_ae_state = ckpt["params"]

        autoencoder.load_state_dict(self.profile_ae_state)
        return autoencoder

    @staticmethod
    def _validate_profile_axis(ae_axis, jepa_axis, ckpt_path) -> None:
        """Requires the checkpoint's z axis to match the JEPA dataset's.

        Args:
            ae_axis   : Z axis in metres stored in the autoencoder checkpoint.
            jepa_axis : Z axis in metres produced by the JEPA dataset.
            ckpt_path : Checkpoint path, quoted in the error message.

        Raises:
            ValueError : If the two axes differ in length or in sampled values.
        """
        if ae_axis.shape != jepa_axis.shape or not np.allclose(ae_axis, jepa_axis, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Profile autoencoder checkpoint '{ckpt_path}' was trained on z axis [{ae_axis[0]:.3f}, {ae_axis[-1]:.3f}] with {ae_axis.size} bins, but the JEPA dataset produces axis [{jepa_axis[0]:.3f}, {jepa_axis[-1]:.3f}] with {jepa_axis.size} bins; retrain the profile autoencoder on this dataset's z axis before coupling it.")

    def _load_image_autoencoder(self):
        """Instantiates the image autoencoder and loads its pretrained weights."""
        autoencoder, _ = get_image_autoencoder(self.image_ae_model_name, self.image_ae_cfg)
        ckpt_path      = self.trainer_config.image_autoencoder_checkpoint

        if self.image_ae_state is None:
            self.validate_checkpoint(ckpt_path, "image")

            self.image_ae_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["params"]

        autoencoder.load_state_dict(self.image_ae_state)
        return autoencoder

    @staticmethod
    def validate_checkpoint(ckpt_path, label: str) -> None:
        """Requires the named autoencoder checkpoint file to exist.

        Args:
            ckpt_path : Path to the expected best_model.pt.
            label     : Label used in the error message ("profile", "image").

        Raises:
            FileNotFoundError : If the checkpoint file is missing.
        """
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"{label} autoencoder checkpoint '{ckpt_path}' does not exist; expected 'best_model.pt' under the selected {label} autoencoder run directory.")

    def _save_metadata(self, run_meta, backbone_cfg, z_len: int) -> None:
        """Writes the trainer, backbone, autoencoder and run-summary metadata files.

        Args:
            run_meta     : Training run metadata owning the metadata directory.
            backbone_cfg : Resolved backbone configuration.
            z_len        : Number of z bins, recorded in the run summary.
        """
        gaussian_cfg = self.trainer_config.gaussian

        run_meta.save_trainer_config()
        run_meta.save_model_config(backbone_cfg, self.backbone_name)

        if self.autoencoder_cfg is not None:
            ProfileAutoencoderConfigIO.save(self.autoencoder_cfg, self.ae_model_name, run_meta.metadata_directory)
        if self.image_ae_cfg is not None:
            ImageAutoencoderConfigIO.save(self.image_ae_cfg, self.image_ae_model_name, run_meta.metadata_directory)

        run_meta.save_run_summary(self.backbone_name, in_channels=self.backbone_in_channels, out_channels=self.backbone_out_channels, x_axis_length=z_len, n_gaussians=gaussian_cfg.n_default_gaussians, seed=self.entry.seed)

    def _profile_normalizer(self, metadata_directory, logger):
        """Loads the profile-autoencoder curve statistics and copies them into this run.

        Args:
            metadata_directory : Directory of this run receiving a copy of the stats.
            logger             : Logger for the statistics load.

        Returns:
            A ProfileNormalizer built from the imported statistics, or None when no
            profile autoencoder is coupled.
        """
        if self.autoencoder_cfg is None:
            return None

        stats = ProfileStats.load(self.profile_autoencoder_meta, logger=logger)
        stats.save(metadata_directory)

        return ProfileNormalizer(stats)

    def _make_trainer(self, model, backbone_cfg, z_axis, run_dir, logger, norm_stats, profile_normalizer, writer=None):
        """Returns a JEPA Trainer bound to this pipeline's trainer configuration."""
        return Trainer(model, backbone_cfg, z_axis, self.trainer_config, run_dir, logger, norm_stats, profile_normalizer, writer)

    def _train(self, run_meta, logger, model, backbone_cfg, z_axis, datasets, loaders, profile_normalizer):
        """Trains the module over the prepared loaders.

        Args:
            run_meta           : Training run metadata owning the run directory and writer.
            logger             : Logger for the training banners.
            model              : Assembled JepaModule.
            backbone_cfg       : Resolved backbone configuration.
            z_axis             : Z axis in metres.
            datasets           : Dataset dict keyed by split; the train split supplies the normalizer.
            loaders            : Tuple of the train, validation and test loaders.
            profile_normalizer : Curve normalizer, or None.

        Returns:
            Tuple of the trainer's results and the run directory.
        """
        train_loader, val_loader, test_loader = loaders
        norm_stats                            = datasets["train"].normalizer

        trainer = self._make_trainer(model, backbone_cfg, z_axis, run_meta.run_directory, logger, norm_stats, profile_normalizer, writer=run_meta.writer)

        trainer.maybe_run_loss_probe(train_loader, self.entry.probe)
        results = trainer.train(train_loader, val_loader, test_loader)

        return results, run_meta.run_directory

    def build_pretrain_trainer(self, work_dir, logger):
        """Builds a throwaway trainer for the pre-training feed and batch-size probes.

        Args:
            work_dir : Scratch directory receiving the probe's metadata.
            logger   : Logger for the preparation banners.

        Returns:
            Tuple of the trainer, the training dataset and the assembled module.
        """
        work_dir = Path(work_dir)

        loaders, datasets, z_axis, z_len = self._prepare_datasets(work_dir, logger, self.entry.seed)

        model, backbone_cfg = self._build_module(datasets, z_len, z_axis, logger)
        profile_normalizer  = self._profile_normalizer(work_dir / "meta", logger)

        trainer = self._make_trainer(model, backbone_cfg, z_axis, work_dir, logger, datasets["train"].normalizer, profile_normalizer)

        return trainer, datasets["train"], model

    def run(self):
        """Prepares data, builds the module, runs the overfit gate and trains the run.

        Returns:
            Tuple of the trainer's results and the run directory.
        """
        with TrainingRunMetadata(self.trainer_config, self.backbone_name, Path(self.trainer_config.io.logdir), self.entry.run_name) as run_meta:
            logger = run_meta.logger

            run_meta.save_dataset_fingerprint(self.entry.paths.dataset_path, self.entry.paths.parameters_path, enabled=self.entry.paths.fingerprint_dataset)

            loaders, datasets, z_axis, z_len = self._prepare_datasets(run_meta.run_directory, logger, self.entry.seed)

            model, backbone_cfg = self._build_module(datasets, z_len, z_axis, logger)

            self._save_metadata(run_meta, backbone_cfg, z_len)

            profile_normalizer = self._profile_normalizer(run_meta.metadata_directory, logger)

            self._run_overfit_check(run_meta, logger, datasets, z_len, z_axis, profile_normalizer)

            return self._train(run_meta, logger, model, backbone_cfg, z_axis, datasets, loaders, profile_normalizer)


class SingleTrainRunner(EntryConfigTrainRunner):
    """Entry runner for one JEPA training run, with optional inference afterwards."""

    pipeline_class = TrainingPipeline

    @property
    def label(self) -> str:
        """Run-name label encoding the backbone, head, coupling, slot count and augmentation."""
        n_gaussians = ConfigFactory(self.config).gaussian_config().n_default_gaussians
        return RunNaming.jepa_tag(self.config.backbone_name, self.config.backbone_head, JepaNamingSpec.resolve(self.config), n_gaussians, self.config.augmentation)

    def _resolve_run_name(self) -> str:
        """Returns the run directory name, composing the label with the run tag."""
        return RunNaming.compose(self.label, self.config.run_name or RunTag.now())

    def _run_inference(self, run_directory) -> None:
        """Runs backbone inference over the just-trained run directory.

        Args:
            run_directory : Directory of the finished training run.
        """
        from pipelines.backbone.inference.pipeline           import InferencePipeline
        from pipelines.shared.inference.inference_components import InferenceComponentsResolver

        inference_config = replace(self.config.inference, run_directory=Path(run_directory), output_subdir=None)
        components       = InferenceComponentsResolver.for_run(Path(run_directory))
        InferencePipeline(inference_config, components=components).run()

    def _skipped_result(self):
        """Returns the placeholder result used when a completed run is resumed and skipped."""
        return None, self.run_directory

    def run(self):
        """Trains the run and, when configured, runs inference on it.

        Returns:
            The trainer's results, or None when the run was skipped as already complete.
        """
        results, run_directory = super().run()

        if (self.config.infer_after or self.config.infer_at_end) and not self.unit_resume.skip_inference(self.config.inference.split):
            self._run_inference(run_directory)

        return results
