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

from configuration.normalization                         import ChannelStats
from configuration.training                              import JepaTrainerConfig
from models                                              import get_backbone
from models.profile_autoencoder                          import get_profile_autoencoder
from models.image_autoencoder                            import get_image_autoencoder
from pipelines.profile_autoencoder.dataset.normalization import ProfileNormalizer, ProfileStats
from pipelines.shared.config.config_factory              import ConfigFactory
from pipelines.shared.config.run_metadata                import TrainingRunMetadata
from pipelines.backbone.dataset.pipeline                 import BackboneDatasetPreparation
from pipelines.backbone.dataset.stats                    import Stats
from pipelines.shared.model.model_builder                import ModelBuilder
from pipelines.shared.training.overfit_check             import OverfitCheck
from pipelines.shared.training.run_naming                import JepaNamingSpec, RunNaming
from pipelines.shared.training.training_runner           import EntryConfigTrainRunner
from pipelines.jepa.training.coupling                    import CouplingMode
from pipelines.jepa.training.loss                        import Loss as EmbeddingLoss
from pipelines.jepa.training.trainer                     import JepaModule, Trainer
from pipelines.shared.config.config_persistence          import ProfileAutoencoderConfigIO, ImageAutoencoderConfigIO, ProfileDatasetConfigIO
from tools.data.io                                       import FileIO
from tools.runtime.completion                            import CompletionMarker
from tools.runtime.reproducibility                       import Reproducibility
from tools.runtime.run_tag                               import RunTag


class CoupledRunProvenance:
    """Compares the dataset provenance of a coupled autoencoder run with the JEPA run's own.

    A frozen image encoder only ever saw the channel selection, the secondary
    channels and the input normalization of its own run, so the channel
    selection and the secondaries must match exactly; the preprocessing run,
    the parameter run and the fitted input statistics may legitimately differ
    (another split, another parameter run) and are reported as warnings naming
    both values. A profile autoencoder is compared on the preprocessing and
    parameter runs its curves were rendered from and on the mixture settings
    behind those curves.

    Attributes:
        STATS_TOLERANCE : Largest per-channel shift of location or scale, in units of the autoencoder run's scale, accepted without a warning.
        dataset_config  : Dataset configuration of the JEPA run.
        logger          : Logger receiving the warnings.
    """

    STATS_TOLERANCE = 0.01

    def __init__(self, dataset_config, logger) -> None:
        """Binds this run's dataset configuration and the logger.

        Args:
            dataset_config : Dataset configuration of the JEPA run.
            logger         : Logger receiving the warnings.
        """
        self.dataset_config = dataset_config
        self.logger         = logger

    @staticmethod
    def _path(value):
        """Returns a path value as a string, or None when it was not set."""
        return None if value is None else str(value)

    @staticmethod
    def _labels(value):
        """Returns the secondary labels as a list, or None when every secondary is loaded."""
        return None if value is None else list(value)

    def _compare(self, label: str, item: str, autoencoder_value, own_value) -> dict:
        """Records one compared item, warning when the coupled run and this run disagree.

        Args:
            label             : Coupled autoencoder, "profile" or "image".
            item              : Name of the compared item.
            autoencoder_value : Value the autoencoder run was trained with.
            own_value         : Value this JEPA run uses.

        Returns:
            Mapping with the autoencoder run's and this run's value.
        """
        if autoencoder_value != own_value:
            self.logger.warning(f"The coupled {label} autoencoder was trained with {item} {autoencoder_value!r} but this JEPA run uses {own_value!r}; the imported weights were fitted on another {item}, which the JEPA result will reflect.")

        return {"autoencoder_run": autoencoder_value, "this_run": own_value}

    def _compare_input_stats(self, autoencoder_stats: ChannelStats, own_stats: ChannelStats) -> dict:
        """Records both runs' input normalization and warns when they differ beyond the tolerance.

        Args:
            autoencoder_stats : Per-channel input statistics the image autoencoder run was trained with.
            own_stats         : Per-channel input statistics fitted on this run's training split.

        Returns:
            Mapping with both runs' locations and scales, the largest shift in
            autoencoder-scale units and the tolerance.

        Raises:
            ValueError : If the two statistics cover a different number of channels.
        """
        if autoencoder_stats.n_channels != own_stats.n_channels:
            raise ValueError(f"The coupled image autoencoder's input normalization covers {autoencoder_stats.n_channels} channels but this run fitted {own_stats.n_channels}; the two runs do not assemble the same input stack.")

        ae_loc    = np.asarray(autoencoder_stats.loc,   dtype=np.float64)
        ae_scale  = np.asarray(autoencoder_stats.scale, dtype=np.float64)
        own_loc   = np.asarray(own_stats.loc,           dtype=np.float64)
        own_scale = np.asarray(own_stats.scale,         dtype=np.float64)

        deviation = float(np.max(np.maximum(np.abs(ae_loc - own_loc), np.abs(ae_scale - own_scale)) / ae_scale))

        if deviation > self.STATS_TOLERANCE:
            self.logger.warning(f"Input normalization of this run differs from the coupled image autoencoder's: the largest per-channel shift of location or scale is {deviation:.3g} autoencoder-scale units (tolerance {self.STATS_TOLERANCE:g}); the frozen encoder receives inputs standardized with statistics it was not trained on. Autoencoder loc {ae_loc.round(4).tolist()} scale {ae_scale.round(4).tolist()}; this run loc {own_loc.round(4).tolist()} scale {own_scale.round(4).tolist()}.")

        return {
            "autoencoder_run" : {"loc": list(autoencoder_stats.loc), "scale": list(autoencoder_stats.scale)},
            "this_run"        : {"loc": list(own_stats.loc),         "scale": list(own_stats.scale)},
            "max_deviation"   : deviation,
            "tolerance"       : self.STATS_TOLERANCE,
        }

    def image(self, meta_directory, input_stats: ChannelStats) -> dict:
        """Checks an image autoencoder run against this run and returns the provenance record.

        Args:
            meta_directory : The image autoencoder run's meta directory, holding its dataset-creation config and normalization statistics.
            input_stats    : Per-channel input statistics fitted on this run's training split.

        Returns:
            Record of the compared preprocessing run, parameter run, input
            configuration, secondary labels and input statistics.

        Raises:
            ValueError : If the input configuration or the secondary labels differ, so the frozen encoder would receive channels it never saw.
        """
        payload           = FileIO.load_json(Path(meta_directory) / "dataset_creation_config.json")
        autoencoder_stats = ChannelStats.from_dict(FileIO.load_json(Path(meta_directory) / Stats.FILENAME)["input_stats"])

        own_input  = self.dataset_config.input_config.as_dict()
        own_labels = self._labels(self.dataset_config.secondary_labels)
        ae_labels  = self._labels(payload["secondary_labels"])

        if payload["input_config"] != own_input:
            raise ValueError(f"The coupled image autoencoder was trained on input_config {payload['input_config']} but this JEPA run assembles {own_input}; the frozen encoder never saw that channel representation. Match the input settings of the image autoencoder run or couple an autoencoder trained on them.")

        if ae_labels != own_labels:
            raise ValueError(f"The coupled image autoencoder was trained on secondary channels {ae_labels} but this JEPA run loads {own_labels}; the channel count may agree while the secondaries behind the channels do not. Match the secondary labels of the image autoencoder run or couple an autoencoder trained on them.")

        return {
            "preprocessing_run_directory" : self._compare("image", "preprocessing run", payload["preprocessing_run_directory"], self._path(self.dataset_config.preprocessing_run_directory)),
            "parameters_path"             : self._compare("image", "parameters path",   payload["parameters_path"],             self._path(self.dataset_config.parameters_path)),
            "input_config"                : own_input,
            "secondary_labels"            : own_labels,
            "input_stats"                 : self._compare_input_stats(autoencoder_stats, input_stats),
        }

    def profile(self, meta_directory, gaussian_cfg) -> dict:
        """Checks a profile autoencoder run against this run and returns the provenance record.

        Args:
            meta_directory : The profile autoencoder run's meta directory, holding its persisted dataset configuration.
            gaussian_cfg   : Gaussian mixture configuration of this run, whose curves the encoder will embed.

        Returns:
            Record of the compared preprocessing run, parameter run, Gaussian
            count and activity threshold.
        """
        config = ProfileDatasetConfigIO.load(meta_directory)

        return {
            "preprocessing_run_directory" : self._compare("profile", "preprocessing run",  self._path(config.preprocessing_run_directory), self._path(self.dataset_config.preprocessing_run_directory)),
            "parameters_path"             : self._compare("profile", "parameters path",    self._path(config.parameters_path),             self._path(self.dataset_config.parameters_path)),
            "n_gaussians"                 : self._compare("profile", "n_gaussians",        config.n_gaussians,                             gaussian_cfg.n_default_gaussians),
            "activity_threshold"          : self._compare("profile", "activity_threshold", config.activity_threshold,                      gaussian_cfg.activity_threshold),
        }

    @staticmethod
    def rows(record: dict) -> dict:
        """Formats a provenance record as log-table rows.

        Args:
            record : Record returned by `image` or `profile`.

        Returns:
            One row per plain item, a pair of rows (autoencoder run, this run)
            per compared item, and a one-line summary of the input statistics.
        """
        rows = {}
        for item, value in record.items():
            if item == "input_stats":
                rows["Input stats shift"] = f"{value['max_deviation']:.3g} autoencoder-scale units (tolerance {value['tolerance']:g})"
            elif isinstance(value, dict) and "autoencoder_run" in value:
                rows[f"{item} (AE run)"]   = value["autoencoder_run"]
                rows[f"{item} (this run)"] = value["this_run"]
            else:
                rows[item] = value

        return rows


class TrainingPipeline:
    """Builds and trains a JEPA module from an entry configuration.

    Attributes:
        COUPLED_RECORD_FILENAME  : Name of the meta file recording the imported autoencoder checkpoints.
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
        patch_squaring           : Warning the run logs because squaring changed the configured patch geometry, or None.
        backbone_name            : Registry name of the backbone.
        profile_ae_state         : Cached profile autoencoder state dict, loaded on first use.
        image_ae_state           : Cached image autoencoder state dict, loaded on first use.
        coupled_autoencoders     : Per-coupling record of the imported checkpoint (epochs, validation loss, commit) and its dataset provenance, keyed by "profile" or "image" once loaded.
    """

    COUPLED_RECORD_FILENAME = "coupled_autoencoders.json"

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

        self.backbone_name        = entry_config.backbone_name
        self.profile_ae_state     = None
        self.image_ae_state       = None
        self.coupled_autoencoders = {}

        patch                    = self.dataset_config.patch
        self.patch_squaring      = ModelBuilder.squaring_note([self.backbone_name], patch.size, patch.stride)
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
            image_autoencoder = self._load_image_autoencoder(logger, datasets["train"].normalizer.stats.input_stats)
            backbone_in       = self.image_ae_cfg.embedding_dim

        profile_autoencoder = None
        if self.autoencoder_cfg is not None:
            if self.autoencoder_cfg.profile_length != z_len:
                raise ValueError(f"Profile autoencoder was trained with profile_length={self.autoencoder_cfg.profile_length} but the JEPA dataset produces curves of length {z_len}; retrain the profile autoencoder on this dataset's z axis before coupling it.")

            backbone_out        = self.autoencoder_cfg.embedding_dim
            profile_autoencoder = self._load_profile_autoencoder(z_axis, logger)
        else:
            backbone_out = self._gaussian_out_channels()

        self.backbone_in_channels  = backbone_in
        self.backbone_out_channels = backbone_out

        backbone, backbone_cfg = self._build_backbone(backbone_in, backbone_out, self.dataset_config.patch.size, config=backbone_config)
        module                 = JepaModule(backbone, profile_autoencoder=profile_autoencoder, image_autoencoder=image_autoencoder)

        self._apply_coupling(module)
        self._log_module(logger, module, backbone, dataset_in_channels, backbone_in, backbone_out)
        return module, backbone_cfg

    def _apply_coupling(self, module) -> None:
        """Applies the configured frozen or fine-tune mode to each coupled autoencoder.

        Freshly built autoencoders keep every weight trainable, and the trainer
        applies the coupling modes only when it is constructed; applying them
        here already makes the parameter counts logged next describe what the
        optimizer will actually train.

        Args:
            module : Assembled JepaModule.
        """
        if module.profile_autoencoder is not None:
            CouplingMode(self.trainer_config.profile_autoencoder_mode, "profile autoencoder").apply(module.profile_autoencoder)

        if module.image_autoencoder is not None:
            CouplingMode(self.trainer_config.image_autoencoder_mode, "image autoencoder").apply(module.image_autoencoder)

    def _log_module(self, logger, module, backbone, dataset_in: int, backbone_in: int, backbone_out: int) -> None:
        """Prints the built-module table with widths, parameter counts after coupling, and coupling state.

        Args:
            logger       : Logger receiving the section and table.
            module       : Assembled JepaModule with the coupling modes applied.
            backbone     : Backbone submodule, counted separately.
            dataset_in   : Input channel count produced by the dataset.
            backbone_in  : Input channel count seen by the backbone.
            backbone_out : Output channel count emitted by the backbone.
        """
        backbone_params  = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        info = {
            "Backbone"             : self.backbone_name,
            "Dataset Channels"     : dataset_in,
            "Backbone In"          : backbone_in,
            "Backbone Out"         : backbone_out,
            "Backbone Parameters"  : f"{backbone_params:,}",
            "Trainable Parameters" : f"{trainable_params:,} (after coupling; what the optimizer trains)",
            "Frozen Parameters"    : f"{total_params - trainable_params:,}",
        }

        if self.autoencoder_cfg is not None:
            info["Target Provider"] = self.trainer_config.target_provider
            info["Embedding Loss"]  = EmbeddingLoss.describe_terms(self.trainer_config.embedding_loss)
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

    def _load_profile_autoencoder(self, z_axis, logger):
        """Instantiates the profile autoencoder and loads its pretrained weights.

        The checkpoint is read once; on that first load the imported run is
        checked for completion, its dataset provenance is compared with this
        run's, and its epoch, validation loss, commit and provenance are
        recorded and logged.

        Args:
            z_axis : Z axis in metres of the JEPA dataset, compared against the axis stored in the checkpoint.
            logger : Logger receiving the coupled-run table and provenance warnings on the first load.

        Returns:
            The profile autoencoder with the checkpoint state loaded.
        """
        autoencoder, _ = get_profile_autoencoder(self.ae_model_name, self.autoencoder_cfg)
        ckpt_path      = self.trainer_config.profile_autoencoder_checkpoint

        if self.profile_ae_state is None:
            self.validate_checkpoint(ckpt_path, "profile")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self._validate_profile_axis(np.asarray(ckpt["z_axis"], dtype=np.float32), np.asarray(z_axis, dtype=np.float32), ckpt_path)

            record               = self._checkpoint_record(ckpt_path, ckpt)
            record["provenance"] = CoupledRunProvenance(self.dataset_config, logger).profile(self.profile_autoencoder_meta, self.trainer_config.gaussian)

            self.profile_ae_state                = ckpt["params"]
            self.coupled_autoencoders["profile"] = record
            self._log_coupled(logger, "profile")

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

    def _load_image_autoencoder(self, logger, input_stats):
        """Instantiates the image autoencoder and loads its pretrained weights.

        The checkpoint is read once; on that first load the imported run is
        checked for completion, its dataset provenance and input normalization
        are compared with this run's, and the checkpoint and provenance are
        recorded and logged.

        Args:
            logger      : Logger receiving the coupled-run table and provenance warnings on the first load.
            input_stats : Per-channel input statistics fitted on this run's training split.

        Returns:
            The image autoencoder with the checkpoint state loaded.
        """
        autoencoder, _ = get_image_autoencoder(self.image_ae_model_name, self.image_ae_cfg)
        ckpt_path      = self.trainer_config.image_autoencoder_checkpoint

        if self.image_ae_state is None:
            self.validate_checkpoint(ckpt_path, "image")

            ckpt                 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            record               = self._checkpoint_record(ckpt_path, ckpt)
            record["provenance"] = CoupledRunProvenance(self.dataset_config, logger).image(self.image_autoencoder_meta, input_stats)

            self.image_ae_state                = ckpt["params"]
            self.coupled_autoencoders["image"] = record
            self._log_coupled(logger, "image")

        autoencoder.load_state_dict(self.image_ae_state)
        return autoencoder

    @staticmethod
    def validate_checkpoint(ckpt_path, label: str) -> None:
        """Requires the autoencoder checkpoint to exist and its run to have finished.

        A run that is still training, crashed or was aborted already holds a
        best_model.pt from an early epoch, so the file alone proves nothing; the
        run's completion marker does.

        Args:
            ckpt_path : Path to the expected best_model.pt.
            label     : Label used in the error message ("profile", "image").

        Raises:
            FileNotFoundError : If the checkpoint file is missing.
            RuntimeError      : If the run directory holding the checkpoint carries no completion marker.
        """
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"{label} autoencoder checkpoint '{ckpt_path}' does not exist; expected 'best_model.pt' under the selected {label} autoencoder run directory.")

        run_directory = Path(ckpt_path).parent
        if not CompletionMarker.is_complete(run_directory):
            raise RuntimeError(f"{label} autoencoder run '{run_directory}' has no {CompletionMarker.FILENAME}: the run is still training, crashed or was aborted, so its '{Path(ckpt_path).name}' is a partial checkpoint. Couple a finished run, or wait for this one to complete.")

    @staticmethod
    def _checkpoint_record(ckpt_path, ckpt: dict) -> dict:
        """Describes the imported checkpoint by its epochs, validation loss, training length and commit.

        Args:
            ckpt_path : Path of the loaded checkpoint, inside the autoencoder run directory.
            ckpt      : Loaded checkpoint payload with epoch, best_epoch and best_val_loss.

        Returns:
            Mapping with the run directory, checkpoint path, one-based checkpoint
            and best epochs, best validation loss, completed and total epochs,
            early-stopping flag and the git commit the autoencoder run was trained
            from. A run whose completion marker was backfilled records no epoch
            counts, so those three fields are None and "marker_backfilled" says why.
        """
        run_directory = Path(ckpt_path).parent
        summary       = FileIO.load_json(run_directory / "meta" / "run_summary.json")
        marker        = CompletionMarker.payload(run_directory)
        backfilled    = "backfilled" in marker

        return {
            "run_directory"     : str(run_directory),
            "checkpoint"        : str(ckpt_path),
            "epoch"             : int(ckpt["epoch"]) + 1,
            "best_epoch"        : int(ckpt["best_epoch"]) + 1,
            "best_val_loss"     : float(ckpt["best_val_loss"]),
            "epochs_completed"  : None if backfilled else int(marker["epochs_completed"]),
            "epochs_total"      : None if backfilled else int(marker["epochs_total"]),
            "early_stopped"     : None if backfilled else bool(marker["early_stopped"]),
            "marker_backfilled" : backfilled,
            "git_commit"        : summary["git_commit"],
        }

    def _log_coupled(self, logger, label: str) -> None:
        """Prints the coupled-run table of one imported autoencoder.

        Args:
            logger : Logger receiving the section and table.
            label  : Coupling key, "profile" or "image".
        """
        record = self.coupled_autoencoders[label]

        rows = {
            "Run directory"    : record["run_directory"],
            "Checkpoint"       : record["checkpoint"],
            "Checkpoint epoch" : record["epoch"],
            "Best epoch"       : f"{record['best_epoch']} (val_loss {record['best_val_loss']:.6g})",
            "Epochs completed" : "not recorded (backfilled completion marker)" if record["marker_backfilled"] else f"{record['epochs_completed']} of {record['epochs_total']}" + (" (early stopped)" if record["early_stopped"] else ""),
            "Git commit"       : record["git_commit"],
        }
        rows.update(CoupledRunProvenance.rows(record["provenance"]))

        logger.section(f"[Coupled {label.title()} Autoencoder]")
        logger.kv_table(rows)

    def _save_metadata(self, run_meta, backbone_cfg, z_len: int) -> None:
        """Writes the trainer, backbone, autoencoder, coupled-run and run-summary metadata files.

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

        FileIO.save_json(self.coupled_autoencoders, run_meta.metadata_directory / self.COUPLED_RECORD_FILENAME)

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

            if self.patch_squaring is not None:
                logger.warning(self.patch_squaring)

            logger.section("[JEPA Training Pipeline Execution]")
            logger.kv_table({
                "Backbone"      : self.backbone_name,
                "Backbone Head" : self.entry.backbone_head,
                "Seed"          : self.entry.seed,
                "Run Directory" : str(run_meta.run_directory),
            })

            run_meta.save_dataset_fingerprint(self.entry.paths.dataset_path, self.entry.paths.parameters_path, enabled=self.entry.paths.fingerprint_dataset)

            loaders, datasets, z_axis, z_len = self._prepare_datasets(run_meta.run_directory, logger, self.entry.seed)

            model, backbone_cfg = self._build_module(datasets, z_len, z_axis, logger)

            self._save_metadata(run_meta, backbone_cfg, z_len)
            logger.ok(f"Run metadata saved : {run_meta.run_directory}")

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
