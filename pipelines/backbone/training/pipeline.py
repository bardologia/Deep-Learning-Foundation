"""End-to-end backbone training pipeline: dataset, model, gate check and trainer.

Wires the patch dataset, the backbone under the configured head and the trainer
together, persists the run metadata and configurations, optionally starts the
conv head at the training-label prior, and runs the overfit gate before the
real training loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from configuration.dataset                         import DatasetConfig
from configuration.training                        import BackboneTrainerConfig, OverfitCheckConfig
from models                                        import get_backbone
from pipelines.backbone.dataset.datasets           import MultiRegionDataset
from pipelines.backbone.dataset.pipeline           import DatasetPipeline
from pipelines.backbone.training.loss_terms        import LossComponentCatalog
from pipelines.backbone.training.trainer           import Trainer
from pipelines.shared.config.config_persistence    import BackboneModelConfigIO
from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.config.run_metadata          import TrainingRunMetadata
from pipelines.shared.model.model_builder          import ModelBuilder
from pipelines.shared.training.overfit_check       import OverfitCheck
from tools.data.gaussians                          import GaussianActivity, GaussianAxis, GaussianHead, GaussianSlotSorter
from tools.data.io                                 import FileIO
from tools.runtime.reproducibility                 import Reproducibility


class HeadPriorBias:
    """Starts the conv output head at the label prior of the training split.

    At PyTorch's default initialization the decoded embedding of a deep trunk is
    small, so every output channel starts as a near-constant equal to its head
    bias, drawn at random. This step replaces that bias with the channel's mean
    training label in the model's normalized output space, so the step-0
    prediction is the label prior. Slots follow the sorted_gt order (active
    slots by ascending z, inactive slots last), and the three channels of a slot
    are averaged over the training pixels where that slot is active, so pixels
    without a component in a slot do not pull its prior towards the absent-slot
    values. Only the training split's labels enter the means.

    Attributes:
        PARAMS             : Parameter names of one Gaussian slot, in channel order.
        ROW_CHUNK          : Label rows sorted and normalized at once, which bounds the transient memory.
        dataset            : Training split, a PatchDataset or a MultiRegionDataset of several regions.
        n_gaussians        : Gaussian slots the model predicts.
        activity_threshold : Physical amplitude above which a label slot counts as active, as recorded by the label run.
        logger             : Logger receiving the prior table.
    """

    PARAMS    = ("amplitude", "mu", "sigma")
    ROW_CHUNK = 256

    def __init__(self, dataset, n_gaussians: int, activity_threshold: float, logger) -> None:
        """Stores the training split and the slot layout the prior is computed for.

        Args:
            dataset            : Training split dataset carrying physical labels and the run's normalizer.
            n_gaussians        : Gaussian slots the model predicts.
            activity_threshold : Physical amplitude above which a label slot counts as active.
            logger             : Logger receiving the prior table.
        """
        self.dataset            = dataset
        self.n_gaussians        = n_gaussians
        self.activity_threshold = activity_threshold
        self.logger             = logger

    @property
    def channel_names(self) -> list[str]:
        """Names of the output channels, slot by slot, such as ``slot_1_sigma``."""
        return [f"slot_{slot}_{param}" for slot in range(self.n_gaussians) for param in self.PARAMS]

    def _head_bias(self, model) -> torch.nn.Parameter:
        """Returns the bias of the model's conv output head.

        Args:
            model : Freshly built backbone exposing ``config.head`` and, for the conv head, ``output_head``.

        Returns:
            The bias parameter of the head's output convolution, of shape (3 * n_gaussians,).

        Raises:
            ValueError : If the model does not use the conv head, or if the head's width differs from three channels per slot.
        """
        head = model.config.head
        if head != "conv":
            raise ValueError(f"prior_bias_init sets the bias of the conv head's single output convolution, but the model uses the '{head}' head; select backbone_head conv or turn prior_bias_init off")

        bias = model.output_head.bias
        if bias.numel() != len(self.PARAMS) * self.n_gaussians:
            raise ValueError(f"The conv head emits {bias.numel()} channels but the prior covers {self.n_gaussians} slots of {len(self.PARAMS)} parameters")

        return bias

    def _parts(self) -> list:
        """Returns the per-region datasets of the training split."""
        return list(self.dataset.parts) if isinstance(self.dataset, MultiRegionDataset) else [self.dataset]

    def _accumulate(self, labels: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> None:
        """Adds one region's active-pixel label sums and counts to the running totals.

        Args:
            labels : Physical training labels of one region, of shape (3 * n_gaussians, rows, cols), amplitude, mu in metres and sigma in metres per slot.
            sums   : Running per-channel sums in normalized output units, of shape (3 * n_gaussians,), updated in place.
            counts : Running per-slot active-pixel counts, of shape (n_gaussians,), updated in place.
        """
        n_params = len(self.PARAMS)

        for start in range(0, labels.shape[1], self.ROW_CHUNK):
            physical   = np.asarray(labels[:, start:start + self.ROW_CHUNK], dtype=np.float32)
            ordered    = GaussianSlotSorter.by_mean(physical, self.n_gaussians, self.activity_threshold)
            normalized = self.dataset.normalizer.normalize_output(ordered)

            for slot in range(self.n_gaussians):
                channels = slice(n_params * slot, n_params * (slot + 1))
                active   = GaussianActivity.is_active(ordered[n_params * slot], self.activity_threshold)

                sums[channels] += normalized[channels][:, active].sum(axis=1, dtype=np.float64)
                counts[slot]   += int(active.sum())

    def prior(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Computes the per-channel label prior of the training split.

        Returns:
            Tuple (prior, counts, n_pixels) of the prior of shape (3 * n_gaussians,) in normalized output units, the active training pixels of each slot of shape (n_gaussians,), and the number of training pixels.

        Raises:
            ValueError : If a region's labels do not hold three channels per slot, if a slot is active on no training pixel, or if a prior entry is not finite.
        """
        n_params   = len(self.PARAMS)
        n_channels = n_params * self.n_gaussians
        sums       = np.zeros(n_channels, dtype=np.float64)
        counts     = np.zeros(self.n_gaussians, dtype=np.int64)
        n_pixels   = 0

        for part in self._parts():
            labels = part.gt_selected
            if labels.shape[0] != n_channels:
                raise ValueError(f"Training region '{part.split_name}' carries {labels.shape[0]} label channels, but the prior needs {n_params} per slot for {self.n_gaussians} slots")

            self._accumulate(labels, sums, counts)
            n_pixels += int(labels.shape[1] * labels.shape[2])

        empty = [slot for slot in range(self.n_gaussians) if counts[slot] == 0]
        if empty:
            raise ValueError(f"Slots {empty} are active on no training pixel at activity threshold {self.activity_threshold:g}, so they have no label prior")

        prior = sums / np.repeat(counts, n_params)

        missing = [name for name, value in zip(self.channel_names, prior) if not np.isfinite(value)]
        if missing:
            raise ValueError(f"The training labels give a non-finite prior for {missing}; check the label run for NaN or infinite values on active pixels")

        return prior, counts, n_pixels

    def _log(self, prior: np.ndarray, default_bias: np.ndarray, counts: np.ndarray, n_pixels: int) -> None:
        """Logs the activity coverage and the prior that replaced each default bias.

        Args:
            prior        : Prior biases in normalized output units, of shape (3 * n_gaussians,).
            default_bias : Biases the head was initialized with, of shape (3 * n_gaussians,).
            counts       : Active training pixels of each slot, of shape (n_gaussians,).
            n_pixels     : Number of training pixels.
        """
        rows = {
            "Activity threshold" : f"{self.activity_threshold:g}",
            "Training pixels"    : f"{n_pixels:,}",
        }

        for slot in range(self.n_gaussians):
            rows[f"Slot {slot} active"] = f"{int(counts[slot]):,} ({counts[slot] / n_pixels:.1%})"

        for name, value, default in zip(self.channel_names, prior, default_bias):
            rows[name] = f"{value:.4f} (default init {default:.4f})"

        self.logger.section("[Head Prior Bias]")
        self.logger.kv_table(rows)

    def apply(self, model) -> dict:
        """Overwrites the conv head bias with the training-label prior.

        Args:
            model : Freshly built backbone with the conv head; every weight other than the head bias is left as built.

        Returns:
            JSON-serialisable record holding the activity threshold, the training and per-slot active pixel counts, the channel names, the default biases and the prior biases.
        """
        bias                    = self._head_bias(model)
        prior, counts, n_pixels = self.prior()
        default_bias            = bias.detach().cpu().numpy().astype(np.float64)

        with torch.no_grad():
            bias.copy_(torch.as_tensor(prior, dtype=bias.dtype, device=bias.device))

        self._log(prior, default_bias, counts, n_pixels)

        return {
            "activity_threshold" : self.activity_threshold,
            "training_pixels"    : n_pixels,
            "active_pixels"      : [int(count) for count in counts],
            "channels"           : self.channel_names,
            "default_bias"       : default_bias.tolist(),
            "prior_bias"         : prior.tolist(),
        }


class TrainingPipeline:
    """Runs one backbone training job from dataset preparation to test metrics.

    Attributes:
        trainer_config   : Trainer, curriculum and gaussian configuration.
        dataset_config   : Patch dataset configuration.
        backbone_name    : Registered backbone architecture name.
        model_config     : Resolved model configuration, or None to take registry defaults.
        seed             : Seed applied to every random source of the run.
        run_name         : Explicit run directory name, or None to derive one.
        overfit_check    : Configuration of the pre-training overfit gate.
        prior_bias_init  : Whether the conv head biases start at the training-label prior instead of the default initialization.
        configured_patch : Patch size and stride as configured, before the backbone's squaring requirement is applied.
        patch_size       : Patch size after the backbone's squaring requirement is applied.
    """

    MODEL_CONFIG_IO = BackboneModelConfigIO

    def __init__(
        self,
        trainer_config  : BackboneTrainerConfig,
        dataset_config  : DatasetConfig,
        backbone_name   : str,
        model_config    = None,
        seed            : int = 0,
        run_name        : str | None = None,
        overfit_check   : OverfitCheckConfig | None = None,
        prior_bias_init : bool = False,
    ) -> None:
        """Seeds the run and reconciles the patch geometry with the backbone requirements.

        Args:
            trainer_config  : Trainer, curriculum and gaussian configuration.
            dataset_config  : Patch dataset configuration.
            backbone_name   : Registered backbone architecture name.
            model_config    : Resolved model configuration, or None for registry defaults.
            seed            : Seed applied to every random source of the run.
            run_name        : Explicit run directory name, or None to derive one.
            overfit_check   : Overfit gate configuration; a default one when omitted.
            prior_bias_init : Whether the conv head biases start at the training-label prior, see HeadPriorBias.
        """

        self.trainer_config  = trainer_config
        self.dataset_config  = dataset_config
        self.backbone_name   = backbone_name
        self.model_config    = model_config
        self.seed            = seed
        self.run_name        = run_name
        self.overfit_check   = overfit_check if overfit_check is not None else OverfitCheckConfig()
        self.prior_bias_init = prior_bias_init

        patch                    = self.dataset_config.patch
        self.configured_patch    = (tuple(patch.size), tuple(patch.stride))
        patch.size, patch.stride = ModelBuilder.square_patch(self._square_patch_backbones(), patch.size, patch.stride)
        self.patch_size          = patch.size

        Reproducibility.seed_everything(self.seed)

        self.dataset_config.n_gaussians = trainer_config.gaussian.n_default_gaussians

    @property
    def effective_patch(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """The patch size and stride the run trains on, after the squaring requirement."""
        return (tuple(self.patch_size), tuple(self.dataset_config.patch.stride))

    @property
    def patch_squared(self) -> bool:
        """Whether the backbone's square-input requirement changed the configured patch geometry."""
        return self.effective_patch != self.configured_patch

    def _square_patch_backbones(self) -> list[str]:
        """Returns the backbones whose square-patch requirement constrains the patch geometry."""
        return [self.backbone_name]

    def _patch_geometry_label(self) -> str:
        """Returns the patch geometry cell of the execution table, naming the configured values when the backbone squared them."""
        size, stride = self.effective_patch
        effective    = f"patch {size} stride {stride}"

        if not self.patch_squared:
            return effective

        configured_size, configured_stride = self.configured_patch

        return f"configured patch {configured_size} stride {configured_stride} -> effective {effective} ({self.backbone_name} needs a square input)"

    def _patch_geometry_payload(self) -> dict:
        """Returns the configured and effective patch geometry recorded in run_summary.json."""
        return {
            "configured_size"   : list(self.configured_patch[0]),
            "configured_stride" : list(self.configured_patch[1]),
            "size"              : list(self.effective_patch[0]),
            "stride"            : list(self.effective_patch[1]),
            "squared"           : self.patch_squared,
        }

    def _build_dataset_pipeline(self, run_directory: Path, logger) -> DatasetPipeline:
        """Builds the patch dataset pipeline and derives the z axis from its profile length.

        Args:
            run_directory : Directory the dataset artefacts are written under.
            logger        : Logger passed to the dataset pipeline.

        Returns:
            The configured dataset pipeline, not yet run.
        """
        gaussian_cfg = self.trainer_config.gaussian

        dataset_pipeline = DatasetPipeline(
            config                 = self.dataset_config,
            training_run_directory = Path(run_directory),
            activity_threshold     = gaussian_cfg.activity_threshold,
            logger                 = logger,
            seed                   = self.seed,
            resume                 = self.trainer_config.training.resume,
        )

        self.dataset_config.z_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, dataset_pipeline.layout.profile_length)

        return dataset_pipeline

    def _run_overfit_check(self, train_dataset, in_channels: int, out_channels: int, z_axis) -> None:
        """Runs the pre-training gate that must overfit a tiny subset before real training.

        The gate trains a sanitized copy of the model on a single-stage loss with
        active normalization disabled, so a failure points at the model or data
        rather than the curriculum.

        Args:
            train_dataset : Training dataset the gate draws its samples from.
            in_channels   : Model input channel count.
            out_channels  : Model output channel count.
            z_axis        : Z sample positions in metres, of shape (profile_length,).
        """
        check = OverfitCheck(self.overfit_check, self.run_metadata.run_directory, self.logger)
        if not check.enabled:
            return

        check.announce()

        gate_trainer_config = check.sanitized_trainer_config(self.trainer_config)

        gate_stage                          = gate_trainer_config.curriculum.initial_stage
        gate_stage.use_active_normalization = False

        gate_trainer_config.curriculum.enabled  = False
        gate_trainer_config.curriculum.warmup   = gate_stage
        gate_trainer_config.curriculum.complete = gate_stage

        check.record("curriculum.enabled", False)
        check.record("gate_loss_stage", "warmup" if self.trainer_config.curriculum.enabled else "complete")
        check.record("gate_stage.use_active_normalization", False)

        base_config       = self.model_config if self.model_config is not None else ModelBuilder.config_from_registry(self.backbone_name, {})
        gate_model_config = check.sanitized_model_config(base_config)

        gate_model, gate_model_cfg = self._model_factory()(self.backbone_name, config=gate_model_config, **self._model_overrides(in_channels, out_channels))

        gate_trainer = Trainer(
            model      = gate_model,
            model_cfg  = gate_model_cfg,
            z_axis     = z_axis,
            config     = gate_trainer_config,
            run_dir    = check.work_directory,
            logger     = self.logger,
            norm_stats = train_dataset.normalizer,
            emit_docs  = False,
        )

        check.run(gate_trainer, train_dataset)

    def _build_model(self, in_channels: int, out_channels: int):
        """Builds the backbone at the given widths and logs its architecture and size.

        Args:
            in_channels  : Model input channel count.
            out_channels : Model output channel count.

        Returns:
            The model and its resolved model configuration.
        """
        model, model_cfg = self._model_factory()(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        n_total     = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params      = f"{n_trainable:,} trainable" if n_total == n_trainable else f"{n_trainable:,} trainable ({n_total:,} total, {n_total - n_trainable:,} frozen)"

        self.logger.section("[Model Built]")
        self.logger.kv_table({
            "Architecture" : self._architecture_label(),
            "Head"         : model_cfg.head,
            "In Channels"  : in_channels,
            "Out Channels" : out_channels,
            "Parameters"   : params,
        })
        return model, model_cfg

    def _architecture_label(self) -> str:
        """Returns the architecture name shown in the model log banner."""
        return self.backbone_name

    def _model_overrides(self, in_channels: int, out_channels: int) -> dict:
        """Returns the channel widths plus any image-size override the backbone requires."""
        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(self.backbone_name, self.patch_size))

        return overrides

    def _model_factory(self):
        """Returns the factory that instantiates the architecture family of this pipeline."""
        return get_backbone

    def _apply_prior_bias(self, model, train_dataset) -> None:
        """Sets the conv head biases to the training-label prior and records them in meta/head_prior_bias.json.

        Args:
            model         : Freshly built model whose conv head bias is overwritten.
            train_dataset : Training split the prior is computed from.
        """
        gaussian_cfg = self.trainer_config.gaussian
        prior_bias   = HeadPriorBias(train_dataset, gaussian_cfg.n_default_gaussians, gaussian_cfg.activity_threshold, self.logger)
        record       = prior_bias.apply(model)
        out_path     = self.run_metadata.metadata_directory / "head_prior_bias.json"

        FileIO.save_json(record, out_path)
        self.logger.info(f"Head prior bias saved: {out_path}")

    def _make_trainer(self, model, model_cfg, z_axis, norm_stats, run_dir: Path, logger, emit_docs: bool = True, writer=None):
        """Builds the trainer driving the training loop.

        Args:
            model      : The backbone to train.
            model_cfg  : Resolved model configuration.
            z_axis     : Z sample positions in metres, of shape (profile_length,).
            norm_stats : Normalizer of the dataset the model was fitted on.
            run_dir    : Directory the trainer writes checkpoints and logs into.
            logger     : Logger for the training loop.
            emit_docs  : Whether the trainer writes the architecture documentation.
            writer     : Optional TensorBoard writer.

        Returns:
            The configured trainer.
        """
        return Trainer(
            model      = model,
            model_cfg  = model_cfg,
            z_axis     = z_axis,
            config     = self.trainer_config,
            run_dir    = run_dir,
            logger     = logger,
            norm_stats = norm_stats,
            emit_docs  = emit_docs,
            writer     = writer,
        )

    def build_pretrain_trainer(self, work_dir: Path, logger):
        """Builds a throwaway trainer over the probe-union loss for the pretraining probes.

        Args:
            work_dir : Scratch directory the probe artefacts are written into.
            logger   : Logger for the probe run.

        Returns:
            Tuple of the trainer, the training dataset and the model.
        """
        work_dir         = Path(work_dir)
        dataset_pipeline = self._build_dataset_pipeline(work_dir, logger)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        gaussian_cfg = self.trainer_config.gaussian
        dataset      = datasets["train"]
        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)
        z_axis       = np.asarray(self.dataset_config.z_axis, dtype=np.float32)

        model, model_cfg = self._model_factory()(self.backbone_name, config=self.model_config, **self._model_overrides(in_channels, out_channels))

        trainer = self._make_trainer(model, model_cfg, z_axis, dataset.normalizer, work_dir, logger, emit_docs=False)
        trainer.criterion.set_curriculum(LossComponentCatalog.probe_union(self.trainer_config.curriculum), context="probe union of all curriculum stages")

        return trainer, dataset, model

    def run(self, probe_config=None, resolved_entry_config=None):
        """Prepares the data and model, runs the gate and trains, then saves test metrics.

        Args:
            probe_config          : Optional loss-scale probe configuration run before training.
            resolved_entry_config : Optional resolved entry config persisted into the run's docs directory.

        Returns:
            The training results returned by the trainer.
        """
        self.run_metadata = TrainingRunMetadata(
            trainer_config = self.trainer_config,
            model_name     = self.backbone_name,
            base_logdir    = Path(self.trainer_config.io.logdir),
            run_name       = self.run_name,
        )
        with self.run_metadata:
            self.logger = self.run_metadata.logger

            if resolved_entry_config is not None:
                ResolvedEntryConfigWriter.save(resolved_entry_config, self.run_metadata.run_directory, self.logger)

            if self.patch_squared:
                self.logger.warning(f"{self.backbone_name} needs a square input: {self._patch_geometry_label()}; pixels per batch and patches per epoch differ from the configured geometry, and the resolved entry config keeps the configured values")

            self.run_metadata.save_dataset_fingerprint(self.dataset_config.preprocessing_run_directory, self.dataset_config.parameters_path, enabled=self.dataset_config.fingerprint_dataset)

            self.logger.section("[PyTorch Training Pipeline Execution]")
            self.logger.kv_table({
                "Seed"           : self.seed,
                "Gaussian slots" : self.trainer_config.gaussian.n_default_gaussians,
                "Z axis"         : f"[{self.trainer_config.gaussian.x_min:g}, {self.trainer_config.gaussian.x_max:g}] m (sample count set by the dataset profile length)",
                "Patch geometry" : self._patch_geometry_label(),
            })

            self.dataset_pipeline = self._build_dataset_pipeline(self.run_metadata.run_directory, self.logger)

            gaussian_cfg = self.trainer_config.gaussian

            train_loader, val_loader, test_loader, datasets = self.dataset_pipeline.run()

            train_dataset = datasets["train"]
            in_channels   = train_dataset.input_channels
            n_gaussians   = gaussian_cfg.n_default_gaussians
            out_channels  = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, n_gaussians)
            z_axis        = np.asarray(self.dataset_config.z_axis, dtype=np.float32)
            x_axis_length = int(z_axis.size)

            model, model_cfg = self._build_model(in_channels=in_channels, out_channels=out_channels)

            if self.prior_bias_init:
                self._apply_prior_bias(model, train_dataset)

            self.run_metadata.save_trainer_config()
            self.run_metadata.save_model_config(model_cfg, self.backbone_name, config_io=self.MODEL_CONFIG_IO)

            self.run_metadata.save_run_summary(
                model_name       = self.backbone_name,
                in_channels      = in_channels,
                out_channels     = out_channels,
                x_axis_length    = x_axis_length,
                n_gaussians      = n_gaussians,
                seed             = self.seed,
                patch_geometry   = self._patch_geometry_payload(),
            )
            self.logger.ok(f"Run metadata saved : {self.run_metadata.run_directory}")

            self._run_overfit_check(train_dataset, in_channels, out_channels, z_axis)

            trainer = self._make_trainer(model, model_cfg, z_axis, train_dataset.normalizer, self.run_metadata.run_directory, self.logger, writer=self.run_metadata.writer)

            trainer.maybe_run_loss_probe(train_loader, probe_config)
            results = trainer.train(train_loader, val_loader, test_loader)
            self.run_metadata.save_test_metrics(trainer.test_metrics)

        return results
