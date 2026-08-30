"""GPU probes finding the largest training batch size that fits a VRAM budget.

Each probe builds the real dataset, model and trainer of one benchmark model,
then measures the peak memory of actual training steps across candidate batch
sizes. The backbone probe assembles its context from the benchmark's sweep
settings; the entry probe borrows the throwaway trainer a training pipeline
builds for its own preflight, covering the JEPA and autoencoder families. When
the LR range test is enabled it runs on the same trainer at the probed batch
size, and the selected multiplier is stored in the probe result for the
training worker to apply.
"""

from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing  import Callable

import torch

from configuration.benchmark                 import BenchmarkConfig
from models                                  import get_backbone
from pipelines.shared.model.model_builder    import ModelBuilder
from pipelines.backbone.dataset.pipeline     import DatasetPipeline
from pipelines.backbone.training.loss_terms  import LossComponentCatalog
from pipelines.backbone.training.trainer     import Trainer
from pipelines.shared.config.config_factory  import ConfigFactory
from tools.data.gaussians                    import GaussianAxis, GaussianHead
from tools.monitoring.logger                 import Logger
from tools.runtime.reproducibility           import Reproducibility
from tools.training.pretraining.batch_finder import BatchSizeFinder, TrainStepMemoryProbe
from tools.training.pretraining.lr_finder    import LrRangeTest


class MaxBatchProbeBase:
    """Shared skeleton of the per-model VRAM probe and its optional LR range test.

    Subclasses supply ``_build_probe_setup`` returning the trainer and training
    dataset the measured steps run on; the skeleton seeds, measures the CUDA
    context, caps the search ceiling at the dataset size, runs the batch-size
    ladder and, when enabled, the LR range test at the probed batch size.

    Attributes:
        config        : Benchmark configuration supplying the probe and dataset settings.
        model_name    : Model key under probe.
        budget_gb     : Peak-memory budget in gigabytes.
        ceiling       : Largest batch size the search may return.
        measure_steps : Training steps executed per candidate batch size.
        seed          : Seed applied before building the probe context.
        device        : CUDA device the probe runs on.
        context_gb    : Memory in gigabytes held by the CUDA context itself.
        work_dir      : Directory the probe writes its log and result into.
        logger        : Logger for the probe.
    """

    CONTEXT_WARN_GB = 1.5

    def __init__(self, config: BenchmarkConfig, model_name: str, work_dir: Path) -> None:
        """Stores the probe settings and opens the log under the work directory.

        Args:
            config     : Benchmark configuration.
            model_name : Model key under probe.
            work_dir   : Directory the probe writes into.
        """
        self.config     = config
        self.model_name = model_name

        self.budget_gb     = config.max_batch.vram_budget_gb
        self.ceiling       = config.max_batch.max_batch
        self.measure_steps = config.max_batch.measure_steps
        self.seed          = config.max_batch.seed

        self.device     = torch.device("cuda")
        self.context_gb = 0.0
        self.work_dir   = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_dir=str(self.work_dir / "logs"), name="max_batch", level="INFO")

    def _measure_context(self) -> float:
        """Returns the memory in gigabytes already occupied on the device before any model."""
        return TrainStepMemoryProbe.measure_context(self.device)

    def _build_probe_setup(self) -> tuple:
        """Returns the (trainer, dataset) pair the probe steps run on; implemented by subclasses."""
        raise NotImplementedError

    def _candidates(self) -> list[int]:
        """Returns the power-of-two batch sizes searched up to the ceiling."""
        return BatchSizeFinder.candidates(self.ceiling)

    def _release(self, trainer) -> None:
        """Drops gradients and the caching allocator's blocks after an out-of-memory trial."""
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def _find_lr(self, trainer, dataset, batch_size: int) -> dict:
        """Runs the LR range test at the probed batch size and returns its record fields.

        The trainer was built with the learning-rate scale of the configured
        batch size, so its warmup-free base rates are first rescaled to the
        probed batch under the batch-size rule; the selected multiplier is then
        relative to the effective rates the training worker will run with.

        Args:
            trainer    : Probe trainer whose optimizer and loss path the test drives.
            dataset    : Training dataset the test batches are drawn from.
            batch_size : Batch size resolved by the batch-size search.

        Returns:
            Dict with "lr_multiplier", "lr_min_loss_multiplier",
            "lr_divergence_multiplier" and "lr_rungs_run"; every entry is None
            when the test found no usable response.
        """
        settings = self.config.find_lr

        rescale          = batch_size / self.config.training.batch_size if self.config.training.scale_lr_with_batch else 1.0
        trainer.base_lrs = [float(lr) * rescale for lr in trainer.base_lrs]

        self.logger.subsection(f"LR range test at batch {batch_size}: base rates rescaled x{rescale:.4g} from the configured batch {self.config.training.batch_size} under the batch-size rule")

        test = LrRangeTest(
            trainer           = trainer,
            dataset           = dataset,
            device            = self.device,
            logger            = self.logger,
            scale_min         = settings.scale_min,
            scale_max         = settings.scale_max,
            rungs             = settings.rungs,
            rung_steps        = settings.rung_steps,
            probe_batches     = settings.probe_batches,
            divergence_factor = settings.divergence_factor,
            safety_factor     = settings.safety_factor,
            min_improvement   = settings.min_improvement,
            noise_sigmas      = settings.noise_sigmas,
            seed              = settings.seed,
        )

        outcome = test.run(batch_size)

        if outcome is None:
            self.logger.subsection(f"{self.model_name}: LR range test found no usable response; training keeps the configured learning rate")
            return {"lr_multiplier": None, "lr_min_loss_multiplier": None, "lr_divergence_multiplier": None, "lr_rungs_run": None}

        divergence = f"x{outcome['divergence_multiplier']:.4g}" if outcome["divergence_multiplier"] is not None else "not reached"

        self.logger.subsection(f"{self.model_name}: LR multiplier x{outcome['multiplier']:.4g} (loss minimum at x{outcome['min_loss_multiplier']:.4g}, divergence at {divergence})")

        return {
            "lr_multiplier"            : outcome["multiplier"],
            "lr_min_loss_multiplier"   : outcome["min_loss_multiplier"],
            "lr_divergence_multiplier" : outcome["divergence_multiplier"],
            "lr_rungs_run"             : outcome["rungs_run"],
        }

    def run(self) -> dict:
        """Searches for the largest batch size fitting the budget and reports the outcome.

        Failures are captured rather than raised, so one unbuildable model does
        not abort the benchmark sweep.

        Returns:
            Dict with the model key, status, chosen batch size, peak and context
            memory in gigabytes, the per-candidate trials, the LR range-test
            multiplier fields, the probe settings and the traceback of any
            failure.
        """
        result = {
            "model"                  : self.model_name,
            "status"                 : None,
            "batch_size"             : None,
            "peak_gb"                : None,
            "budget_gb"              : self.budget_gb,
            "ceiling"                : self.ceiling,
            "context_gb"             : None,
            "trials"                   : [],
            "lr_multiplier"            : None,
            "lr_min_loss_multiplier"   : None,
            "lr_divergence_multiplier" : None,
            "lr_rungs_run"             : None,
            "error"                    : None,
            "settings"                 : self.config.max_batch_settings(),
        }

        try:
            Reproducibility.seed_everything(self.seed)

            self.context_gb      = self._measure_context()
            result["context_gb"] = self.context_gb
            self.logger.subsection(f"CUDA context: {self.context_gb:.2f} GB")
            if self.context_gb > self.CONTEXT_WARN_GB:
                self.logger.warning(f"Context {self.context_gb:.2f} GB exceeds a bare CUDA context; other processes are using this GPU and shrink the measured budget")

            trainer, dataset = self._build_probe_setup()

            ceiling = min(self.ceiling, len(dataset))
            if ceiling < self.ceiling:
                self.logger.warning(f"Ceiling lowered from {self.ceiling} to {ceiling}: the train split holds {len(dataset)} samples")

            finder = BatchSizeFinder(
                trial_step = lambda batch_size: TrainStepMemoryProbe(trainer, dataset, self.measure_steps, self.device, self.context_gb)(batch_size),
                budget_gb  = self.budget_gb,
                ceiling    = ceiling,
                device     = self.device,
                logger     = self.logger,
                model_name = self.model_name,
                context_gb = self.context_gb,
                on_oom     = lambda: self._release(trainer),
            )

            result.update(finder.run())

            self.logger.subsection(f"{self.model_name}: status {result['status']}, batch size {result['batch_size']}, peak {result['peak_gb']} GB")

            if result["status"] == "PASS" and self.config.find_lr.enabled:
                result.update(self._find_lr(trainer, dataset, int(result["batch_size"])))

            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()
            self.logger.error(f"{self.model_name}: probe failed\n{result['error']}")

        self.logger.close()

        return result


class MaxBatchProbe(MaxBatchProbeBase):
    """Probes one backbone benchmark model, assembling its context from the sweep settings.

    Attributes:
        overrides : Model configuration overrides, typically the size-matched widths.
    """

    def __init__(self, config: BenchmarkConfig, model_name: str, overrides: dict, work_dir: Path) -> None:
        """Stores the probe settings and the size-match overrides.

        Args:
            config     : Benchmark configuration.
            model_name : Model key (backbone and head) under probe.
            overrides  : Model configuration overrides applied when building the model.
            work_dir   : Directory the probe writes into.
        """
        super().__init__(config, model_name, work_dir)
        self.overrides = overrides

    def _build_context(self):
        """Builds the dataset the probe trains on, at the benchmark's loss and patch settings.

        Returns:
            Tuple of the trainer config, the dataset config, the training
            dataset and the Gaussian configuration.
        """
        factory        = ConfigFactory(self.config)
        dataset_config = factory.training_dataset_config()
        trainer_config = factory.training_trainer_config(logdir=self.work_dir)

        dataset_config.input_config = self.config.input
        trainer_config.curriculum   = LossComponentCatalog.combined_curriculum(self.config.sweep_loss_components, base=self.config.loss)

        name, _head              = ModelBuilder.split_key(self.model_name)
        patch                    = dataset_config.patch
        patch.size, patch.stride = ModelBuilder.square_patch([name], patch.size, patch.stride)

        gaussian_cfg               = trainer_config.gaussian
        dataset_config.n_gaussians = gaussian_cfg.n_default_gaussians

        dataset_pipeline = DatasetPipeline(config=dataset_config, training_run_directory=self.work_dir, logger=self.logger, seed=self.seed)

        profile_length        = dataset_pipeline.layout.profile_length
        dataset_config.z_axis = GaussianAxis.build(gaussian_cfg.x_min, gaussian_cfg.x_max, profile_length)

        _train_loader, _val_loader, _test_loader, datasets = dataset_pipeline.run()

        return trainer_config, dataset_config, datasets["train"], gaussian_cfg

    def _build_model(self, dataset_config, dataset, gaussian_cfg):
        """Builds the probed model at the dataset input width and Gaussian output width.

        Args:
            dataset_config : Dataset configuration supplying the patch size.
            dataset        : Training dataset supplying the input channel count.
            gaussian_cfg   : Gaussian configuration supplying the output width.

        Returns:
            The model and its resolved model configuration.
        """
        name, head   = ModelBuilder.split_key(self.model_name)
        model_config = ModelBuilder.config_from_registry(name, self.overrides, head=head)

        in_channels  = dataset.input_channels
        out_channels = GaussianHead.total_channels(gaussian_cfg.params_per_gaussian, gaussian_cfg.n_default_gaussians)

        overrides = {"in_channels": in_channels, "out_channels": out_channels}
        overrides.update(ModelBuilder.image_size_override(name, dataset_config.patch.size))

        return get_backbone(name, config=model_config, **overrides)

    def _build_trainer(self, trainer_config, dataset_config, model, model_cfg, dataset):
        """Builds a trainer pinned to the complete loss stage, in train mode.

        The complete stage is the heaviest composition the run will use, so the
        measured peak bounds every later epoch.

        Args:
            trainer_config : Trainer configuration for the probe.
            dataset_config : Dataset configuration supplying the z axis.
            model          : The probed model.
            model_cfg      : Resolved model configuration.
            dataset        : Training dataset supplying the normalizer.

        Returns:
            The trainer whose training step the probe times.
        """
        trainer = Trainer(
            model      = model,
            model_cfg  = model_cfg,
            z_axis     = dataset_config.z_axis,
            config     = trainer_config,
            run_dir    = self.work_dir,
            logger     = self.logger,
            norm_stats = dataset.normalizer,
            emit_docs  = False,
        )

        trainer.criterion.set_curriculum(trainer_config.curriculum.complete, context="VRAM probe on the complete stage")
        trainer.model.train()

        return trainer

    def _build_probe_setup(self) -> tuple:
        """Builds the backbone probe context and validates the dataset channel width.

        Returns:
            Tuple of the trainer and the training dataset.

        Raises:
            RuntimeError : If the dataset channel count differs from the width capacity matching counted with.
        """
        trainer_config, dataset_config, dataset, gaussian_cfg = self._build_context()

        if dataset.input_channels != self.config.size_match.in_channels:
            raise RuntimeError(f"size_match.in_channels={self.config.size_match.in_channels} but the dataset provides {dataset.input_channels} input channels; capacity matching counted the wrong width — fix size_match.in_channels and re-run size matching without resume")

        model, model_cfg = self._build_model(dataset_config, dataset, gaussian_cfg)
        trainer          = self._build_trainer(trainer_config, dataset_config, model, model_cfg, dataset)

        return trainer, dataset


class EntryMaxBatchProbe(MaxBatchProbeBase):
    """Probes one JEPA or autoencoder benchmark model through its training pipeline.

    The probed trainer, dataset and model come from the family pipeline's own
    ``build_pretrain_trainer``, so the measured steps run the exact loss and
    data path that unit will train with.

    Attributes:
        build_trainer : Callable taking (work_dir, logger) and returning the (trainer, dataset, model) triple of the probed unit.
    """

    def __init__(self, config: BenchmarkConfig, model_name: str, work_dir: Path, build_trainer: Callable) -> None:
        """Stores the probe settings and the pipeline trainer builder.

        Args:
            config        : Benchmark configuration.
            model_name    : Model key under probe.
            work_dir      : Directory the probe writes into.
            build_trainer : Callable taking (work_dir, logger) and returning (trainer, dataset, model).
        """
        super().__init__(config, model_name, work_dir)
        self.build_trainer = build_trainer

    def _build_probe_setup(self) -> tuple:
        """Builds the probed trainer through the family pipeline, in train mode.

        Returns:
            Tuple of the trainer and the training dataset.
        """
        trainer, dataset, _model = self.build_trainer(self.work_dir / "context", self.logger)
        trainer.model.train()

        return trainer, dataset
