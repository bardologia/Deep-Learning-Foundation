"""Sequencing of the pre-training probes that override batch size, loader settings and the learning rate."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing      import Callable, Optional

import torch

from tools.monitoring.logger                 import Logger
from tools.training.pretraining.batch_finder import BatchSizeFinder
from tools.training.pretraining.loader_tuner import LoaderTuner
from tools.training.pretraining.lr_finder    import LrRangeTest


@dataclass
class PretrainContext:
    """Throwaway dataset, model and callables one probe run operates on.

    A fresh context is built per probe so the probe's allocations never survive
    into the training run.

    Attributes:
        dataset        : Dataset the probe batches are drawn from.
        model          : Model the probe runs its passes on.
        to_model_input : Callable moving a batch to the device.
        forward_loss   : Callable returning the loss of a batch.
        trial_step     : Callable mapping a batch size to its peak memory in GB.
        device         : Device the probe runs on.
        use_amp        : Whether the probe runs under autocast.
        context_gb     : Fixed device memory in GB held outside the probe.
        on_oom         : Callback invoked when a trial runs out of memory, or None.
        trainer        : Trainer the learning-rate range test steps through.
    """

    dataset        : object
    model          : object
    to_model_input : Callable
    forward_loss   : Callable
    trial_step     : Callable[[int], float]
    device         : torch.device
    trainer        : object
    use_amp        : bool                         = False
    context_gb     : float                        = 0.0
    on_oom         : Optional[Callable[[], None]] = None


class PretrainOrchestrator:
    """Runs the enabled pre-training probes and writes their results into the training config.

    Attributes:
        TUNED_LEAVES : Entry-config probe flag to the flattened leaves that probe rewrites once it ran, the flag itself included.
        pretrain     : Pre-training configuration selecting and parameterising the probes.
        training     : Training configuration whose batch size and loader fields the probes override for this run.
        build        : Factory returning a fresh probe context per probe.
        logger       : Logger for the probe sections and resolved values.
        label        : Model label reported by the batch-size finder, or None.
    """

    TUNED_LEAVES = {
        "pretrain.find_batch_size" : ("pretrain.find_batch_size", "training.batch_size"),
        "pretrain.tune_loader"     : ("pretrain.tune_loader", "training.num_workers", "training.prefetch_factor"),
        "pretrain.find_lr"         : ("pretrain.find_lr", "training.lr_multiplier"),
    }

    def __init__(self, pretrain_config, training_config, build_context: Callable[[], PretrainContext], logger: Logger, label: Optional[str] = None) -> None:
        """Stores the configurations, the context factory and the logger.

        Args:
            pretrain_config : Pre-training configuration.
            training_config : Training configuration modified in place by the probes.
            build_context   : Factory returning a fresh probe context.
            logger          : Logger for the probe output.
            label           : Model label used in the batch-size result.
        """
        self.pretrain = pretrain_config
        self.training = training_config
        self.build    = build_context
        self.logger   = logger
        self.label    = label

    @classmethod
    def tuned_leaves(cls, mapping: dict) -> set:
        """Returns the flattened entry-config leaves the probes enabled in a launch rewrite.

        A recording of a finished run legitimately differs from the launch
        configuration on these leaves: the probe wrote the value and switched
        its own flag off.

        Args:
            mapping : Flattened entry configuration, dotted leaf path to value, as ConfigCli.to_mapping returns it.

        Returns:
            The rewritten leaves of every probe whose flag is set in the mapping.
        """
        leaves = set()

        for flag, rewritten in cls.TUNED_LEAVES.items():
            if flag in mapping and mapping[flag]:
                leaves.update(rewritten)

        return leaves

    def _enabled(self) -> bool:
        """Returns whether any probe is switched on."""
        return bool(self.pretrain.find_batch_size or self.pretrain.tune_loader or self.pretrain.find_lr)

    def _release_cache(self) -> None:
        """Collects garbage and empties the CUDA caching allocator after a probe."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_batch_size(self, context: PretrainContext) -> None:
        """Resolves the batch size within the VRAM budget and writes it into the training config.

        ``find_batch_size`` is switched off once the value is folded in, so the
        persisted entry config carries the measurement rather than an
        instruction to measure again, under whatever else occupies the GPU, on
        a relaunch or a resume.

        Args:
            context : Probe context supplying the dataset, device and trial step.

        Raises:
            RuntimeError : If not even a batch of one fits the budget.
        """
        ceiling = min(self.pretrain.max_batch, len(context.dataset))
        if ceiling < self.pretrain.max_batch:
            self.logger.subsection(f"Ceiling lowered from {self.pretrain.max_batch} to {ceiling}: the dataset holds {len(context.dataset)} samples")

        finder = BatchSizeFinder(
            trial_step = context.trial_step,
            budget_gb  = self.pretrain.vram_budget_gb,
            ceiling    = ceiling,
            device     = context.device,
            logger     = self.logger,
            model_name = self.label,
            context_gb = context.context_gb,
            on_oom     = context.on_oom,
        )

        result = finder.run()

        if result["status"] != "PASS":
            raise RuntimeError(f"max batch-size finder failed: {result['error']}")

        self.training.batch_size      = int(result["batch_size"])
        self.pretrain.find_batch_size = False

        self.logger.subsection(f"Resolved batch size: {self.training.batch_size} overrides training.batch_size for this run (peak {result['peak_gb']:.2f} GB, scale_lr_with_batch={self.training.scale_lr_with_batch})")
        self.logger.subsection("find_batch_size switched off for the rest of this run: the resolved entry config now records the probed batch size, so a relaunch or resume reuses it instead of probing again")

    def _tune_loader(self, context: PretrainContext) -> None:
        """Resolves worker count and prefetch factor and writes them into the training config.

        The training config is left untouched when the sweep finds no successful
        configuration. ``tune_loader`` is switched off either way once the sweep
        ran, so the persisted entry config describes the loader the run uses.

        Args:
            context : Probe context supplying the dataset, model and feed callables.
        """
        tuner = LoaderTuner(
            dataset          = context.dataset,
            model            = context.model,
            to_model_input   = context.to_model_input,
            forward_loss     = context.forward_loss,
            device           = context.device,
            logger           = self.logger,
            use_amp          = context.use_amp,
            seed             = self.pretrain.seed,
            warmup_batches   = self.pretrain.warmup_batches,
            timed_batches    = self.pretrain.timed_batches,
            worker_counts    = self.pretrain.worker_counts,
            prefetch_factors = self.pretrain.prefetch_factors,
            data_wait_target = self.pretrain.data_wait_target,
        )

        choice = tuner.run(self.training.batch_size)

        self.pretrain.tune_loader = False

        if choice is None:
            self.logger.subsection(f"tune_loader switched off for the rest of this run: the configured workers={self.training.num_workers} prefetch={self.training.prefetch_factor} stay, and a relaunch or resume keeps them instead of sweeping again")
            return

        self.training.num_workers     = int(choice["num_workers"])
        self.training.prefetch_factor = int(choice["prefetch_factor"])
        self.logger.subsection(f"Resolved loader: workers={self.training.num_workers} prefetch={self.training.prefetch_factor} override training.num_workers/prefetch_factor for this run (the probe pins memory throughout, as the training loader does)")
        self.logger.subsection("tune_loader switched off for the rest of this run: the resolved entry config now records the tuned loader, so a relaunch or resume reuses it instead of sweeping again")

    def _find_lr(self, context: PretrainContext) -> None:
        """Runs the LR range test and folds its multiplier into the training config.

        The training config is left untouched when the test refuses to answer.
        The result composes multiplicatively with any multiplier already
        configured, since the probe trainer's learning rates were built from it,
        and ``find_lr`` is switched off once the test ran, answer or refusal, so
        the persisted entry config carries the measurement rather than an
        instruction to measure again and square it on the next relaunch, or to
        fold a multiplier into a resumed run mid-way.

        Args:
            context : Probe context supplying the trainer, dataset and device.
        """
        test = LrRangeTest(
            trainer           = context.trainer,
            dataset           = context.dataset,
            device            = context.device,
            logger            = self.logger,
            scale_min         = self.pretrain.lr_scale_min,
            scale_max         = self.pretrain.lr_scale_max,
            rungs             = self.pretrain.lr_rungs,
            rung_steps        = self.pretrain.lr_rung_steps,
            probe_batches     = self.pretrain.lr_probe_batches,
            divergence_factor = self.pretrain.lr_divergence_factor,
            safety_factor     = self.pretrain.lr_safety_factor,
            min_improvement   = self.pretrain.lr_min_improvement,
            noise_sigmas      = self.pretrain.lr_noise_sigmas,
            seed              = self.pretrain.seed,
        )

        result = test.run(self.training.batch_size)

        self.pretrain.find_lr = False

        if result is None:
            self.logger.subsection(f"find_lr switched off for the rest of this run: the test refused to answer, the configured training.lr_multiplier={self.training.lr_multiplier:.4g} stays, and a relaunch or resume keeps it instead of probing again")
            return

        divergence = f"x{result['divergence_multiplier']:.4g}" if result["divergence_multiplier"] is not None else "not reached"

        self.training.lr_multiplier = float(self.training.lr_multiplier) * float(result["multiplier"])

        self.logger.subsection(f"Resolved learning rate: multiplier x{result['multiplier']:.4g} (loss minimum at x{result['min_loss_multiplier']:.4g}, divergence at {divergence}, safety /{self.pretrain.lr_safety_factor:g}) sets training.lr_multiplier={self.training.lr_multiplier:.4g} for this run, applied on top of the batch-size rule")
        self.logger.subsection("find_lr switched off for the rest of this run: the resolved entry config now records the measured multiplier, so a relaunch reuses it instead of probing again and compounding a second one.")

    def run(self) -> None:
        """Runs the enabled probes in order, each on a fresh context, releasing memory in between."""
        if not self._enabled():
            return

        if self.pretrain.find_batch_size:
            self.logger.section("[Pretrain] Max batch-size finder")
            self.logger.subsection("Probe run: the dataset and model below belong to the batch-size probe, not the training run.")
            self.logger.kv_table({
                "VRAM budget"   : f"{self.pretrain.vram_budget_gb:g} GB",
                "Batch ceiling" : f"{self.pretrain.max_batch} (trials are powers of two up to the ceiling)",
                "Measure steps" : self.pretrain.measure_steps,
            })
            self._find_batch_size(self.build())
            self._release_cache()

        if self.pretrain.tune_loader:
            self.logger.section("[Pretrain] DataLoader tuner")
            self.logger.subsection("Probe run: the dataset and model below belong to the loader-timing probe, not the training run.")
            self.logger.kv_table({
                "Worker counts"    : list(self.pretrain.worker_counts),
                "Prefetch factors" : list(self.pretrain.prefetch_factors),
                "Warmup batches"   : self.pretrain.warmup_batches,
                "Timed batches"    : self.pretrain.timed_batches,
                "Data-wait target" : self.pretrain.data_wait_target,
                "Probe seed"       : self.pretrain.seed,
            })
            self._tune_loader(self.build())
            self._release_cache()

        if self.pretrain.find_lr:
            self.logger.section("[Pretrain] LR range test")
            self.logger.subsection("Probe run: the dataset and model below belong to the LR range test, not the training run.")
            self.logger.kv_table({
                "Multiplier range"    : f"x{self.pretrain.lr_scale_min:g} to x{self.pretrain.lr_scale_max:g} (geometric, on the trainer's warmup-free base learning rates)",
                "Ramp rungs"          : self.pretrain.lr_rungs,
                "Steps per rung"      : f"{self.pretrain.lr_rung_steps} (the trainer is rewound to the same state before each rung)",
                "Measurement batches" : self.pretrain.lr_probe_batches,
                "Divergence factor"   : self.pretrain.lr_divergence_factor,
                "Safety factor"       : self.pretrain.lr_safety_factor,
                "Minimum improvement" : f"{self.pretrain.lr_min_improvement:g} relative, or {self.pretrain.lr_noise_sigmas:g} standard errors",
                "Probe seed"          : self.pretrain.seed,
            })
            self._find_lr(self.build())
            self._release_cache()
