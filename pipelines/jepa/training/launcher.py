"""Entry-point machinery for JEPA training: single runs and trial sweeps.

Holds the scheduler that maps every JEPA trial mode to its planner and the
launcher that decides between a seed sweep and a trial sweep, locally and on
the cluster.
"""

from __future__ import annotations

import argparse
import sys

from configuration.normalization                 import Presets
from configuration.training.general.optimization import ClipMode, WarmupMode
from configuration.training.jepa                 import JepaEntryConfig
from models                                      import BACKBONE_CONFIG_REGISTRY
from pipelines.backbone.training.experiments     import AugmentationTrialPlanner, ContextTrialPlanner, InputTrialPlanner, NormalizationTrialPlanner
from pipelines.jepa.training.coupling            import TargetProvider
from pipelines.jepa.training.experiments         import AeRunTrialPlanner, CouplingTrialPlanner, CurveTrialPlanner, EmbeddingLossTrialPlanner, JepaCurriculumTrialPlanner, MaskingTrialPlanner
from pipelines.jepa.training.pipeline            import SingleTrainRunner
from pipelines.shared.model.model_builder        import ModelBuilder
from pipelines.shared.training.ddp_fanout        import DdpDispatch
from pipelines.shared.training.training_launcher import SeedSweepClusterLauncher, SeedSweepLauncher
from pipelines.shared.training.trial_planners    import AmpTrialPlanner, EmaTrialPlanner, StabilityTrialPlanner
from pipelines.shared.training.trial_sweep       import TrialSweepScheduler
from tools.runtime.config_cli                    import ConfigCli


class JepaTrialScheduler(TrialSweepScheduler):
    """JEPA trial scheduler mapping every sweep mode to its planner."""

    SCHEDULER_FIELDS = ("trials_enabled", "trials_mode", "embedding_trials", "curve_trials", "masking_trials", "curriculum_trials", "coupling_trials", "ae_run_trials", "context_trials", "input_trials", "augmentation_trials", "normalization_trials", "ema_trials", "stability_trials", "infer_at_end", "gpu", "gpus", "gpus_file", "poll_interval_s", "resume_from")

    MODE_SUBDIRS = {
        "embedding"     : "embedding",
        "curve"         : "curve",
        "masking"       : "masking",
        "curriculum"    : "curriculum",
        "coupling"      : "coupling",
        "ae_run"        : "ae_run",
        "context"       : "context",
        "input"         : "input",
        "augmentation"  : "augmentation",
        "normalization" : "normalization",
        "ema"           : "ema",
        "amp"           : "amp",
        "stability"     : "stability",
    }

    @property
    def staged_inference(self) -> bool:
        """Whether inference runs as a separate stage after all training units."""
        return self.config.infer_at_end

    def _inference_rows(self) -> dict:
        """Returns the per-unit and batched inference flags for the configuration table."""
        return {
            "Infer after"  : self.config.infer_after,
            "Infer at end" : self.config.infer_at_end,
        }

    def _build_planner(self):
        """Builds the planner matching the configured trial mode.

        Returns:
            The planner instance for `config.trials_mode`.

        Raises:
            ValueError : If the mode is not one of the known trial modes, or the base configuration leaves the mode with arms the trainer would reject.
        """
        mode = self.config.trials_mode
        if mode == "embedding":
            return EmbeddingLossTrialPlanner(self.config.embedding_trials)
        if mode == "curve":
            return CurveTrialPlanner(self.config.curve_trials, self.config.target_provider)
        if mode == "masking":
            return MaskingTrialPlanner(self.config.masking_trials, self.config.profile_autoencoder_run)
        if mode == "curriculum":
            return JepaCurriculumTrialPlanner(self.config.curriculum_trials, self.config.training.epochs)
        if mode == "coupling":
            return CouplingTrialPlanner(self.config.coupling_trials, TargetProvider.VALID, self.config.embedding_loss)
        if mode == "ae_run":
            return AeRunTrialPlanner(self.config.ae_run_trials)
        if mode == "context":
            return ContextTrialPlanner(self.config.context_trials, tuple(BACKBONE_CONFIG_REGISTRY))
        if mode == "input":
            return InputTrialPlanner.from_dataset(self.config.input_trials, self.config.geometry, self.config.paths.dataset_path)
        if mode == "augmentation":
            return AugmentationTrialPlanner(self.config.augmentation_trials)
        if mode == "normalization":
            return NormalizationTrialPlanner(self.config.normalization_trials, tuple(Presets.names()))
        if mode == "ema":
            return EmaTrialPlanner(self.config.ema_trials)
        if mode == "amp":
            return AmpTrialPlanner()
        if mode == "stability":
            return StabilityTrialPlanner(self.config.stability_trials, tuple(mode.value for mode in WarmupMode), tuple(mode.value for mode in ClipMode))

        raise ValueError(f"Unknown trials_mode '{mode}', expected one of {sorted(self.MODE_SUBDIRS)}")

    def _model_key(self) -> str:
        """Returns the backbone-and-head key of the base configuration."""
        return ModelBuilder.model_key(self.config.backbone_name, self.config.backbone_head)


class JepaTrainingLauncher(SeedSweepClusterLauncher):
    """Dispatches JEPA training between seed sweeps and trial sweeps.

    Locally, `run` trains a single seed, fans seeds out over the GPU pool, or
    schedules the configured trial sweep; on the cluster, `build_scheduler`
    hands the submitter the matching seed-sweep or trial-sweep scheduler.
    """

    entry_config_class    = JepaEntryConfig
    description           = "Train one JEPA predictor, fan seeds out over GPUs, or run an embedding/masking/coupling trial sweep, with optional batched inference"
    sweep_label           = "jepa"
    supports_infer_at_end = True

    def _configure(self, argv: list[str]):
        """Parses argv into a resolved entry config and the CLI holding the overrides.

        Args:
            argv : Command-line arguments without the program name.

        Returns:
            A tuple of the resolved config and the CLI object.

        Raises:
            SystemExit : If both `infer_after` and `infer_at_end` are set.
        """
        cli    = ConfigCli(self.entry_config_class(), description=self.description)
        config = cli.apply(argv)

        if config.infer_after and config.infer_at_end:
            raise SystemExit("infer_after and infer_at_end are mutually exclusive; infer_after runs inference inside each training job, infer_at_end batches inference after every training finishes")

        return config, cli

    def build_scheduler(self, argv: list[str]):
        """Returns the sweep scheduler configured from `argv`, trials-aware, without running it.

        Args:
            argv : Command-line arguments without the program name.

        Returns:
            The trial scheduler when `trials_enabled` is set, otherwise the
            seed-sweep cluster scheduler.
        """
        config, cli = self._configure(list(argv))

        if config.trials_enabled:
            return JepaTrialScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)

        return super().build_scheduler(argv)

    def run(self, argv: list[str] | None = None) -> None:
        """Runs a single unit, a seed fan-out or a trial sweep, as the arguments dictate.

        Args:
            argv : Command-line arguments without the program name; defaults to the process arguments.

        Raises:
            ValueError : If `ddp` is combined with a trial unit or a trial sweep, each of which already claims one GPU per unit.
        """
        argv = list(sys.argv[1:] if argv is None else argv)

        config, cli = self._configure(argv)

        trial_parser = argparse.ArgumentParser(add_help=False)
        trial_parser.add_argument("--trial", action="store_true")
        trial, _ = trial_parser.parse_known_args(argv)

        if trial.trial or not config.trials_enabled:
            if trial.trial:
                DdpDispatch.reject_trials(config, trial_unit=True)

            SeedSweepLauncher(self.entry_config_class(), SingleTrainRunner, self.description, entry_script=self.entry_script, supports_infer_at_end=True).run(argv)
            return

        JepaTrialScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script).run()
