"""Entry-point machinery for profile-autoencoder training: single runs and trial sweeps.

Holds the scheduler that maps every profile-autoencoder trial mode to its
planner and the launcher that decides between a seed sweep and a trial sweep,
locally and on the cluster.
"""

from __future__ import annotations

import argparse
import sys

from configuration.training.general.optimization       import ClipMode, WarmupMode
from configuration.training.profile_autoencoder        import ProfileAeEntryConfig
from models.profile_autoencoder                        import PROFILE_AE_CONFIG_REGISTRY
from pipelines.profile_autoencoder.training.experiments import ArchTrialPlanner, CapacityTrialPlanner, EmbeddingDimTrialPlanner, ProfileAugmentationTrialPlanner, ProfileLossTrialPlanner, SamplingTrialPlanner, SmoothnessTrialPlanner
from pipelines.profile_autoencoder.training.pipeline   import SingleTrainRunner
from pipelines.shared.training.ddp_fanout              import DdpDispatch
from pipelines.shared.training.training_launcher       import SeedSweepClusterLauncher, SeedSweepLauncher
from pipelines.shared.training.trial_planners          import AmpTrialPlanner, EmaTrialPlanner, StabilityTrialPlanner
from pipelines.shared.training.trial_sweep             import TrialSweepScheduler
from tools.runtime.config_cli                          import ConfigCli


class ProfileAeTrialScheduler(TrialSweepScheduler):
    """Profile-autoencoder trial scheduler mapping every sweep mode to its planner."""

    SCHEDULER_FIELDS = ("trials_enabled", "trials_mode", "arch_trials", "embedding_trials", "capacity_trials", "loss_trials", "smoothness_trials", "augmentation_trials", "sampling_trials", "ema_trials", "stability_trials", "gpu", "gpus", "gpus_file", "poll_interval_s", "resume_from")

    MODE_SUBDIRS = {
        "arch"         : "arch",
        "embedding"    : "embedding",
        "capacity"     : "capacity",
        "loss"         : "loss",
        "smoothness"   : "smoothness",
        "augmentation" : "augmentation",
        "sampling"     : "sampling",
        "ema"          : "ema",
        "amp"          : "amp",
        "stability"    : "stability",
    }

    def _build_planner(self):
        """Builds the planner matching the configured trial mode.

        Returns:
            The planner instance for `config.trials_mode`.

        Raises:
            ValueError : If the mode is not one of the known trial modes.
        """
        mode = self.config.trials_mode
        if mode == "arch":
            return ArchTrialPlanner(self.config.arch_trials, tuple(PROFILE_AE_CONFIG_REGISTRY), self.config.model_overrides)
        if mode == "embedding":
            return EmbeddingDimTrialPlanner(self.config.embedding_trials, self.config.model_overrides)
        if mode == "capacity":
            return CapacityTrialPlanner(self.config.capacity_trials, self.config.ae_model_name, self.config.model_overrides, PROFILE_AE_CONFIG_REGISTRY)
        if mode == "loss":
            return ProfileLossTrialPlanner(self.config.loss_trials)
        if mode == "smoothness":
            return SmoothnessTrialPlanner(self.config.smoothness_trials, self.config.ae_loss)
        if mode == "augmentation":
            return ProfileAugmentationTrialPlanner(self.config.augmentation_trials)
        if mode == "sampling":
            return SamplingTrialPlanner(self.config.sampling_trials)
        if mode == "ema":
            return EmaTrialPlanner(self.config.ema_trials)
        if mode == "amp":
            return AmpTrialPlanner()
        if mode == "stability":
            return StabilityTrialPlanner(self.config.stability_trials, tuple(mode.value for mode in WarmupMode), tuple(mode.value for mode in ClipMode))

        raise ValueError(f"Unknown trials_mode '{mode}', expected one of {sorted(self.MODE_SUBDIRS)}")

    def _model_key(self) -> str:
        """Returns the configured autoencoder family name."""
        return self.config.ae_model_name


class ProfileAeTrainingLauncher(SeedSweepClusterLauncher):
    """Dispatches profile-autoencoder training between seed sweeps and trial sweeps.

    Locally, `run` trains a single seed, fans seeds out over the GPU pool, or
    schedules the configured trial sweep; on the cluster, `build_scheduler`
    hands the submitter the matching seed-sweep or trial-sweep scheduler.
    """

    entry_config_class = ProfileAeEntryConfig
    description        = "Train one profile autoencoder, fan seeds out over GPUs, or run an architecture/loss/smoothness trial sweep"
    sweep_label        = "profile_autoencoder"

    def _configure(self, argv: list[str]):
        """Parses argv into a resolved entry config and the CLI holding the overrides.

        Args:
            argv : Command-line arguments without the program name.

        Returns:
            A tuple of the resolved config and the CLI object.
        """
        cli    = ConfigCli(self.entry_config_class(), description=self.description)
        config = cli.apply(argv)

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
            return ProfileAeTrialScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script)

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

            SeedSweepLauncher(self.entry_config_class(), SingleTrainRunner, self.description, entry_script=self.entry_script).run(argv)
            return

        ProfileAeTrialScheduler(config=config, cli_overrides=cli.overrides, entry_script=self.entry_script).run()
