"""Shared scheduler/worker plumbing behind the per-run-type inference entry scripts.

Each ``main/inference/infer_*.py`` script instantiates ``InferenceEntry`` with its own run
type; this module holds the config lookup, the CLI split and the dispatch to runners.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap                                import EnvironmentPinner
from pipelines.shared.inference.run_classifier import RunType


class InferenceEntry:
    """Dual-role inference entry point for one run type.

    Invoked without ``--worker`` it schedules inference over every matching run under the
    runs directory; invoked with ``--worker`` it runs one run directory on a pinned GPU.

    Attributes:
        entry_script : Path to the concrete entry script, re-invoked for worker processes.
        run_type     : Run family this entry serves, selecting config, description and runner.
        CONFIGS      : Entry configuration class name per run family.
        BLOCKS       : Name of the per-run inference block inside that configuration, whose split and checkpoint decide which runs are already scored.
        DESCRIPTIONS : Help text per run family.
    """

    CONFIGS = {
        RunType.BACKBONE   : "BackboneInferenceEntryConfig",
        RunType.PROFILE_AE : "ProfileAeInferenceEntryConfig",
        RunType.IMAGE_AE   : "ImageAeInferenceEntryConfig",
        RunType.DUAL       : "DualInferenceEntryConfig",
    }

    BLOCKS = {
        RunType.BACKBONE   : "inference",
        RunType.PROFILE_AE : "profile_inference",
        RunType.IMAGE_AE   : "image_inference",
        RunType.DUAL       : "inference",
    }

    DESCRIPTIONS = {
        RunType.BACKBONE   : "Backbone and JEPA inference: sliding-window prediction, stitched cubes, and reports over every backbone/JEPA run found at any depth under runs_dir (one run, a sweep, an experiment or a whole tree); runs already scored on the split are skipped unless skip_completed is off.",
        RunType.PROFILE_AE : "Profile-autoencoder inference: reconstruction scoring over every standalone profile-autoencoder run found at any depth under runs_dir; runs already scored on the split are skipped unless skip_completed is off.",
        RunType.IMAGE_AE   : "Image-autoencoder inference: reconstruction scoring over every standalone image-autoencoder run found at any depth under runs_dir; runs already scored on the split are skipped unless skip_completed is off.",
        RunType.DUAL       : "Dual-trunk model inference: sliding-window prediction, stitched cubes, and reports over every dual run found at any depth under runs_dir; runs already scored on the split are skipped unless skip_completed is off.",
    }

    def __init__(self, entry_script: Path, run_type: RunType) -> None:
        """Binds the entry to a concrete script path and run type.

        Args:
            entry_script : Path of the calling ``infer_*.py`` script, passed to the scheduler so worker processes re-invoke the same file.
            run_type     : Run family whose configuration and runner this entry uses.
        """
        self.entry_script = Path(entry_script)
        self.run_type     = run_type

    def _entry_config(self):
        """Returns a fresh inference entry-config instance for this run type."""
        import configuration.inference as inference_configs

        return getattr(inference_configs, self.CONFIGS[self.run_type])()

    def _worker(self, run_dir: str, config_path: str, gpu_id: int) -> None:
        """Runs inference for a single run directory on a pinned GPU.

        Args:
            run_dir     : Directory of the trained run to run inference on.
            config_path : Path of the resolved inference config written by the scheduler.
            gpu_id      : Index of the GPU this worker process is pinned to.
        """
        EnvironmentPinner.gpu(gpu_id, expandable_segments=True)

        from pipelines.shared.inference.inference_dispatch import BackboneInferenceRunner, DualInferenceRunner, ImageAeInferenceRunner, ProfileAeInferenceRunner
        from tools.runtime.config_cli           import ConfigCli

        runners = {
            RunType.BACKBONE   : BackboneInferenceRunner,
            RunType.PROFILE_AE : ProfileAeInferenceRunner,
            RunType.IMAGE_AE   : ImageAeInferenceRunner,
            RunType.DUAL       : DualInferenceRunner,
        }

        config = ConfigCli.load_resolved(self._entry_config(), Path(config_path))

        runners[self.run_type](config).run(Path(run_dir))

    def _scheduler(self) -> None:
        """Schedules inference over every matching run and fails if any unit did not finish.

        Raises:
            SystemExit : With status 1 if any scheduled unit ended in a status other than ``DONE``.
        """
        EnvironmentPinner.threads()

        from pipelines.shared.inference.inference_scheduler import InferenceScheduler
        from tools.runtime.config_cli            import ConfigCli

        config = ConfigCli(self._entry_config(), description=self.DESCRIPTIONS[self.run_type]).apply()
        block  = getattr(config, self.BLOCKS[self.run_type])

        results = InferenceScheduler(config=config, entry_script=self.entry_script, run_type=self.run_type, block=block).run()

        if any(result.status != "DONE" for result in results):
            sys.exit(1)

    def run(self) -> None:
        """Parses the role flags and dispatches to the worker or the scheduler.

        Raises:
            SystemExit : If ``--worker`` is given without both ``--run-dir`` and ``--config``, or propagated from the scheduler when a unit failed.
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--worker",  action="store_true")
        parser.add_argument("--run-dir", type=str, default=None)
        parser.add_argument("--config",  type=str, default=None)
        parser.add_argument("--gpu",     type=int, default=0)
        args, _ = parser.parse_known_args()

        if args.worker:
            if not args.run_dir or not args.config:
                sys.exit("ERROR: --worker requires --run-dir and --config")
            self._worker(run_dir=args.run_dir, config_path=args.config, gpu_id=args.gpu)
        else:
            self._scheduler()
