
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap                                import EnvironmentPinner
from pipelines.shared.inference.run_classifier import RunType


class InferenceEntry:

    CONFIGS = {
        RunType.BACKBONE   : "BackboneInferenceEntryConfig",
        RunType.PROFILE_AE : "ProfileAeInferenceEntryConfig",
        RunType.IMAGE_AE   : "ImageAeInferenceEntryConfig",
    }

    BLOCKS = {
        RunType.BACKBONE   : "inference",
        RunType.PROFILE_AE : "profile_inference",
        RunType.IMAGE_AE   : "image_inference",
    }

    DESCRIPTIONS = {
        RunType.BACKBONE   : "Backbone and JEPA inference: sliding-window prediction, stitched cubes, and reports over every backbone/JEPA run found at any depth under runs_dir (one run, a sweep, an experiment or a whole tree); runs already scored on the split are skipped unless skip_completed is off.",
        RunType.PROFILE_AE : "Profile-autoencoder inference: reconstruction scoring over every standalone profile-autoencoder run found at any depth under runs_dir; runs already scored on the split are skipped unless skip_completed is off.",
        RunType.IMAGE_AE   : "Image-autoencoder inference: reconstruction scoring over every standalone image-autoencoder run found at any depth under runs_dir; runs already scored on the split are skipped unless skip_completed is off.",
    }

    def __init__(self, entry_script: Path, run_type: RunType) -> None:
        self.entry_script = Path(entry_script)
        self.run_type     = run_type

    def _entry_config(self):
        import configuration.inference as inference_configs

        return getattr(inference_configs, self.CONFIGS[self.run_type])()

    def _worker(self, run_dir: str, config_path: str, gpu_id: int) -> None:
        EnvironmentPinner.gpu(gpu_id, expandable_segments=True)

        from pipelines.shared.inference.inference_dispatch import BackboneInferenceRunner, ImageAeInferenceRunner, ProfileAeInferenceRunner
        from tools.runtime.config_cli           import ConfigCli

        runners = {
            RunType.BACKBONE   : BackboneInferenceRunner,
            RunType.PROFILE_AE : ProfileAeInferenceRunner,
            RunType.IMAGE_AE   : ImageAeInferenceRunner,
        }

        config = ConfigCli.load_resolved(self._entry_config(), Path(config_path))

        runners[self.run_type](config).run(Path(run_dir))

    def _scheduler(self) -> None:
        EnvironmentPinner.threads()

        from pipelines.shared.inference.inference_scheduler import InferenceScheduler
        from tools.runtime.config_cli            import ConfigCli

        config = ConfigCli(self._entry_config(), description=self.DESCRIPTIONS[self.run_type]).apply()
        block  = getattr(config, self.BLOCKS[self.run_type])

        results = InferenceScheduler(config=config, entry_script=self.entry_script, run_type=self.run_type, block=block).run()

        if any(result.status != "DONE" for result in results):
            sys.exit(1)

    def run(self) -> None:
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
