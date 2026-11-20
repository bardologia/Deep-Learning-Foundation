from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import EnvironmentPinner


class InferenceEntry:

    DESCRIPTION = "Backbone inference: sliding-window prediction, stitched target maps, and reports over every backbone run found at any depth under runs_dir (one run, a sweep, an experiment or a whole tree); runs already scored on the split are skipped unless skip_completed is off."

    def __init__(self, entry_script: Path) -> None:
        self.entry_script = Path(entry_script)

    def _entry_config(self):
        from configuration.inference import BackboneInferenceEntryConfig

        return BackboneInferenceEntryConfig()

    def _worker(self, run_dir: str, config_path: str, gpu_id: int) -> None:
        EnvironmentPinner.gpu(gpu_id, expandable_segments=True)

        from pipelines.shared.inference.inference_dispatch import BackboneInferenceRunner
        from tools.runtime.config_cli                      import ConfigCli

        config = ConfigCli.load_resolved(self._entry_config(), Path(config_path))

        BackboneInferenceRunner(config).run(Path(run_dir))

    def _scheduler(self) -> None:
        EnvironmentPinner.threads()

        from pipelines.shared.inference.inference_scheduler import InferenceScheduler
        from tools.runtime.config_cli                       import ConfigCli

        config  = ConfigCli(self._entry_config(), description=self.DESCRIPTION).apply()
        results = InferenceScheduler(config=config, entry_script=self.entry_script).run()

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
            self._worker(run_dir=args.run_dir, config_path=args.config, gpu_id=args.gpu)
        else:
            self._scheduler()
