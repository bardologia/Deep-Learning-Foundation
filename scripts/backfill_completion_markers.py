from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.monitoring.logger  import Logger
from tools.runtime.completion import CompletionMarker


class CompletionMarkerBackfiller:

    def __init__(self, root: Path, logger: Logger, dry_run: bool = False) -> None:
        self.root          = Path(root)
        self.logger        = logger
        self.dry_run       = dry_run
        self.stamped       = 0
        self.indeterminate = []

    def _backfill_training(self) -> None:
        for metrics_path in sorted(self.root.rglob("meta/test_metrics.json")):
            run_dir = metrics_path.parent.parent

            if CompletionMarker.is_complete(run_dir):
                continue

            self._stamp(run_dir, {"stage": "training"})

    def _backfill_inference(self) -> None:
        for report_path in sorted(self.root.rglob("report.md")):
            output_dir = report_path.parent

            if output_dir.parent.name != "inference":
                continue
            if CompletionMarker.is_complete(output_dir):
                continue

            self._stamp(output_dir, {"stage": "inference"})

    def _stamp(self, directory: Path, payload: dict) -> None:
        self.stamped += 1
        self.logger.info(f"stamp {directory}")

        if not self.dry_run:
            CompletionMarker.stamp(directory, {**payload, "backfilled": True})

    def _collect_indeterminate(self) -> None:
        for checkpoint in sorted(self.root.rglob("best_model.pt")):
            run_dir = checkpoint.parent

            if run_dir.name == "overfit_check":
                continue
            if CompletionMarker.is_complete(run_dir):
                continue
            if (run_dir / "meta" / "test_metrics.json").is_file():
                continue

            self.indeterminate.append(run_dir)

    def run(self) -> int:
        if not self.root.is_dir():
            self.logger.error(f"Not a directory: {self.root}")
            return 1

        self._backfill_training()
        self._backfill_inference()
        self._collect_indeterminate()

        self.logger.info(f"{self.root}: {self.stamped} marker(s) {'would be ' if self.dry_run else ''}stamped")

        for run_dir in self.indeterminate:
            self.logger.warning(f"indeterminate (checkpoint without test metrics, cannot prove completion): {run_dir}")

        if self.indeterminate:
            self.logger.warning(f"{len(self.indeterminate)} run(s) left unstamped; a resumed pipeline will delete and retrain them. Stamp manually only if you know they finished.")

        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="TEMPORARY: stamp complete.json markers onto training runs and inference outputs generated before completion markers existed, so resumed pipelines reuse them instead of retraining. Training runs are proven complete by meta/test_metrics.json, inference outputs by their report.md.")
    parser.add_argument("paths", type=str, nargs="+", help="Run directories (or parents) to scan recursively.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be stamped without writing.")
    args = parser.parse_args()

    logger    = Logger(log_dir="logs", name="backfill_completion_markers")
    exit_code = 0

    for path in args.paths:
        exit_code = max(exit_code, CompletionMarkerBackfiller(Path(path), logger, dry_run=args.dry_run).run())

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
