"""One-off backfill of completion markers onto pre-marker runs and inference outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.data.io            import FileIO
from tools.monitoring.logger  import Logger
from tools.runtime.completion import CompletionMarker


class CompletionMarkerBackfiller:
    """Stamps completion markers onto runs finished before markers existed.

    A training run counts as finished when it carries meta/test_metrics.json, an
    inference output when it carries report.md and a recorded split. Runs holding only a
    checkpoint are reported as indeterminate and left alone.

    Attributes:
        root          : Directory scanned recursively.
        logger        : Logger for progress and warnings.
        dry_run       : Whether markers are reported instead of written.
        stamped       : Number of markers stamped so far.
        indeterminate : Run directories that cannot be proven complete.
        splitless     : Inference outputs whose split could not be recovered.
    """

    METRICS_FILENAME = "metrics.json"

    def __init__(self, root: Path, logger: Logger, dry_run: bool = False) -> None:
        """Binds the backfiller to a scan root.

        Args:
            root    : Directory scanned recursively.
            logger  : Logger for progress and warnings.
            dry_run : When set, nothing is written.
        """
        self.root          = Path(root)
        self.logger        = logger
        self.dry_run       = dry_run
        self.stamped       = 0
        self.indeterminate = []
        self.splitless     = []

    def _scan(self) -> dict[str, list[Path]]:
        """Collects the completion evidence found under the root.

        Returns:
            Mapping with sorted path lists under "metrics", "reports" and "checkpoints".
        """
        found = {"metrics": [], "reports": [], "checkpoints": []}

        for path in self.root.rglob("*"):
            if path.name == "test_metrics.json" and path.parent.name == "meta":
                found["metrics"].append(path)
            elif path.name == "report.md":
                found["reports"].append(path)
            elif path.name == "best_model.pt":
                found["checkpoints"].append(path)

        for paths in found.values():
            paths.sort()

        return found

    def _backfill_training(self, metrics_paths: list[Path]) -> None:
        """Stamps the training runs proven complete by their test metrics.

        Args:
            metrics_paths : Paths of meta/test_metrics.json files.
        """
        for metrics_path in metrics_paths:
            run_dir = metrics_path.parent.parent

            if CompletionMarker.is_complete(run_dir):
                continue

            self._stamp(run_dir, {"stage": "training"})

    def _backfill_inference(self, report_paths: list[Path]) -> None:
        """Stamps inference outputs whose split can be recovered from their metrics.

        Args:
            report_paths : Paths of report.md files.
        """
        for report_path in report_paths:
            output_dir = report_path.parent

            if output_dir.parent.name != "inference":
                continue
            if CompletionMarker.is_complete(output_dir) and "split" in CompletionMarker.payload(output_dir):
                continue

            split = self._split_of(output_dir)

            if split is None:
                self.splitless.append(output_dir)
                continue

            self._stamp(output_dir, {"stage": "inference", "split": split})

    def _split_of(self, output_dir: Path) -> str | None:
        """Returns the split recorded in an inference output's metrics, or None."""
        metrics_path = output_dir / self.METRICS_FILENAME

        if not metrics_path.is_file():
            return None

        metrics = FileIO.load_json(metrics_path)

        return metrics["split"] if "split" in metrics else None

    def _stamp(self, directory: Path, payload: dict) -> None:
        """Stamps one directory, marking the payload as backfilled.

        Args:
            directory : Directory receiving complete.json.
            payload   : Marker payload, extended with a backfilled flag.
        """
        self.stamped += 1
        self.logger.info(f"stamp {directory}")

        if not self.dry_run:
            CompletionMarker.stamp(directory, {**payload, "backfilled": True})

    def _collect_indeterminate(self, checkpoints: list[Path]) -> None:
        """Records runs that hold a checkpoint but no proof of completion.

        Args:
            checkpoints : Paths of best_model.pt files.
        """
        for checkpoint in checkpoints:
            run_dir = checkpoint.parent

            if run_dir.name == "overfit_check":
                continue
            if CompletionMarker.is_complete(run_dir):
                continue
            if (run_dir / "meta" / "test_metrics.json").is_file():
                continue

            self.indeterminate.append(run_dir)

    def run(self) -> int:
        """Scans the root, stamps what it can, and reports the rest.

        Returns:
            Process exit code, 1 when the root is not a directory and 0 otherwise.
        """
        if not self.root.is_dir():
            self.logger.error(f"Not a directory: {self.root}")
            return 1

        found = self._scan()

        self._backfill_training(found["metrics"])
        self._backfill_inference(found["reports"])
        self._collect_indeterminate(found["checkpoints"])

        self.logger.info(f"{self.root}: {self.stamped} marker(s) {'would be ' if self.dry_run else ''}stamped")

        for output_dir in self.splitless:
            self.logger.warning(f"inference output left unstamped (no '{self.METRICS_FILENAME}' split to record, and a marker without it breaks every resume reader): {output_dir}")

        for run_dir in self.indeterminate:
            self.logger.warning(f"indeterminate (checkpoint without test metrics, cannot prove completion): {run_dir}")

        if self.indeterminate:
            self.logger.warning(f"{len(self.indeterminate)} run(s) left unstamped; a resumed pipeline will delete and retrain them. Stamp manually only if you know they finished.")

        return 0


def main() -> None:
    """Parses the command line, backfills every given path and exits with its status."""
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
