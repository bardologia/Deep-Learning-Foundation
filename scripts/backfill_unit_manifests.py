"""Backfill of unit manifests onto the fold and benchmark units of pre-manifest experiments.

A staged experiment launched before ``meta/unit.json`` existed carries units that
a resume cannot identify: the ledger lists them as unidentified and the run is
trained again unless its directory happens to be the one today's settings
compose. This writes the manifest each unit should have, matching planned units
to existing directories by the name the recorded configuration composes, which
is exactly the rule those runs were created under.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.shared.orchestration.resumed_run   import StagedRunLocator
from pipelines.shared.orchestration.unit_manifest import UnitManifest
from tools.monitoring.logger                      import Logger
from tools.runtime.config_cli                     import ConfigCli


class FoldManifestPlan:
    """The manifests a recorded cross-validation configuration says its fold units should carry.

    The unit expansion and the fold plan are taken from the pipeline's own
    components, so the names matched on disk and the regions written are the
    ones the run was created with.

    Attributes:
        KIND          : Manifest kind the units are written with.
        STAGE_DIRNAME : Directory of the run holding the unit directories.
        run_dir       : Cross-validation run directory being backfilled.
        config        : Configuration the run recorded.
    """

    KIND          = "cross_validation"
    STAGE_DIRNAME = "folds"

    @classmethod
    def claims(cls, run_dir: Path) -> bool:
        """Returns whether a run directory is a cross-validation run, which its folds directory shows."""
        return (Path(run_dir) / cls.STAGE_DIRNAME).is_dir()

    def __init__(self, run_dir: Path) -> None:
        """Loads the configuration the cross-validation run recorded.

        Args:
            run_dir : Cross-validation run directory being backfilled.
        """
        from configuration.cross_validation import CrossValidationConfig

        self.run_dir = Path(run_dir)
        self.config  = ConfigCli.load_resolved(CrossValidationConfig(), StagedRunLocator.config_path(self.run_dir))

    def units(self) -> list[tuple[str, dict]]:
        """Returns each planned unit's run name and the payload its manifest is written from.

        Returns:
            Pairs of the unit's directory name relative to the folds directory
            and the keyword arguments of the fold manifest writer.
        """
        from pipelines.cross_validation.folds import FoldConfigFactory, FoldPlanner, FoldRunNaming

        planner = FoldConfigFactory(self.config).planner()

        planned = []
        for fold_index, seed, name in FoldRunNaming(self.config).units():
            regions = {split: planner.inference_region(fold_index, split).as_tuple() for split in FoldPlanner.INFERENCE_SPLITS}

            planned.append((name, {
                "run_tag"           : self.run_dir.name,
                "unit"              : name,
                "fold_index"        : fold_index,
                "seed"              : seed,
                "training_type"     : self.config.training_type,
                "inference_regions" : regions,
            }))

        return planned

    def write(self, run_directory: Path, payload: dict) -> Path:
        """Writes one fold unit's manifest into its run directory."""
        from pipelines.cross_validation.folds import FoldUnitManifest

        return FoldUnitManifest.write(run_directory, **payload)


class BenchmarkManifestPlan:
    """The manifests a recorded benchmark configuration says its training units should carry.

    Attributes:
        KIND          : Manifest kind the units are written with.
        STAGE_DIRNAME : Directory of the run holding the unit directories.
        run_dir       : Benchmark run directory being backfilled.
        config        : Configuration the run recorded.
    """

    KIND          = "benchmark"
    STAGE_DIRNAME = "training"

    @classmethod
    def claims(cls, run_dir: Path) -> bool:
        """Returns whether a run directory is a benchmark run, which its training directory shows."""
        return (Path(run_dir) / cls.STAGE_DIRNAME).is_dir()

    def __init__(self, run_dir: Path) -> None:
        """Loads the configuration the benchmark run recorded.

        Args:
            run_dir : Benchmark run directory being backfilled.
        """
        from configuration.benchmark import BenchmarkConfig

        self.run_dir = Path(run_dir)
        self.config  = ConfigCli.load_resolved(BenchmarkConfig(), StagedRunLocator.config_path(self.run_dir))

    def units(self) -> list[tuple[str, dict]]:
        """Returns each planned unit's run name and the payload its manifest is written from.

        Returns:
            Pairs of the unit's directory name relative to the training
            directory and the keyword arguments of the benchmark manifest writer.
        """
        from pipelines.benchmark.pipeline import BenchmarkPipeline
        from pipelines.benchmark.stages   import SeedExpandedStage

        planned = []
        for name, (model, component, seed) in SeedExpandedStage.planned_units(self.config, BenchmarkPipeline.select_models(self.config)):
            planned.append((name, {
                "run_tag"        : self.run_dir.name,
                "unit"           : name,
                "model"          : model,
                "loss_component" : component,
                "seed"           : seed,
            }))

        return planned

    def write(self, run_directory: Path, payload: dict) -> Path:
        """Writes one benchmark unit's manifest into its run directory."""
        from pipelines.benchmark.stages import BenchmarkUnitManifest

        return BenchmarkUnitManifest.write(run_directory, **payload)


class UnitManifestBackfiller:
    """Writes the missing unit manifests of one staged experiment run.

    Only units whose directory exists and carries no manifest are written, so
    running this twice changes nothing and a unit renamed by hand is reported
    rather than guessed at.

    Attributes:
        PLANS   : Plan classes probed in order to classify the run.
        path    : Path pasted on the command line, anywhere at or inside the run.
        logger  : Logger for progress and warnings.
        dry_run : Whether manifests are reported instead of written.
        written : Unit directories that received a manifest.
        kept    : Unit directories that already carried one.
        absent  : Planned units with no directory on disk.
    """

    PLANS = (FoldManifestPlan, BenchmarkManifestPlan)

    def __init__(self, path: Path, logger: Logger, dry_run: bool = False) -> None:
        """Binds the backfiller to a pasted path.

        Args:
            path    : Run directory of the experiment, or any path inside it.
            logger  : Logger for progress and warnings.
            dry_run : When set, nothing is written.
        """
        self.path    = Path(path)
        self.logger  = logger
        self.dry_run = dry_run
        self.written = []
        self.kept    = []
        self.absent  = []

    def _plan(self):
        """Returns the plan matching the located run.

        Returns:
            An instance of the plan class that claims the run directory.

        Raises:
            FileNotFoundError : If no staged run holds the pasted path; propagated from the locator.
            SystemExit        : If the run is neither a cross-validation nor a benchmark run.
        """
        root = StagedRunLocator.root(str(self.path))

        for plan_class in self.PLANS:
            if plan_class.claims(root):
                return plan_class(root)

        raise SystemExit(f"{root} holds no '{FoldManifestPlan.STAGE_DIRNAME}' or '{BenchmarkManifestPlan.STAGE_DIRNAME}' directory, so it is neither a cross-validation nor a benchmark run; only those two kinds write unit manifests")

    def _pending(self, plan) -> list[tuple[Path, dict]]:
        """Returns the unit directories that exist without a manifest, recording the other outcomes.

        Args:
            plan : Plan describing the units the run's recorded configuration schedules.

        Returns:
            Pairs of unit run directory and the payload its manifest is written from.
        """
        stage_dir = plan.run_dir / plan.STAGE_DIRNAME
        pending   = []

        for name, payload in plan.units():
            run_directory = stage_dir / name

            if not run_directory.is_dir():
                self.absent.append(name)
            elif UnitManifest.exists(run_directory):
                self.kept.append(name)
            else:
                pending.append((run_directory, payload))

        return pending

    def _unclaimed(self, plan, pending: list[tuple[Path, dict]]) -> list[Path]:
        """Returns the run directories under the stage directory that no planned unit accounts for.

        Args:
            plan    : Plan describing the scheduled units.
            pending : Unit directories about to receive a manifest.

        Returns:
            Run directories left without an identity after the backfill.
        """
        from pipelines.shared.inference.run_classifier import RunDirectoryWalk

        stage_dir = plan.run_dir / plan.STAGE_DIRNAME
        if not stage_dir.is_dir():
            return []

        claimed = {directory for directory, _ in pending}

        return [directory for directory in RunDirectoryWalk.walk(stage_dir) if directory not in claimed and not UnitManifest.exists(directory)]

    def _report(self, plan, unclaimed: list[Path]) -> None:
        """Logs what the backfill did and what it could not identify.

        Args:
            plan      : Plan describing the scheduled units.
            unclaimed : Run directories no planned unit accounts for.
        """
        stage_dir = plan.run_dir / plan.STAGE_DIRNAME

        self.logger.kv_table({
            "Run"             : str(plan.run_dir),
            "Kind"            : plan.KIND,
            "Units written"   : len(self.written),
            "Already carried" : len(self.kept),
            "Not on disk"     : len(self.absent),
            "Unidentified"    : len(unclaimed),
        }, title="Dry run" if self.dry_run else "Backfill")

        for directory in unclaimed:
            self.logger.warning(f"{directory.relative_to(stage_dir)}: run directory no planned unit accounts for, left without a manifest; it was renamed or belongs to different settings")

    def run(self) -> int:
        """Locates the run, writes every missing unit manifest and reports the outcome.

        Returns:
            Zero when every planned unit that exists on disk now carries a
            manifest, one when some directory was left unidentified.
        """
        plan      = self._plan()
        pending   = self._pending(plan)
        unclaimed = self._unclaimed(plan, pending)

        for run_directory, payload in pending:
            if not self.dry_run:
                plan.write(run_directory, payload)
            self.written.append(run_directory)
            self.logger.info(f"{'would write' if self.dry_run else 'wrote'} {UnitManifest.path(run_directory)}")

        self._report(plan, unclaimed)

        return 1 if unclaimed else 0


def main() -> None:
    """Parses the command line, backfills every given run and exits with the worst status."""
    parser = argparse.ArgumentParser(description="Write meta/unit.json onto the units of a cross-validation or benchmark run that ran before unit manifests existed, so a resume identifies them instead of training them again. Planned units are matched to directories by the name the run's recorded configuration composes, which is the rule those directories were created under; a directory no planned unit accounts for is reported and left alone.")
    parser.add_argument("paths", type=str, nargs="+", help="Run directory of a cross-validation or benchmark experiment, or any path inside it.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without writing.")
    args = parser.parse_args()

    logger    = Logger(log_dir="logs", name="backfill_unit_manifests")
    exit_code = 0

    for path in args.paths:
        exit_code = max(exit_code, UnitManifestBackfiller(Path(path), logger, dry_run=args.dry_run).run())

    logger.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
