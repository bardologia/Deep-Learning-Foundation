"""Resume policy for a single training or inference unit.

Decides whether a run directory can be reused, resumed from its checkpoint or
must be wiped and restarted, based on the completion marker written when a run
finishes. A reused run is compared with the configuration and the commit of the
launch reusing it, so results trained under other settings or other code never
enter a comparison unnoticed. Unfinished inference stamps are deleted so partial
outputs never look complete.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.inference.metadata           import CompletedInference
from pipelines.shared.training.resume_source       import TrainingResume
from tools.data.io                                 import FileIO
from tools.monitoring.logger                       import Logger
from tools.runtime.completion                      import CompletionMarker
from tools.runtime.config_cli                      import ConfigCli
from tools.runtime.git_state                       import GitState
from tools.training.pretraining                    import PretrainOrchestrator


class UnitResume:
    """Decides whether a unit's training or inference can be skipped.

    Attributes:
        EXCLUDED_FIELDS : Top-level entry-config fields never compared with a reused run: the hardware and resume flags of the launch, and the fields a sweep scheduler rewrites per unit.
        ABSENT          : Marker printed for a leaf one of the two configurations does not have.
        run_directory   : Run directory of the unit.
        enabled         : Whether resume logic applies at all.
        trainer_resume  : Whether the trainer itself continues from a checkpoint, which keeps unfinished run directories in place.
        config          : Entry configuration of this launch, compared with a reused run's recording; None for callers that only consult `skip_inference`.
        logger          : Logger created lazily on the first message.
    """

    EXCLUDED_FIELDS = tuple(TrainingResume.LAUNCH_FIELDS) + tuple(TrainingResume.UNIT_FIELDS)
    ABSENT          = "<absent>"

    def __init__(self, run_directory: Path, enabled: bool, trainer_resume: bool = False, config=None) -> None:
        """Stores the run directory, the resume flags and the launch configuration of this unit."""
        self.run_directory  = Path(run_directory)
        self.enabled        = enabled
        self.trainer_resume = trainer_resume
        self.config         = config
        self.logger         = None

    def _recorded_differences(self) -> dict | None:
        """Returns the leaves on which the reused run's recording and this launch disagree.

        Launch and per-unit fields are left out, as are the leaves the probes
        enabled in this launch would rewrite anyway. A leaf present on one side
        only is reported against the ABSENT marker, since that means the config
        schema changed between the two launches.

        Returns:
            Mapping of dotted leaf path to (recorded, current) pairs in path
            order, empty when the recording matches; None when the run recorded
            no configuration.
        """
        path = ResolvedEntryConfigWriter.path_for(self.run_directory)
        if not path.is_file():
            return None

        recorded = FileIO.load_json(path)
        current  = ConfigCli.to_mapping(self.config)
        tuned    = PretrainOrchestrator.tuned_leaves(current)

        differences = {}
        for leaf in sorted(set(recorded) | set(current)):
            if leaf.split(".")[0] in self.EXCLUDED_FIELDS or leaf in tuned:
                continue

            before = recorded[leaf] if leaf in recorded else self.ABSENT
            after  = current[leaf]  if leaf in current  else self.ABSENT

            if before != after:
                differences[leaf] = (before, after)

        return differences

    def _commit_difference(self) -> tuple | None:
        """Returns the (recorded, current) commits when other code produced the reused run, else None.

        A run without a provenance record predates provenance recording, and
        its commit is reported as unknown.
        """
        path    = ResolvedEntryConfigWriter.provenance_path_for(self.run_directory)
        current = GitState.commit()

        if not path.is_file():
            return (f"unknown (no {ResolvedEntryConfigWriter.PROVENANCE_PATH} recorded)", current)

        recorded = FileIO.load_json(path)["git_commit"]

        return None if recorded == current else (recorded, current)

    def _report_reuse(self) -> None:
        """Warns that the completed run is reused and names every setting and the commit on which it differs from this launch."""
        logger       = self._logger()
        name         = self.run_directory.name
        completed_at = CompletionMarker.payload(self.run_directory)["completed_at"]
        differences  = self._recorded_differences()
        commits      = self._commit_difference()

        logger.warning(f"{name}: completed training reused (completed {completed_at}); no training runs for this unit")

        if differences is None:
            logger.warning(f"{name}: the reused run recorded no {ResolvedEntryConfigWriter.RELATIVE_PATH}, so its configuration cannot be compared with this launch")
        elif differences:
            rows = [{"Field": leaf, "Recorded": repr(before), "Now": repr(after)} for leaf, (before, after) in differences.items()]
            logger.warning(f"{name}: the reused run was trained under a different configuration on {len(rows)} leaf(s); its results enter this launch with those settings, not the current ones")
            logger.metrics_table(rows, columns=["Field", "Recorded", "Now"], title="Reused run vs this launch (launch, unit and probe-tuned fields excluded)")
        else:
            logger.ok(f"{name}: the reused run recorded the configuration of this launch")

        if commits is not None:
            logger.warning(f"{name}: the reused run was produced at commit {commits[0]}; this launch runs {commits[1]}")

    def skip_training(self) -> bool:
        """Returns whether the completed run can be reused instead of retrained.

        A completed run is reused after its recording and commit are compared
        with this launch. An unfinished run is left alone when the trainer
        resumes from its checkpoint, and otherwise deleted so training restarts
        from scratch. With resume disabled an existing directory is trained
        into as it is, which is reported because the earlier run's files stay.

        Raises:
            ValueError : If the unit was built without the entry configuration the reuse check compares.
        """
        if self.config is None:
            raise ValueError(f"UnitResume for {self.run_directory} was built without the entry configuration; skip_training compares a reused run with it, pass config=")

        if not self.enabled:
            if self.run_directory.is_dir():
                self._logger().warning(f"{self.run_directory.name}: resume disabled, a fresh run is written into the existing directory {self.run_directory}; the earlier run's tensorboard events, snapshots and inference outputs stay beside the new ones")
            return False

        if CompletionMarker.is_complete(self.run_directory):
            self._report_reuse()
            return True

        if self.trainer_resume:
            return False

        if self.run_directory.is_dir():
            shutil.rmtree(self.run_directory)
            self._logger().info(f"{self.run_directory.name}: unfinished run deleted, restarting from scratch")

        return False

    def skip_inference(self, split: str, subdir: str | None = None) -> bool:
        """Returns whether an existing inference for a split can be reused.

        Unfinished inference stamps are deleted first, so only completed
        stamps count as reusable.

        Args:
            split  : Dataset split the inference covers (e.g. ``test``).
            subdir : Optional subdirectory of ``inference`` holding the stamps.

        Returns:
            True when a completed stamp for the split already exists.
        """
        if not self.enabled:
            return False

        inference_dir = self.run_directory / "inference" / subdir if subdir else self.run_directory / "inference"
        if not inference_dir.is_dir():
            return False

        for candidate in sorted(inference_dir.iterdir()):
            if candidate.is_dir() and not CompletionMarker.is_complete(candidate):
                shutil.rmtree(candidate)
                self._logger().info(f"{self.run_directory.name}: unfinished inference '{candidate.name}' deleted")

        if CompletedInference.stamps(inference_dir, split):
            self._logger().info(f"{self.run_directory.name}: existing '{split}' inference reused")
            return True

        return False

    def _logger(self) -> Logger:
        """Returns the appending logger of the unit's run directory, created on first use."""
        if self.logger is None:
            self.logger = Logger(log_dir=str(self.run_directory / "logs"), name="unit_resume", level="INFO", file_mode="a")

        return self.logger
