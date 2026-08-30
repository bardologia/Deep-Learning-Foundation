"""Regression test for the settled marker the DDP fan-out waits on.

The fan-out launches the secondary ranks only once rank zero has written
``RunDirectorySettledMarker`` into its run directory. The backbone runner (and
the dual runner inheriting its ``run``) overrides the base ``run`` method, so
this test pins that the override still writes the marker before the preflight;
without it a backbone DDP run deadlocks with rank zero alone at the rendezvous.
"""

from __future__ import annotations

from types import SimpleNamespace

from pipelines.backbone.training.launcher      import SingleTrainRunner
from pipelines.shared.training.training_runner import RunDirectorySettledMarker


class MarkerProbe:
    """Records whether the settled marker existed when each training step ran.

    Attributes:
        seen : Mapping of step name to whether the marker file existed when the step was entered.
    """

    def __init__(self) -> None:
        """Starts with no observed steps."""
        self.seen = {}

    def observe(self, step: str, runner) -> None:
        """Records the marker's existence at the entry of one step.

        Args:
            step   : Name of the training step being entered.
            runner : Runner whose run directory holds the marker.
        """
        self.seen[step] = RunDirectorySettledMarker.path_for(runner.run_directory).is_file()


def test_backbone_run_writes_settled_marker_before_preflight_and_pipeline(tmp_path, monkeypatch):
    probe  = MarkerProbe()
    runner = SingleTrainRunner.__new__(SingleTrainRunner)

    runner.config = SimpleNamespace(
        logdir       = tmp_path,
        run_name     = "unit",
        resume       = False,
        resume_from  = None,
        infer_after  = False,
        infer_at_end = False,
        probe        = None,
        training     = SimpleNamespace(resume=False),
    )

    monkeypatch.setattr(SingleTrainRunner, "_resolve_run_name",   lambda self: "unit")
    monkeypatch.setattr(SingleTrainRunner, "_pretrain_preflight", lambda self: probe.observe("preflight", self))
    monkeypatch.setattr(SingleTrainRunner, "_pipeline",           lambda self, logdir: SimpleNamespace(run=lambda probe_config, resolved_entry_config: probe.observe("pipeline", self)))

    runner.run()

    assert probe.seen == {"preflight": True, "pipeline": True}
