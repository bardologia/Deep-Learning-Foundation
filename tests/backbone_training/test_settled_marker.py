
from __future__ import annotations

from types import SimpleNamespace

from pipelines.backbone.training.launcher      import SingleTrainRunner
from pipelines.shared.training.training_runner import RunDirectorySettledMarker


class MarkerProbe:

    def __init__(self) -> None:
        self.seen = {}

    def observe(self, step: str, runner) -> None:
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
