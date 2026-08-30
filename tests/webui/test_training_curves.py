"""Tests covering the TensorBoard scalar reader behind the training-curves tab."""
from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from training_curves import TrainingCurves
from web_logger      import WebLogger


def _make_run(base: Path, name: str, tags: dict) -> Path:
    """Writes a TensorBoard run with the given scalar tags and returns its directory."""
    run    = base / name
    writer = SummaryWriter(log_dir=str(run / "tensorboard"))

    for tag, values in tags.items():
        for step, value in enumerate(values):
            writer.add_scalar(tag, value, step)

    writer.close()
    return run


def test_runs_lists_training_dirs(tmp_path):
    """Nested run directories are listed by their path relative to the base."""
    _make_run(tmp_path, "run_a", {"loss/train": [1.0, 0.5]})
    _make_run(tmp_path / "group", "run_b", {"loss/train": [2.0]})
    (tmp_path / "not_a_run").mkdir()

    curves = TrainingCurves(WebLogger())
    result = curves.runs(str(tmp_path))

    assert result["ok"]
    assert sorted(run["name"] for run in result["runs"]) == ["group/run_b", "run_a"]


def test_runs_rejects_bad_base(tmp_path):
    """Empty, relative and non-existent bases are rejected."""
    curves = TrainingCurves(WebLogger())

    assert not curves.runs("")["ok"]
    assert not curves.runs("relative")["ok"]
    assert not curves.runs(str(tmp_path / "missing"))["ok"]


def test_curves_overlay_and_default_tag(tmp_path):
    """An empty tag falls back to a shared tag and overlays one series per run."""
    run_a = _make_run(tmp_path, "run_a", {"loss/train": [1.0, 0.5, 0.2], "loss/val": [1.2, 0.7, 0.4]})
    run_b = _make_run(tmp_path, "run_b", {"loss/val": [2.0, 1.5]})

    curves = TrainingCurves(WebLogger())
    assert curves.runs(str(tmp_path))["ok"]

    result = curves.curves([str(run_a), str(run_b)], tag="")
    assert result["ok"]
    assert result["tag"] == "loss/val"
    assert set(result["tags"]) == {"loss/train", "loss/val"}
    assert len(result["series"]) == 2

    by_name = {series["name"]: series for series in result["series"]}
    assert by_name["run_a"]["steps"] == [0, 1, 2]
    assert abs(by_name["run_a"]["values"][2] - 0.4) < 1e-6
    assert by_name["run_b"]["steps"] == [0, 1]


def test_curves_respects_requested_tag_and_skips_missing(tmp_path):
    """A requested tag is honoured and runs lacking it are dropped from the overlay."""
    run_a = _make_run(tmp_path, "run_a", {"loss/train": [1.0], "lr": [0.001]})
    run_b = _make_run(tmp_path, "run_b", {"loss/train": [2.0]})

    curves = TrainingCurves(WebLogger())
    curves.runs(str(tmp_path))

    result = curves.curves([str(run_a), str(run_b)], tag="lr")
    assert result["ok"] and result["tag"] == "lr"
    assert [series["name"] for series in result["series"]] == ["run_a"]


def test_curves_rejects_unknown_run(tmp_path):
    """Curves are refused for runs the last listing did not discover."""
    run_a  = _make_run(tmp_path, "run_a", {"loss/train": [1.0]})
    curves = TrainingCurves(WebLogger())
    curves.runs(str(tmp_path))

    assert not curves.curves([str(tmp_path / "nowhere")], tag="")["ok"]

    other = TrainingCurves(WebLogger())
    assert not other.curves([str(run_a)], tag="")["ok"]


def test_the_scalar_cache_is_bounded_and_reused(tmp_path, monkeypatch):
    """The scalar cache evicts the oldest run once the cap is reached and serves the rest from memory."""
    monkeypatch.setattr(TrainingCurves, "CACHE_CAP", 2)
    runs   = [_make_run(tmp_path, f"run_{index}", {"loss/val": [float(index)]}) for index in range(3)]
    curves = TrainingCurves(WebLogger())
    curves.runs(str(tmp_path))

    for run in runs:
        assert curves.curves([str(run)], tag="loss/val")["ok"]

    assert curves.cache.size() == 2
    assert curves.cache.get(str(runs[0])) is None
    assert curves.cache.get(str(runs[2]))[1]["loss/val"] == [(0, 2.0)]


def test_downsample_caps_points(tmp_path):
    """Long series are downsampled to the point cap while keeping the first and last point."""
    curves = TrainingCurves(WebLogger())
    points = [(i, float(i)) for i in range(5000)]

    steps, values = curves._downsample(points)

    assert len(steps) <= TrainingCurves.MAX_POINTS + 1
    assert steps[0] == 0 and steps[-1] == 4999
    assert values[-1] == 4999.0
