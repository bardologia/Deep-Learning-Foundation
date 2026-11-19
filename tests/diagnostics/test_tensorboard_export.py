
from __future__ import annotations

import contextlib

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from configuration.diagnostics            import TensorboardExportConfig, TensorboardExportEntryConfig
from tools.diagnostics.tensorboard_export import CurveGroup, CurveLabeler, ScalarCurvePlots, ScalarTagGrouper, SeedRunGrouper, SeedScalarMerger, TensorboardExport, TensorboardExportBatch, TensorboardScalarReader

from tests.conftest import RecordingLogger


class _RecordingProgress:

    def __init__(self) -> None:
        self.tasks    = []
        self.advanced = []

    def add_task(self, description, total=None, **kwargs):
        self.tasks.append((description, total))
        self.advanced.append(0)
        return len(self.tasks) - 1

    def advance(self, task, advance=1):
        self.advanced[task] += advance


class _TrackingLogger(RecordingLogger):

    def __init__(self) -> None:
        super().__init__()
        self.progress = _RecordingProgress()

    @contextlib.contextmanager
    def track(self, transient: bool = False):
        yield self.progress

    def of(self, level: str) -> list:
        return [message for recorded, message in self.messages if recorded == level]


def _series(tags):
    steps = np.arange(3, dtype=np.int64)
    return {tag: (steps, np.linspace(0.0, 1.0, 3)) for tag in tags}


def test_train_and_val_series_share_one_group():
    groups = ScalarTagGrouper().group(_series(["loss/train", "loss/val"]))

    assert len(groups) == 1
    assert groups[0].stem == "loss"
    assert [label for label, _steps, _values in groups[0].series] == ["Training", "Validation"]


def test_validation_segment_pairs_with_train_segment():
    groups = ScalarTagGrouper().group(_series(["loss_components/train/param_l1", "loss_components/validation/param_l1"]))

    assert len(groups) == 1
    assert groups[0].stem == "loss_components/param_l1"


def test_unpaired_split_tag_keeps_its_original_path():
    groups = ScalarTagGrouper().group(_series(["train/grad_norm_before_clip"]))

    assert len(groups) == 1
    assert groups[0].stem == "train/grad_norm_before_clip"
    assert groups[0].series[0][0] is None


def test_split_segment_matching_is_exact():
    groups = ScalarTagGrouper().group(_series(["early_stop/best_val_loss"]))

    assert len(groups) == 1
    assert groups[0].stem == "early_stop/best_val_loss"


def test_tag_segments_are_sanitized_for_filesystem_paths():
    groups = ScalarTagGrouper().group(_series(["metrics/my metric: a*b"]))

    assert groups[0].stem == "metrics/my_metric_a_b"


def test_labels_humanize_segments_acronyms_and_units():
    labeler = CurveLabeler()

    assert labeler.title("loss")                              == "Loss"
    assert labeler.title("loss_components/param_l1")          == "Loss Components / Param L1"
    assert labeler.title("system/resources/ram_used_gb")      == "System / Resources / RAM Used (GB)"
    assert labeler.y_label("system/resources/gpu0_vram_pct")  == "GPU0 VRAM (%)"
    assert labeler.y_label("system/resources/disk_read_mb_s") == "Disk Read (MB/s)"
    assert labeler.y_label("lr/warmup_factor")                == "Warmup Factor"


def test_long_titles_wrap_instead_of_overflowing():
    title = CurveLabeler().title("permutation/validation/perm/placeholder/precision/slot_0")

    assert "\n" in title
    assert all(len(line) <= CurveLabeler.WRAP_WIDTH for line in title.split("\n"))


def _value_arrays(values):
    return [np.asarray(values, dtype=np.float64)]


def test_wide_positive_ranges_render_on_a_log_axis():
    assert ScalarCurvePlots()._log_scaled(_value_arrays([1e-6, 1e-4, 1e-2])) is True


def test_narrow_or_signed_ranges_stay_linear():
    assert ScalarCurvePlots()._log_scaled(_value_arrays([0.1, 0.4]))        is False
    assert ScalarCurvePlots()._log_scaled(_value_arrays([-1e-6, 1e-2]))     is False
    assert ScalarCurvePlots()._log_scaled(_value_arrays([0.0, 1e-2, 1e4]))  is False


@pytest.fixture
def run_directory(tmp_path):
    run_dir = tmp_path / "run_demo"
    writer  = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    for step in range(5):
        writer.add_scalar("loss/train", 1.0 / (step + 1), step)
        writer.add_scalar("lr/encoder", 1e-3, step)
        if step % 2 == 0:
            writer.add_scalar("loss/val", 1.5 / (step + 1), step)

    writer.flush()
    writer.close()
    return run_dir


def test_reader_loads_all_scalar_series(run_directory):
    series = TensorboardScalarReader(TensorboardExportConfig(run_directory=run_directory)).read()

    assert set(series) == {"loss/train", "loss/val", "lr/encoder"}
    steps, values = series["loss/train"]
    assert steps.tolist() == [0, 1, 2, 3, 4]
    assert values.shape == (5,)


def test_export_writes_plots_into_the_run_directory(run_directory):
    config = TensorboardExportConfig(run_directory=run_directory)
    result = TensorboardExport(config, RecordingLogger()).run()

    assert result["n_series"] == 3
    assert result["n_plots"]  == 2
    assert (run_directory / "tensorboard_plots" / "loss.png").is_file()
    assert (run_directory / "tensorboard_plots" / "lr" / "encoder.png").is_file()


def test_batch_exports_filtered_runs(run_directory, tmp_path):
    entry   = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["run_demo"])
    logger  = _TrackingLogger()
    results = TensorboardExportBatch(entry, logger).run()

    assert len(results) == 1
    assert results[0]["run_directory"] == str(run_directory)
    assert (run_directory / "tensorboard_plots" / "loss.png").is_file()

    assert ("metrics_table", "Tensorboard export summary") in logger.messages
    assert ("metrics_table", "Seed overlay summary")   not in logger.messages
    assert logger.of("ok")[-1] == "Exported 1 run(s) (2 plots); 0 multi-seed trial(s) (0 overlay plots)"


def _seed_run(root, trial, seed, tags=("loss/train", "loss/val")):
    run_dir = root / trial / f"seed{seed}"
    writer  = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    for step in range(4):
        for tag in tags:
            writer.add_scalar(tag, (seed + 1.0) / (step + 1), step)

    writer.flush()
    writer.close()
    return run_dir


def test_merger_combines_seed_groups_per_stem():
    steps, values = np.arange(3, dtype=np.int64), np.linspace(0.0, 1.0, 3)
    group         = CurveGroup(stem="loss", title="loss", series=[("Training", steps, values), ("Validation", steps, values)])

    merged = SeedScalarMerger().merge({"seed1": [group], "seed0": [group]})

    assert len(merged) == 1
    assert merged[0].stem == "loss"
    assert [(seed, role) for seed, role, _s, _v in merged[0].series] == [("seed0", "Training"), ("seed0", "Validation"), ("seed1", "Training"), ("seed1", "Validation")]


def test_grouper_collects_sibling_seed_runs_only(tmp_path):
    seed_dirs = [tmp_path / "trial_a" / "seed0", tmp_path / "trial_a" / "seed1", tmp_path / "trial_b" / "seed0", tmp_path / "flat_run"]

    trials = SeedRunGrouper.trials(seed_dirs)

    assert list(trials) == [tmp_path / "trial_a"]
    assert trials[tmp_path / "trial_a"] == [tmp_path / "trial_a" / "seed0", tmp_path / "trial_a" / "seed1"]


def test_batch_writes_seed_overlays_into_the_trial_directory(tmp_path):
    for seed in (0, 1):
        _seed_run(tmp_path, "trial_a", seed)

    entry   = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["trial_a/seed0", "trial_a/seed1"])
    results = TensorboardExportBatch(entry, RecordingLogger()).run()

    assert len(results) == 3
    assert results[-1]["n_seeds"] == 2
    assert (tmp_path / "trial_a" / "seed0" / "tensorboard_plots" / "loss.png").is_file()
    assert (tmp_path / "trial_a" / "tensorboard_plots" / "loss.png").is_file()


def test_batch_numbers_each_run_and_closes_with_run_and_overlay_totals(tmp_path):
    for seed in (0, 1):
        _seed_run(tmp_path, "trial_a", seed)

    entry  = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["trial_a"])
    logger = _TrackingLogger()
    TensorboardExportBatch(entry, logger).run()

    assert logger.of("section") == ["Tensorboard plot export [1/2] trial_a/seed0", "Tensorboard plot export [2/2] trial_a/seed1"]
    assert ("metrics_table", "Tensorboard export summary") in logger.messages
    assert ("metrics_table", "Seed overlay summary")       in logger.messages
    assert logger.of("ok")[-1] == "Exported 2 run(s) (2 plots); 1 multi-seed trial(s) (1 overlay plots)"


def test_seed_reads_and_renders_report_progress(tmp_path):
    for seed in (0, 1):
        _seed_run(tmp_path, "trial_a", seed, tags=("loss/train", "loss/val", "lr/encoder"))

    entry  = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["trial_a"])
    logger = _TrackingLogger()
    TensorboardExportBatch(entry, logger).run()

    assert logger.of("info")         == ["seed0: 3 scalar series", "seed1: 3 scalar series"]
    assert logger.progress.tasks    == [("seed0 plots", 2), ("seed1 plots", 2), ("trial_a overlays", 2)]
    assert logger.progress.advanced == [2, 2, 2]


def test_trial_name_filter_expands_to_the_seed_runs(tmp_path):
    for seed in (0, 1):
        _seed_run(tmp_path, "trial_a", seed)

    entry   = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["trial_a"])
    results = TensorboardExportBatch(entry, RecordingLogger()).run()

    assert len(results) == 3
    assert (tmp_path / "trial_a" / "tensorboard_plots" / "loss.png").is_file()

