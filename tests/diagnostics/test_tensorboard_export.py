"""Tests for exporting TensorBoard scalars into publication-style curve plots.

Covers tag grouping and train/validation pairing, filename sanitisation, axis
labelling and title wrapping, log-axis selection, event-file reading, the
single-run and batch exports, and the seed overlay written per trial.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from configuration.diagnostics            import TensorboardExportConfig, TensorboardExportEntryConfig
from tools.diagnostics.tensorboard_export import CurveGroup, CurveLabeler, ScalarCurvePlots, ScalarTagGrouper, SeedRunGrouper, SeedScalarMerger, TensorboardExport, TensorboardExportBatch, TensorboardScalarReader

from tests.conftest import RecordingLogger


def _series(tags):
    """Returns a scalar series mapping each tag to three steps and values in [0, 1]."""
    steps = np.arange(3, dtype=np.int64)
    return {tag: (steps, np.linspace(0.0, 1.0, 3)) for tag in tags}


def test_train_and_val_series_share_one_group():
    """Train and validation tags of the same stem merge into a single labelled curve group."""
    groups = ScalarTagGrouper().group(_series(["loss/train", "loss/val"]))

    assert len(groups) == 1
    assert groups[0].stem == "loss"
    assert [label for label, _steps, _values in groups[0].series] == ["Training", "Validation"]


def test_validation_segment_pairs_with_train_segment():
    """A split segment in the middle of a tag path pairs train with validation under one stem."""
    groups = ScalarTagGrouper().group(_series(["loss_components/train/param_l1", "loss_components/validation/param_l1"]))

    assert len(groups) == 1
    assert groups[0].stem == "loss_components/param_l1"


def test_unpaired_split_tag_keeps_its_original_path():
    """A split tag with no partner keeps its full path and carries no role label."""
    groups = ScalarTagGrouper().group(_series(["train/grad_norm_before_clip"]))

    assert len(groups) == 1
    assert groups[0].stem == "train/grad_norm_before_clip"
    assert groups[0].series[0][0] is None


def test_split_segment_matching_is_exact():
    """A tag merely containing 'val' as part of a word is not treated as a validation split."""
    groups = ScalarTagGrouper().group(_series(["early_stop/best_val_loss"]))

    assert len(groups) == 1
    assert groups[0].stem == "early_stop/best_val_loss"


def test_duplicate_roles_in_one_pairing_raise():
    """Two tags claiming the same role for one stem raise ValueError."""
    with pytest.raises(ValueError, match="duplicate roles"):
        ScalarTagGrouper().group(_series(["loss/train", "train/loss"]))


def test_colliding_plot_paths_raise():
    """A bare tag colliding with a grouped stem's output path raises ValueError."""
    with pytest.raises(ValueError, match="colliding"):
        ScalarTagGrouper().group(_series(["loss", "loss/train", "loss/val"]))


def test_tag_segments_are_sanitized_for_filesystem_paths():
    """Spaces and reserved characters in tag segments become underscores."""
    groups = ScalarTagGrouper().group(_series(["metrics/my metric: a*b"]))

    assert groups[0].stem == "metrics/my_metric_a_b"


def test_labels_humanize_segments_acronyms_and_units():
    """Titles and axis labels expand segments, uppercase known acronyms and move units into parentheses."""
    labeler = CurveLabeler()

    assert labeler.title("loss")                              == "Loss"
    assert labeler.title("loss_components/param_l1")          == "Loss Components / Param L1"
    assert labeler.title("system/resources/ram_used_gb")      == "System / Resources / RAM Used (GB)"
    assert labeler.y_label("system/resources/gpu0_vram_pct")  == "GPU0 VRAM (%)"
    assert labeler.y_label("system/resources/disk_read_mb_s") == "Disk Read (MB/s)"
    assert labeler.y_label("lr/warmup_factor")                == "Warmup Factor"


def test_long_titles_wrap_instead_of_overflowing():
    """A long tag path wraps onto lines no wider than the configured wrap width."""
    title = CurveLabeler().title("permutation/validation/perm/placeholder/precision/slot_0")

    assert "\n" in title
    assert all(len(line) <= CurveLabeler.WRAP_WIDTH for line in title.split("\n"))


def _value_arrays(values):
    """Returns the values wrapped as a single float64 array in a list."""
    return [np.asarray(values, dtype=np.float64)]


def test_wide_positive_ranges_render_on_a_log_axis():
    """Strictly positive values spanning orders of magnitude select a logarithmic axis."""
    assert ScalarCurvePlots()._log_scaled(_value_arrays([1e-6, 1e-4, 1e-2])) is True


def test_narrow_or_signed_ranges_stay_linear():
    """Narrow, signed or zero-containing ranges keep a linear axis."""
    assert ScalarCurvePlots()._log_scaled(_value_arrays([0.1, 0.4]))        is False
    assert ScalarCurvePlots()._log_scaled(_value_arrays([-1e-6, 1e-2]))     is False
    assert ScalarCurvePlots()._log_scaled(_value_arrays([0.0, 1e-2, 1e4]))  is False


@pytest.fixture
def run_directory(tmp_path):
    """Returns a run directory holding a TensorBoard event file with train, validation and learning-rate scalars."""
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
    """The reader returns every logged tag with its step and value arrays."""
    series = TensorboardScalarReader(TensorboardExportConfig(run_directory=run_directory)).read()

    assert set(series) == {"loss/train", "loss/val", "lr/encoder"}
    steps, values = series["loss/train"]
    assert steps.tolist() == [0, 1, 2, 3, 4]
    assert values.shape == (5,)


def test_reader_raises_without_tensorboard_directory(tmp_path):
    """Reading a run without a tensorboard subdirectory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TensorboardScalarReader(TensorboardExportConfig(run_directory=tmp_path)).read()


def test_export_writes_plots_into_the_run_directory(run_directory):
    """The export writes one plot per curve group under tensorboard_plots, nesting subpaths."""
    config = TensorboardExportConfig(run_directory=run_directory)
    result = TensorboardExport(config, RecordingLogger()).run()

    assert result["n_series"] == 3
    assert result["n_plots"]  == 2
    assert (run_directory / "tensorboard_plots" / "loss.png").is_file()
    assert (run_directory / "tensorboard_plots" / "lr" / "encoder.png").is_file()


def test_batch_exports_filtered_runs(run_directory, tmp_path):
    """The batch export processes only the runs matched by the filter."""
    entry   = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["run_demo"])
    results = TensorboardExportBatch(entry, RecordingLogger()).run()

    assert len(results) == 1
    assert results[0]["run_directory"] == str(run_directory)
    assert (run_directory / "tensorboard_plots" / "loss.png").is_file()


def _seed_run(root, trial, seed, tags=("loss/train", "loss/val")):
    """Returns a seed run directory under the trial with a TensorBoard event file for the given tags."""
    run_dir = root / trial / f"seed{seed}"
    writer  = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    for step in range(4):
        for tag in tags:
            writer.add_scalar(tag, (seed + 1.0) / (step + 1), step)

    writer.flush()
    writer.close()
    return run_dir


def test_merger_combines_seed_groups_per_stem():
    """Seed groups sharing a stem merge into one group with seed-sorted, role-labelled series."""
    steps, values = np.arange(3, dtype=np.int64), np.linspace(0.0, 1.0, 3)
    group         = CurveGroup(stem="loss", title="loss", series=[("Training", steps, values), ("Validation", steps, values)])

    merged = SeedScalarMerger().merge({"seed1": [group], "seed0": [group]})

    assert len(merged) == 1
    assert merged[0].stem == "loss"
    assert [(seed, role) for seed, role, _s, _v in merged[0].series] == [("seed0", "Training"), ("seed0", "Validation"), ("seed1", "Training"), ("seed1", "Validation")]


def test_grouper_collects_sibling_seed_runs_only(tmp_path):
    """Only parents holding more than one seed directory are grouped as trials."""
    seed_dirs = [tmp_path / "trial_a" / "seed0", tmp_path / "trial_a" / "seed1", tmp_path / "trial_b" / "seed0", tmp_path / "flat_run"]

    trials = SeedRunGrouper.trials(seed_dirs)

    assert list(trials) == [tmp_path / "trial_a"]
    assert trials[tmp_path / "trial_a"] == [tmp_path / "trial_a" / "seed0", tmp_path / "trial_a" / "seed1"]


def test_batch_writes_seed_overlays_into_the_trial_directory(tmp_path):
    """Exporting sibling seed runs also writes a combined overlay plot in the trial directory."""
    for seed in (0, 1):
        _seed_run(tmp_path, "trial_a", seed)

    entry   = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["trial_a/seed0", "trial_a/seed1"])
    results = TensorboardExportBatch(entry, RecordingLogger()).run()

    assert len(results) == 3
    assert results[-1]["n_seeds"] == 2
    assert (tmp_path / "trial_a" / "seed0" / "tensorboard_plots" / "loss.png").is_file()
    assert (tmp_path / "trial_a" / "tensorboard_plots" / "loss.png").is_file()


def test_trial_name_filter_expands_to_the_seed_runs(tmp_path):
    """Filtering by trial name expands to its seed runs and still produces the trial overlay."""
    for seed in (0, 1):
        _seed_run(tmp_path, "trial_a", seed)

    entry   = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["trial_a"])
    results = TensorboardExportBatch(entry, RecordingLogger()).run()

    assert len(results) == 3
    assert (tmp_path / "trial_a" / "tensorboard_plots" / "loss.png").is_file()


def test_ambiguous_bare_seed_name_raises(tmp_path):
    """A bare seed name matching several trials raises FileNotFoundError."""
    for trial in ("trial_a", "trial_b"):
        _seed_run(tmp_path, trial, 0)

    entry = TensorboardExportEntryConfig(runs_dir=tmp_path, run_filter=["seed0"])

    with pytest.raises(FileNotFoundError, match="seed0"):
        TensorboardExportBatch(entry, RecordingLogger()).run()
