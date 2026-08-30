"""Tests for the backbone and profile dataset pipelines and their normalization-stats persistence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.dataset                          import DatasetConfig, InputConfig, PatchConfig, ProfileDatasetConfig, Representation, SplitRegions
from configuration.dataset.gaussian                 import GaussianConfig
from configuration.normalization.general            import ChannelStats, ChannelStrategy, NormMethod
from configuration.synthesis                        import SynthesisConfig
from pipelines.backbone.dataset.pipeline            import DatasetPipeline
from pipelines.profile_autoencoder.dataset.pipeline import ProfileDatasetPipeline
from pipelines.profile_autoencoder.dataset          import ProfileStats
from pipelines.backbone.dataset.stats               import Stats
from tools.data.io                                  import FileIO
from tools.data.regions                             import CropRegion
from tools.monitoring.logger                        import Logger


def _logger(tmp_path, name) -> Logger:
    """Returns a silent logger writing under tmp_path."""
    return Logger(log_dir=str(tmp_path / "logs"), name=name, level="ERROR")


def _backbone_config(test_data_dir, params_dir) -> DatasetConfig:
    """Returns a backbone dataset config over three contiguous azimuth splits of the test data."""
    ic = InputConfig(
        use_primary=True,        primary_representation=Representation.MAG_ONLY,
        use_secondaries=True,    secondaries_representation=Representation.MAG_ONLY,
        use_derived=True, derived_representation=Representation.ANGLE_ONLY,
    )

    splits = SplitRegions(
        train = CropRegion(1000, 1024, 500, 524),
        val   = CropRegion(1024, 1040, 500, 524),
        test  = CropRegion(1040, 1056, 500, 524),
    )

    return DatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = splits,
        secondary_labels            = ("CH01", "CH02"),
        patch                       = PatchConfig(size=(8, 8), stride=(8, 8)),
        input_config                = ic,
        batch_size                  = 2,
        num_workers                 = 0,
        prefetch_factor             = 8,
        shuffle_train               = True,
        n_gaussians                 = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy").n_default_gaussians,
    )


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_writes_metadata_and_stats(test_data_dir, params_dir, tmp_path):
    """The backbone pipeline writes its metadata files and returns normalized datasets and loaders."""
    config   = _backbone_config(test_data_dir, params_dir)
    pipeline = DatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "bb_pipe"), seed=0)

    train_loader, val_loader, test_loader, datasets = pipeline.run()

    meta = tmp_path / "meta"

    assert (meta / "normalization_stats.json").is_file()
    assert (meta / "dataset_creation_config.json").is_file()
    assert (meta / "crop.json").is_file()
    assert (meta / "patch.json").is_file()

    assert pipeline.profile_length == SynthesisConfig().profile_length
    assert datasets["train"].normalizer is not None
    assert len(train_loader) >= 1


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_train_order_reproducible(test_data_dir, params_dir, tmp_path):
    """Two runs with the same seed yield the same first training batch."""
    torch.use_deterministic_algorithms(False)

    def _first_batch(run_dir):
        """Returns the first training batch of a backbone pipeline run seeded at 7 under run_dir."""
        config   = _backbone_config(test_data_dir, params_dir)
        pipeline = DatasetPipeline(config, run_dir, logger=_logger(run_dir, "bb_repro"), seed=7)
        loaders  = pipeline.run()
        x, _     = next(iter(loaders[0]))
        return x.numpy()

    a = _first_batch(tmp_path / "a")
    b = _first_batch(tmp_path / "b")

    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_rectangular_patches_yield_rectangular_batches(test_data_dir, params_dir, tmp_path):
    """A rectangular patch config produces batches with matching rectangular spatial extent."""
    config       = _backbone_config(test_data_dir, params_dir)
    config.patch = PatchConfig(size=(8, 16), stride=(4, 8))

    pipeline = DatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "bb_rect"), seed=0)

    train_loader, _val_loader, _test_loader, _datasets = pipeline.run()
    x, y = next(iter(train_loader))

    assert tuple(x.shape[-2:]) == (8, 16)
    assert tuple(y.shape[-2:]) == (8, 16)


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_pipeline_crops_only_the_enabled_input_groups(test_data_dir, params_dir, tmp_path):
    """Disabling the primary and secondary groups loads only the interferogram layers."""
    full    = _backbone_config(test_data_dir, params_dir)
    trimmed = _backbone_config(test_data_dir, params_dir)

    trimmed.input_config = InputConfig(
        use_primary=False,
        use_secondaries=False,
        use_derived=True, derived_representation=Representation.ANGLE_ONLY,
    )

    full_datasets    = DatasetPipeline(full,    tmp_path / "full",    logger=_logger(tmp_path, "bb_full"),    seed=0).run()[3]
    trimmed_datasets = DatasetPipeline(trimmed, tmp_path / "trimmed", logger=_logger(tmp_path, "bb_trimmed"), seed=0).run()[3]

    full_train    = full_datasets["train"]
    trimmed_train = trimmed_datasets["train"]

    assert (full_train.n_primary, full_train.n_secondaries, full_train.n_derived)          == (1, 2, 2)
    assert (trimmed_train.n_primary, trimmed_train.n_secondaries, trimmed_train.n_derived) == (0, 0, 2)

    assert full_train.inputs.shape[0]    == 5
    assert trimmed_train.inputs.shape[0] == 2
    assert trimmed_train.input_channels  == 2

    np.testing.assert_array_equal(trimmed_train.inputs, full_train.inputs[3:])


@pytest.mark.dataset
def test_backbone_pipeline_rejects_rot90_with_rectangular_patch(test_data_dir, params_dir, tmp_path):
    """Enabling 90-degree rotation with a rectangular patch raises ValueError."""
    config                      = _backbone_config(test_data_dir, params_dir)
    config.patch                = PatchConfig(size=(8, 16), stride=(4, 8))
    config.augmentation.p_rot90 = 0.5

    with pytest.raises(ValueError, match="rectangular patch"):
        DatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "bb_rot90"), seed=0)


@pytest.mark.dataset
@pytest.mark.slow
def test_profile_pipeline_runs_and_returns_normalizer(test_data_dir, params_dir, tmp_path):
    """The profile pipeline writes its stats file and returns loaders, height axis, and normalizer."""
    config = ProfileDatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = SplitRegions(
            train = CropRegion(1000, 1040, 500, 540),
            val   = CropRegion(1040, 1060, 500, 540),
            test  = CropRegion(1060, 1080, 500, 540),
        ),
        n_gaussians       = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy").n_default_gaussians,
        batch_size        = 16,
        num_workers       = 0,
        prefetch_factor   = 2,
        stats_max_samples = 500,
    )

    pipeline = ProfileDatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "prof_pipe"), seed=0)

    (train_loader, val_loader, test_loader), datasets, z_axis, z_len, normalizer = pipeline.run()

    assert z_len == SynthesisConfig().profile_length
    assert z_axis.shape == (SynthesisConfig().profile_length,)
    assert (tmp_path / "meta" / "profile_normalization_stats.json").is_file()
    assert np.isfinite(normalizer.loc)
    assert normalizer.scale >= normalizer.SCALE_FLOOR

    noisy, clean = next(iter(train_loader))
    assert noisy.shape[1] == SynthesisConfig().profile_length
    assert clean.shape[1] == SynthesisConfig().profile_length
    assert len(val_loader)  >= 1
    assert len(test_loader) >= 1


@pytest.mark.dataset
@pytest.mark.slow
def test_profile_pipeline_train_order_reproducible(test_data_dir, params_dir, tmp_path):
    """Two profile runs with the same seed yield the same first training batch."""
    def _first_batch(run_dir):
        """Returns the first training batch of a profile pipeline run seeded at 3 under run_dir."""
        config = ProfileDatasetConfig(
            preprocessing_run_directory = test_data_dir,
            parameters_path             = params_dir / "parameters.npy",
            split_regions               = SplitRegions(
                train = CropRegion(1000, 1040, 500, 540),
                val   = CropRegion(1040, 1060, 500, 540),
                test  = CropRegion(1060, 1080, 500, 540),
            ),
            n_gaussians       = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy").n_default_gaussians,
            batch_size        = 16,
            num_workers       = 0,
            prefetch_factor   = 2,
            stats_max_samples = 500,
        )
        pipeline = ProfileDatasetPipeline(config, run_dir, logger=_logger(run_dir, "prof_repro"), seed=3)
        loaders  = pipeline.run()[0]
        return next(iter(loaders[0]))[0].numpy()

    a = _first_batch(tmp_path / "a")
    b = _first_batch(tmp_path / "b")

    np.testing.assert_allclose(a, b, atol=1e-5)


def _stats(scale: float) -> Stats:
    """Returns single-channel input and output statistics with the given z-score scale."""
    channels = ChannelStats(
        loc        = [0.0],
        scale      = [scale],
        names      = ["pass/mag"],
        strategies = [ChannelStrategy(NormMethod.ZSCORE)],
        clampable  = [False],
    )

    return Stats(input_stats=channels, output_stats=channels)


def _resumable_pipeline(run_dir, resume: bool) -> DatasetPipeline:
    """Returns a bare DatasetPipeline wired only with the run directory, resume flag, and logger."""
    pipeline                        = DatasetPipeline.__new__(DatasetPipeline)
    pipeline.training_run_directory = run_dir
    pipeline.resume                 = resume
    pipeline.logger                 = _logger(run_dir, "persist_stats")

    (run_dir / "meta").mkdir(parents=True, exist_ok=True)

    return pipeline


def test_resumed_run_keeps_matching_normalization_stats(tmp_path):
    """A resumed run accepts statistics identical to the stored ones."""
    pipeline = _resumable_pipeline(tmp_path, resume=True)

    _stats(2.0).save(tmp_path / "meta")
    pipeline._persist_stats(_stats(2.0))

    stored = FileIO.load_json(tmp_path / "meta" / Stats.FILENAME)

    assert stored["input_stats"]["channels"][0]["scale"] == 2.0


def test_resumed_run_refuses_to_overwrite_different_stats(tmp_path):
    """A resumed run raises ValueError when the recomputed statistics differ from the stored ones."""
    pipeline = _resumable_pipeline(tmp_path, resume=True)

    _stats(2.0).save(tmp_path / "meta")

    with pytest.raises(ValueError, match="normalization statistics"):
        pipeline._persist_stats(_stats(5.0))


def test_fresh_run_overwrites_normalization_stats(tmp_path):
    """A fresh run replaces any statistics left on disk."""
    pipeline = _resumable_pipeline(tmp_path, resume=False)

    _stats(2.0).save(tmp_path / "meta")
    pipeline._persist_stats(_stats(5.0))

    stored = FileIO.load_json(tmp_path / "meta" / Stats.FILENAME)

    assert stored["input_stats"]["channels"][0]["scale"] == 5.0


def _resumable_profile_pipeline(run_dir, resume: bool) -> ProfileDatasetPipeline:
    """Returns a bare ProfileDatasetPipeline wired only with the run directory, resume flag, and logger."""
    pipeline                        = ProfileDatasetPipeline.__new__(ProfileDatasetPipeline)
    pipeline.training_run_directory = run_dir
    pipeline.resume                 = resume
    pipeline.logger                 = _logger(run_dir, "persist_profile_stats")

    (run_dir / "meta").mkdir(parents=True, exist_ok=True)

    return pipeline


def test_resumed_profile_run_keeps_matching_stats(tmp_path):
    """A resumed profile run accepts statistics identical to the stored ones."""
    pipeline = _resumable_profile_pipeline(tmp_path, resume=True)

    ProfileStats(loc=0.5, scale=2.0).save(tmp_path / "meta")
    pipeline._persist_stats(ProfileStats(loc=0.5, scale=2.0))

    stored = FileIO.load_json(tmp_path / "meta" / ProfileStats.FILENAME)

    assert stored["scale"] == 2.0


def test_resumed_profile_run_refuses_to_overwrite_different_stats(tmp_path):
    """A resumed profile run raises ValueError when the recomputed statistics differ from the stored ones."""
    pipeline = _resumable_profile_pipeline(tmp_path, resume=True)

    ProfileStats(loc=0.5, scale=2.0).save(tmp_path / "meta")

    with pytest.raises(ValueError, match="profile normalization statistics"):
        pipeline._persist_stats(ProfileStats(loc=0.5, scale=5.0))


def test_fresh_profile_run_overwrites_stats(tmp_path):
    """A fresh profile run replaces any statistics left on disk."""
    pipeline = _resumable_profile_pipeline(tmp_path, resume=False)

    ProfileStats(loc=0.5, scale=2.0).save(tmp_path / "meta")
    pipeline._persist_stats(ProfileStats(loc=0.5, scale=5.0))

    stored = FileIO.load_json(tmp_path / "meta" / ProfileStats.FILENAME)

    assert stored["scale"] == 5.0


class _CapturingDatasetPipeline:
    """DatasetPipeline stand-in recording its construction keywords."""

    captured = {}

    def __init__(self, *args, **kwargs):
        """Records the keywords and exposes the profile length the callers read."""
        type(self).captured = kwargs
        self.profile_length = 16
        self.layout         = SimpleNamespace(profile_length=16)

    def run(self):
        """Returns an empty loader/dataset payload."""
        return None, None, None, {}


def test_backbone_preparation_threads_the_training_resume_flag(tmp_path, monkeypatch):
    """BackboneDatasetPreparation hands the trainer's resume flag to the dataset pipeline."""
    import pipelines.backbone.dataset.pipeline as module
    monkeypatch.setattr(module, "DatasetPipeline", _CapturingDatasetPipeline)

    gaussian       = SimpleNamespace(n_default_gaussians=3, x_min=-5.0, x_max=25.0)
    trainer_config = SimpleNamespace(gaussian=gaussian, training=SimpleNamespace(resume=True))
    dataset_config = SimpleNamespace(n_gaussians=None, x_axis=None)

    module.BackboneDatasetPreparation(dataset_config, trainer_config, tmp_path, _logger(tmp_path, "prep_resume"), seed=0).run()

    assert _CapturingDatasetPipeline.captured["resume"] is True


