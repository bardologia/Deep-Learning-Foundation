"""Tests for the patch and profile datasets that feed the training pipelines."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from configuration.dataset                              import InputConfig, OutputConfig, ProfileAugmentationConfig, Representation
from configuration.normalization                        import ChannelStats, ChannelStrategy, NormMethod
from pipelines.backbone.dataset.datasets                import MultiRegionDataset, PatchDataset
from pipelines.backbone.dataset.normalizer              import Normalizer
from pipelines.backbone.dataset.spatial                 import Cropper, Patcher
from pipelines.backbone.dataset.stats                   import Stats
from pipelines.profile_autoencoder.dataset.augmentation import ProfileAugmenter
from pipelines.profile_autoencoder.dataset.datasets     import ProfileDataset
from tools.monitoring.logger                            import Logger


def _identity_channel_stats(n: int) -> ChannelStats:
    """Returns z-score channel statistics that leave each of the n channels unchanged."""
    strategy = ChannelStrategy(NormMethod.ZSCORE)
    return ChannelStats(loc=[0.0] * n, scale=[1.0] * n, names=[f"c{i}" for i in range(n)], strategies=[strategy] * n, clampable=[False] * n)


def _identity_normalizer(n_inputs: int, n_outputs: int) -> Normalizer:
    """Returns a normalizer that is the identity on n_inputs input and n_outputs output channels."""
    return Normalizer(Stats(input_stats=_identity_channel_stats(n_inputs), output_stats=_identity_channel_stats(n_outputs)))


def _backbone_dataset(derived, parameters, split_name="train", n_derived=3):
    """Builds a PatchDataset over a 24x24 crop with an identity normalizer attached.

    Args:
        derived    : Complex derived-channel stack of shape (channels, rows, cols).
        parameters : Gaussian parameter cube of shape (3 * n_gaussians, rows, cols).
        split_name : Split the dataset belongs to, which gates augmentation.
        n_derived  : Number of derived layers to keep.

    Returns:
        Tuple of the dataset and the Patcher describing its 8x8 patch grid.
    """
    derived = np.ascontiguousarray(np.asarray(derived[:n_derived, :24, :24]))
    primary = derived[:1]
    inputs  = np.concatenate([primary, derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :24, :24]))

    patcher = Patcher.build(spatial_size=(24, 24), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_primary=True, primary_representation=Representation.MAG_ONLY,
                          use_secondaries=False,
                          use_derived=True, derived_representation=Representation.ANGLE_ONLY)

    ds = PatchDataset(
        inputs=inputs, gt_parameters=params, grid=patcher,
        input_config=ic, output_config=OutputConfig(), split_name=split_name,
        n_secondaries=0, n_derived=n_derived, n_gaussians=params.shape[0] // 3,
    )
    ds.normalizer = _identity_normalizer(ds.input_channels, ds.gt_channels)

    return ds, patcher


def _synthetic_dataset(n_gaussians: int, stored_gaussians: int, output_config: OutputConfig) -> PatchDataset:
    """Builds a PatchDataset over random 16x16 inputs and a ramp parameter cube.

    Args:
        n_gaussians      : Number of Gaussian slots the dataset exposes.
        stored_gaussians : Number of Gaussian slots present in the stored parameter cube.
        output_config    : Output channel selection under test.

    Returns:
        The dataset with an identity normalizer attached.
    """
    rng     = np.random.default_rng(0)
    inputs  = rng.standard_normal((2, 16, 16)).astype(np.complex64)
    params  = np.arange(stored_gaussians * 3 * 16 * 16, dtype=np.float32).reshape(stored_gaussians * 3, 16, 16)
    patcher = Patcher.build(spatial_size=(16, 16), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_primary=True, primary_representation=Representation.MAG_ONLY,
                          use_secondaries=True, secondaries_representation=Representation.MAG_ONLY,
                          use_derived=False)

    ds = PatchDataset(
        inputs=inputs, gt_parameters=params, grid=patcher,
        input_config=ic, output_config=output_config, split_name="val",
        n_secondaries=1, n_derived=0, n_gaussians=n_gaussians,
    )
    ds.normalizer = _identity_normalizer(ds.input_channels, ds.gt_channels)

    return ds


def test_patchdataset_selects_gt_channels_before_patching():
    """Ground-truth channel selection is a view of the stored cube and precedes patch extraction."""
    ds = _synthetic_dataset(n_gaussians=2, stored_gaussians=6, output_config=OutputConfig())

    assert ds.gt_selected.shape[0] == 6
    assert np.shares_memory(ds.gt_selected, ds.gt_parameters)

    _x, y = ds[0]

    np.testing.assert_allclose(y, ds.gt_parameters[:6, :8, :8])


@pytest.mark.dataset
def test_patchdataset_len_matches_grid(derived, parameters):
    """Dataset length equals the number of patches in the grid."""
    ds, patcher = _backbone_dataset(derived, parameters)

    assert len(ds) == patcher.grid.number_of_patches
    assert len(ds) == 9


@pytest.mark.dataset
def test_patchdataset_sample_shapes_and_dtypes(derived, parameters):
    """Samples carry the configured channel counts, the patch size, and float32 dtype."""
    ds, _ = _backbone_dataset(derived, parameters)

    x, y = ds[0]

    assert x.shape == (4, 8, 8)
    assert y.shape == (parameters.shape[0], 8, 8)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


@pytest.mark.dataset
def test_patchdataset_channel_counts(derived, parameters):
    """Input, ground-truth, and selected output channel counts follow the input and output configs."""
    ds, _ = _backbone_dataset(derived, parameters)

    assert ds.input_channels == 4
    assert ds.gt_channels    == parameters.shape[0]
    assert ds.output_channel_indices == list(range(parameters.shape[0]))


@pytest.mark.dataset
def test_patchdataset_no_nan_in_any_sample(derived, parameters):
    """Every sample in the dataset is finite in both input and target."""
    ds, _ = _backbone_dataset(derived, parameters)

    for idx in range(len(ds)):
        x, y = ds[idx]
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))




@pytest.mark.dataset
def test_patchdataset_angle_channels_in_pi_range(derived, parameters):
    """Angle-only representation channels stay within plus or minus pi radians."""
    ds, _ = _backbone_dataset(derived, parameters)

    x, _ = ds[0]

    assert np.all(np.abs(x) <= np.pi + 1e-4)


@pytest.mark.dataset
def test_patchdataset_layer_mismatch_raises(derived, parameters):
    """A declared derived count larger than the stack raises ValueError."""
    derived     = np.ascontiguousarray(np.asarray(derived[:3, :16, :16]))
    inputs  = np.concatenate([derived[:1], derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :16, :16]))
    patcher = Patcher.build(spatial_size=(16, 16), patch_size=(8, 8), stride=(8, 8))

    with pytest.raises(ValueError):
        PatchDataset(
            inputs=inputs, gt_parameters=params, grid=patcher,
            input_config=InputConfig(), output_config=OutputConfig(), split_name="train",
            n_secondaries=0, n_derived=99, n_gaussians=params.shape[0] // 3,
        )


@pytest.mark.dataset
def test_patchdataset_use_aux_without_aux_raises(derived, parameters):
    """Requesting the aux channel without supplying an aux field raises ValueError."""
    derived     = np.ascontiguousarray(np.asarray(derived[:3, :16, :16]))
    inputs  = np.concatenate([derived[:1], derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :16, :16]))
    patcher = Patcher.build(spatial_size=(16, 16), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_aux=True)

    with pytest.raises(ValueError, match="Aux"):
        PatchDataset(
            inputs=inputs, gt_parameters=params, grid=patcher,
            input_config=ic, output_config=OutputConfig(), split_name="train",
            n_secondaries=0, n_derived=3, n_gaussians=params.shape[0] // 3,
        )


@pytest.mark.dataset
def test_patchdataset_dem_fills_last_channel(derived, parameters):
    """The Aux occupies the final input channel and is patched on the same grid."""
    derived     = np.ascontiguousarray(np.asarray(derived[:3, :24, :24]))
    inputs  = np.concatenate([derived[:1], derived], axis=0)
    params  = np.ascontiguousarray(np.asarray(parameters[:, :24, :24]))
    aux     = np.random.default_rng(1).standard_normal((24, 24)).astype(np.float32)
    patcher = Patcher.build(spatial_size=(24, 24), patch_size=(8, 8), stride=(8, 8))
    ic      = InputConfig(use_primary=True, primary_representation=Representation.MAG_ONLY,
                          use_secondaries=False,
                          use_derived=True, derived_representation=Representation.ANGLE_ONLY,
                          use_aux=True)

    ds = PatchDataset(
        inputs=inputs, gt_parameters=params, grid=patcher,
        input_config=ic, output_config=OutputConfig(), split_name="val",
        n_secondaries=0, n_derived=3, n_gaussians=params.shape[0] // 3, aux=aux,
    )
    ds.normalizer = _identity_normalizer(ds.input_channels, ds.gt_channels)

    x, _ = ds[0]

    assert x.shape == (5, 8, 8)
    np.testing.assert_array_equal(x[-1], patcher.extract(aux, 0))


@pytest.mark.dataset
def test_multiregion_dataset_concatenates_parts(derived, parameters):
    """MultiRegionDataset length is the sum of its parts and samples keep their shape."""
    part_a, _ = _backbone_dataset(derived, parameters)
    part_b, _ = _backbone_dataset(derived, parameters)

    multi = MultiRegionDataset([part_a, part_b])

    assert len(multi) == len(part_a) + len(part_b)

    first = multi[0]
    last  = multi[len(multi) - 1]

    assert first[0].shape == (4, 8, 8)
    assert last[0].shape  == (4, 8, 8)


@pytest.mark.dataset
def test_multiregion_index_out_of_range_raises(derived, parameters):
    """Indexing past the concatenated length raises IndexError."""
    part, _ = _backbone_dataset(derived, parameters)
    multi   = MultiRegionDataset([part])

    with pytest.raises(IndexError):
        multi[len(multi)]


def _profile_dataset(parameters, tmp_path, **kw):
    """Builds a ProfileDataset over a 40x40 parameter crop on a 150-bin height axis.

    Args:
        parameters : Gaussian parameter cube of shape (3 * n_gaussians, rows, cols).
        tmp_path   : Directory the logger writes into.
        **kw       : Overrides for the ProfileDataset keyword defaults.

    Returns:
        The constructed ProfileDataset.
    """
    params = [np.ascontiguousarray(np.asarray(parameters[:, :40, :40]))]
    z_axis = np.linspace(0.0, 1.0, 150, dtype=np.float32)
    logger = Logger(log_dir=str(tmp_path / "logs"), name="prof_ds", level="ERROR")

    defaults = dict(n_gaussians=params[0].shape[0] // 3, split_name="train", activity_threshold=1e-3, keep_empty_frac=0.05, pixel_subsample=1.0, seed=0)
    defaults.update(kw)

    return ProfileDataset(param_arrays=params, z_axis=z_axis, logger=logger, **defaults)


@pytest.mark.dataset
def test_profiledataset_sample_shape_and_dtype(parameters, tmp_path):
    """Samples are float32 curves of the x-axis length, input equal to target when unaugmented."""
    ds            = _profile_dataset(parameters, tmp_path)
    curve, target = ds[0]

    assert curve.shape == (150,)
    assert curve.dtype == np.float32
    np.testing.assert_array_equal(curve, target)


@pytest.mark.dataset
def test_profiledataset_no_nan_and_nonnegative(parameters, tmp_path):
    """Reconstructed profiles are finite and non-negative."""
    ds = _profile_dataset(parameters, tmp_path)

    for i in range(min(len(ds), 50)):
        curve, _ = ds[i]
        assert np.all(np.isfinite(curve))
        assert np.all(curve >= 0.0)


@pytest.mark.dataset
def test_profiledataset_noise_touches_only_the_input(parameters, tmp_path):
    """Noise augmentation perturbs the input curve at the configured std while the target stays clean."""
    config    = ProfileAugmentationConfig(p_flip=0.0, p_noise=1.0, noise_std=0.05)
    augmenter = ProfileAugmenter(config, logger=Logger(log_dir=str(tmp_path / "aug"), name="aug", level="ERROR"), seed=0)
    ds        = _profile_dataset(parameters, tmp_path, augmenter=augmenter)

    noisy, clean = ds[0]

    assert not np.array_equal(noisy, clean)
    assert np.all(clean >= 0.0)
    assert np.std(noisy - clean) == pytest.approx(0.05, rel=0.5)


@pytest.mark.dataset
def test_profiledataset_eval_split_ignores_the_augmenter(parameters, tmp_path):
    """On a val split the augmenter is bypassed and input equals target."""
    config    = ProfileAugmentationConfig(p_flip=1.0, p_noise=1.0, noise_std=0.05)
    augmenter = ProfileAugmenter(config, logger=Logger(log_dir=str(tmp_path / "aug"), name="aug", level="ERROR"), seed=0)
    ds        = _profile_dataset(parameters, tmp_path, augmenter=augmenter, split_name="val")

    noisy, clean = ds[0]

    np.testing.assert_array_equal(noisy, clean)


@pytest.mark.dataset
def test_profiledataset_len_consistent_with_index(parameters, tmp_path):
    """Dataset length matches the kept index and never exceeds the pixel count."""
    ds = _profile_dataset(parameters, tmp_path)

    assert len(ds) == ds.index.shape[0]
    assert len(ds) <= ds.amps.shape[0]
    assert ds.amps.shape[0] == 40 * 40


@pytest.mark.dataset
def test_profiledataset_full_subsample_keeps_all_active(parameters, tmp_path):
    """With no empty pixels kept and full subsampling the length equals the active pixel count."""
    ds = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=1.0)

    assert len(ds) == ds.n_active


@pytest.mark.dataset
def test_profiledataset_subsample_reduces_active(parameters, tmp_path):
    """Halving pixel_subsample shortens the dataset."""
    full = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=1.0)
    half = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=0.5)

    assert len(half) < len(full)


@pytest.mark.dataset
def test_profiledataset_index_deterministic_for_seed(parameters, tmp_path):
    """Two datasets built with the same seed select the same pixel index."""
    a = _profile_dataset(parameters, tmp_path, seed=11)
    b = _profile_dataset(parameters, tmp_path, seed=11)

    np.testing.assert_array_equal(a.index, b.index)


@pytest.mark.dataset
def test_profiledataset_index_changes_with_seed(parameters, tmp_path):
    """Different seeds select different subsampled pixel indices."""
    a = _profile_dataset(parameters, tmp_path, seed=1, pixel_subsample=0.5, keep_empty_frac=0.0)
    b = _profile_dataset(parameters, tmp_path, seed=2, pixel_subsample=0.5, keep_empty_frac=0.0)

    assert not np.array_equal(a.index, b.index)


@pytest.mark.parametrize("pixel_subsample", [0.5, 1.0])
def test_profiledataset_without_active_pixels_names_the_split_and_threshold(tmp_path, pixel_subsample):
    """An all-zero parameter cube raises a ValueError naming the amplitude threshold."""
    empty = np.zeros((15, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="activity threshold"):
        _profile_dataset(empty, tmp_path, pixel_subsample=pixel_subsample)


def _sparse_cube(n_active: int = 4) -> np.ndarray:
    """Returns a 5-Gaussian parameter cube over 8 x 8 pixels whose first n_active pixels carry one component."""
    cube = np.zeros((15, 8, 8), dtype=np.float32)
    flat = cube.reshape(15, -1)

    flat[0, :n_active] = 1.0
    flat[1, :n_active] = 0.5
    flat[2, :n_active] = 0.1

    return cube


@pytest.mark.parametrize("pixel_subsample", [0.0, -0.5, 1.5])
def test_profiledataset_rejects_pixel_subsample_outside_its_range(tmp_path, pixel_subsample):
    """A pixel_subsample outside (0, 1] raises instead of silently keeping every active pixel."""
    with pytest.raises(ValueError, match="pixel_subsample must lie in"):
        _profile_dataset(_sparse_cube(), tmp_path, pixel_subsample=pixel_subsample)


@pytest.mark.parametrize("keep_empty_frac", [-0.1, 1.5])
def test_profiledataset_rejects_keep_empty_frac_outside_its_range(tmp_path, keep_empty_frac):
    """A keep_empty_frac outside [0, 1] raises."""
    with pytest.raises(ValueError, match="keep_empty_frac must lie in"):
        _profile_dataset(_sparse_cube(), tmp_path, keep_empty_frac=keep_empty_frac)


def test_profiledataset_counts_the_kept_active_and_empty_pixels(tmp_path):
    """The dataset exposes how many active and empty pixels it kept, and their sum is its length."""
    ds = _profile_dataset(_sparse_cube(n_active=4), tmp_path, pixel_subsample=0.5, keep_empty_frac=0.5)

    assert ds.n_active      == 4
    assert ds.n_kept_active == 2
    assert ds.n_kept_empty  == 30
    assert len(ds)          == 32


def test_profiledataset_keeps_every_active_pixel_at_full_subsample_and_no_empty_at_zero(tmp_path):
    """The range bounds 1.0 and 0.0 keep all active pixels and no empty pixel."""
    ds = _profile_dataset(_sparse_cube(n_active=4), tmp_path, pixel_subsample=1.0, keep_empty_frac=0.0)

    assert ds.n_kept_active == 4
    assert ds.n_kept_empty  == 0
    assert len(ds)          == 4


def _stack_shim(enabled: set[str], views: dict) -> SimpleNamespace:
    """Returns a Cropper stand-in whose groups and views are fixed and whose layout refuses every artifact read.

    Args:
        enabled : Input flags the stand-in reports as enabled.
        views   : Cropped view returned per artifact group, None for a disabled group.
    """
    def _refuse(artifact):
        """Fails the test when the stack reads an artifact directly."""
        raise AssertionError(f"the stack read the {artifact} artifact although its group is disabled")

    return SimpleNamespace(
        _group_enabled = lambda flag: flag in enabled,
        _group_view    = lambda artifact, flag, row_slice, col_slice: views.get(artifact),
        layout         = SimpleNamespace(artifact_path=_refuse),
    )


def test_input_stack_without_complex_groups_is_empty_over_the_region():
    """With no complex group enabled, as an aux-only input selects, the stack has zero layers over the region and no artifact is read."""
    stack, n_primary, n_secondaries, n_derived = Cropper._stack_inputs(_stack_shim(set(), {}), slice(2, 6), slice(1, 4))

    assert stack.shape == (0, 4, 3)
    assert (n_primary, n_secondaries, n_derived) == (0, 0, 0)


def test_input_stack_skips_the_primary_artifact_when_its_group_is_off():
    """With the primary group off the stack takes its shape from the secondaries and never reads the primary artifact."""
    secondaries = (np.arange(2 * 4 * 3) + 1j).reshape(2, 4, 3).astype(np.complex64)

    stack, n_primary, n_secondaries, n_derived = Cropper._stack_inputs(_stack_shim({"use_secondaries"}, {"secondaries": secondaries}), slice(2, 6), slice(1, 4))

    assert (n_primary, n_secondaries, n_derived) == (0, 2, 0)
    np.testing.assert_array_equal(stack, secondaries)
