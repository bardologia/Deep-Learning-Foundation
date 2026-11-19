
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
    strategy = ChannelStrategy(NormMethod.ZSCORE)
    return ChannelStats(loc=[0.0] * n, scale=[1.0] * n, names=[f"c{i}" for i in range(n)], strategies=[strategy] * n, clampable=[False] * n)


def _identity_normalizer(n_inputs: int, n_outputs: int) -> Normalizer:
    return Normalizer(Stats(input_stats=_identity_channel_stats(n_inputs), output_stats=_identity_channel_stats(n_outputs)))


def _backbone_dataset(derived, parameters, split_name="train", n_derived=3):
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
    ds = _synthetic_dataset(n_gaussians=2, stored_gaussians=6, output_config=OutputConfig())

    assert ds.gt_selected.shape[0] == 6
    assert np.shares_memory(ds.gt_selected, ds.gt_parameters)

    _x, y = ds[0]

    np.testing.assert_allclose(y, ds.gt_parameters[:6, :8, :8])


@pytest.mark.dataset
def test_patchdataset_len_matches_grid(derived, parameters):
    ds, patcher = _backbone_dataset(derived, parameters)

    assert len(ds) == patcher.grid.number_of_patches
    assert len(ds) == 9


@pytest.mark.dataset
def test_patchdataset_sample_shapes_and_dtypes(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    x, y = ds[0]

    assert x.shape == (4, 8, 8)
    assert y.shape == (parameters.shape[0], 8, 8)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


@pytest.mark.dataset
def test_patchdataset_channel_counts(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    assert ds.input_channels == 4
    assert ds.gt_channels    == parameters.shape[0]
    assert ds.output_channel_indices == list(range(parameters.shape[0]))


@pytest.mark.dataset
def test_patchdataset_no_nan_in_any_sample(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    for idx in range(len(ds)):
        x, y = ds[idx]
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))




@pytest.mark.dataset
def test_patchdataset_angle_channels_in_pi_range(derived, parameters):
    ds, _ = _backbone_dataset(derived, parameters)

    x, _ = ds[0]

    assert np.all(np.abs(x) <= np.pi + 1e-4)


@pytest.mark.dataset
def test_patchdataset_layer_mismatch_raises(derived, parameters):
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
    part, _ = _backbone_dataset(derived, parameters)
    multi   = MultiRegionDataset([part])

    with pytest.raises(IndexError):
        multi[len(multi)]


def _profile_dataset(parameters, tmp_path, **kw):
    params = [np.ascontiguousarray(np.asarray(parameters[:, :40, :40]))]
    z_axis = np.linspace(0.0, 1.0, 150, dtype=np.float32)
    logger = Logger(log_dir=str(tmp_path / "logs"), name="prof_ds", level="ERROR")

    defaults = dict(n_gaussians=params[0].shape[0] // 3, split_name="train", activity_threshold=1e-3, keep_empty_frac=0.05, pixel_subsample=1.0, seed=0)
    defaults.update(kw)

    return ProfileDataset(param_arrays=params, z_axis=z_axis, logger=logger, **defaults)


@pytest.mark.dataset
def test_profiledataset_sample_shape_and_dtype(parameters, tmp_path):
    ds            = _profile_dataset(parameters, tmp_path)
    curve, target = ds[0]

    assert curve.shape == (150,)
    assert curve.dtype == np.float32
    np.testing.assert_array_equal(curve, target)


@pytest.mark.dataset
def test_profiledataset_no_nan_and_nonnegative(parameters, tmp_path):
    ds = _profile_dataset(parameters, tmp_path)

    for i in range(min(len(ds), 50)):
        curve, _ = ds[i]
        assert np.all(np.isfinite(curve))
        assert np.all(curve >= 0.0)


@pytest.mark.dataset
def test_profiledataset_noise_touches_only_the_input(parameters, tmp_path):
    config    = ProfileAugmentationConfig(p_flip=0.0, p_noise=1.0, noise_std=0.05)
    augmenter = ProfileAugmenter(config, logger=Logger(log_dir=str(tmp_path / "aug"), name="aug", level="ERROR"), seed=0)
    ds        = _profile_dataset(parameters, tmp_path, augmenter=augmenter)

    noisy, clean = ds[0]

    assert not np.array_equal(noisy, clean)
    assert np.all(clean >= 0.0)
    assert np.std(noisy - clean) == pytest.approx(0.05, rel=0.5)


@pytest.mark.dataset
def test_profiledataset_eval_split_ignores_the_augmenter(parameters, tmp_path):
    config    = ProfileAugmentationConfig(p_flip=1.0, p_noise=1.0, noise_std=0.05)
    augmenter = ProfileAugmenter(config, logger=Logger(log_dir=str(tmp_path / "aug"), name="aug", level="ERROR"), seed=0)
    ds        = _profile_dataset(parameters, tmp_path, augmenter=augmenter, split_name="val")

    noisy, clean = ds[0]

    np.testing.assert_array_equal(noisy, clean)


@pytest.mark.dataset
def test_profiledataset_len_consistent_with_index(parameters, tmp_path):
    ds = _profile_dataset(parameters, tmp_path)

    assert len(ds) == ds.index.shape[0]
    assert len(ds) <= ds.amps.shape[0]
    assert ds.amps.shape[0] == 40 * 40


@pytest.mark.dataset
def test_profiledataset_full_subsample_keeps_all_active(parameters, tmp_path):
    ds = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=1.0)

    assert len(ds) == ds.n_active


@pytest.mark.dataset
def test_profiledataset_subsample_reduces_active(parameters, tmp_path):
    full = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=1.0)
    half = _profile_dataset(parameters, tmp_path, keep_empty_frac=0.0, pixel_subsample=0.5)

    assert len(half) < len(full)


@pytest.mark.dataset
def test_profiledataset_index_deterministic_for_seed(parameters, tmp_path):
    a = _profile_dataset(parameters, tmp_path, seed=11)
    b = _profile_dataset(parameters, tmp_path, seed=11)

    np.testing.assert_array_equal(a.index, b.index)


@pytest.mark.dataset
def test_profiledataset_index_changes_with_seed(parameters, tmp_path):
    a = _profile_dataset(parameters, tmp_path, seed=1, pixel_subsample=0.5, keep_empty_frac=0.0)
    b = _profile_dataset(parameters, tmp_path, seed=2, pixel_subsample=0.5, keep_empty_frac=0.0)

    assert not np.array_equal(a.index, b.index)


@pytest.mark.parametrize("pixel_subsample", [0.5, 1.0])
def test_profiledataset_without_active_pixels_names_the_split_and_threshold(tmp_path, pixel_subsample):
    empty = np.zeros((15, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="activity threshold"):
        _profile_dataset(empty, tmp_path, pixel_subsample=pixel_subsample)


def _sparse_cube(n_active: int = 4) -> np.ndarray:
    cube = np.zeros((15, 8, 8), dtype=np.float32)
    flat = cube.reshape(15, -1)

    flat[0, :n_active] = 1.0
    flat[1, :n_active] = 0.5
    flat[2, :n_active] = 0.1

    return cube


@pytest.mark.parametrize("pixel_subsample", [0.0, -0.5, 1.5])
def test_profiledataset_rejects_pixel_subsample_outside_its_range(tmp_path, pixel_subsample):
    with pytest.raises(ValueError, match="pixel_subsample must lie in"):
        _profile_dataset(_sparse_cube(), tmp_path, pixel_subsample=pixel_subsample)


@pytest.mark.parametrize("keep_empty_frac", [-0.1, 1.5])
def test_profiledataset_rejects_keep_empty_frac_outside_its_range(tmp_path, keep_empty_frac):
    with pytest.raises(ValueError, match="keep_empty_frac must lie in"):
        _profile_dataset(_sparse_cube(), tmp_path, keep_empty_frac=keep_empty_frac)


def test_profiledataset_counts_the_kept_active_and_empty_pixels(tmp_path):
    ds = _profile_dataset(_sparse_cube(n_active=4), tmp_path, pixel_subsample=0.5, keep_empty_frac=0.5)

    assert ds.n_active      == 4
    assert ds.n_kept_active == 2
    assert ds.n_kept_empty  == 30
    assert len(ds)          == 32


def test_profiledataset_keeps_every_active_pixel_at_full_subsample_and_no_empty_at_zero(tmp_path):
    ds = _profile_dataset(_sparse_cube(n_active=4), tmp_path, pixel_subsample=1.0, keep_empty_frac=0.0)

    assert ds.n_kept_active == 4
    assert ds.n_kept_empty  == 0
    assert len(ds)          == 4


def _stack_shim(enabled: set[str], views: dict) -> SimpleNamespace:
    def _refuse(artifact):
        raise AssertionError(f"the stack read the {artifact} artifact although its group is disabled")

    return SimpleNamespace(
        _group_enabled = lambda flag: flag in enabled,
        _group_view    = lambda artifact, flag, row_slice, col_slice: views.get(artifact),
        layout         = SimpleNamespace(artifact_path=_refuse),
    )


def test_input_stack_without_complex_groups_is_empty_over_the_region():
    stack, n_primary, n_secondaries, n_derived = Cropper._stack_inputs(_stack_shim(set(), {}), slice(2, 6), slice(1, 4))

    assert stack.shape == (0, 4, 3)
    assert (n_primary, n_secondaries, n_derived) == (0, 0, 0)


def test_input_stack_skips_the_primary_artifact_when_its_group_is_off():
    secondaries = (np.arange(2 * 4 * 3) + 1j).reshape(2, 4, 3).astype(np.complex64)

    stack, n_primary, n_secondaries, n_derived = Cropper._stack_inputs(_stack_shim({"use_secondaries"}, {"secondaries": secondaries}), slice(2, 6), slice(1, 4))

    assert (n_primary, n_secondaries, n_derived) == (0, 2, 0)
    np.testing.assert_array_equal(stack, secondaries)
