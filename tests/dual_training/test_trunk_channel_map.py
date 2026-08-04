from __future__ import annotations

import pytest

from configuration.dataset            import InputConfig, Representation
from pipelines.dual.training.pipeline import TrunkChannelMap


def test_full_stack_four_tracks_maps_trailing_ifg_block():
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("ifg",)) == (5, 6, 7, 8)


def test_all_groups_select_the_whole_stack():
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("pass", "ifg")) == tuple(range(9))


def test_pass_group_selects_amplitudes_only():
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("pass",)) == (0, 1, 2, 3, 4)


def test_primary_and_ifgs_only_maps_after_primary():
    input_config = InputConfig()

    assert TrunkChannelMap.resolve(input_config, in_channels=5, groups=("ifg",)) == (1, 2, 3, 4)


def test_dem_bearing_stack_never_reaches_a_trunk():
    input_config         = InputConfig.full_stack()
    input_config.use_dem = True

    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("ifg",))        == (5, 6, 7, 8)
    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("pass", "ifg")) == tuple(range(9))


def test_dem_group_rejected():
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.resolve(input_config, in_channels=9, groups=("dem",))


def test_multi_channel_ifg_representation_expands_the_block():
    input_config = InputConfig.full_stack()
    input_config.interferograms_representation = Representation.MAG_ANGLE

    assert TrunkChannelMap.resolve(input_config, in_channels=7, groups=("ifg",)) == (3, 4, 5, 6)


def test_unknown_group_rejected():
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.resolve(input_config, in_channels=9, groups=("kz",))


def test_empty_selection_rejected():
    input_config = InputConfig.full_stack()
    input_config.use_interferograms = False

    with pytest.raises(ValueError, match="select no channels"):
        TrunkChannelMap.resolve(input_config, in_channels=5, groups=("ifg",))


def test_inconsistent_channel_count_rejected():
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="unique track count"):
        TrunkChannelMap.resolve(input_config, in_channels=2, groups=("ifg",))


def test_group_label_names_the_selection():
    assert TrunkChannelMap.group_label(("pass", "ifg")) == "full"
    assert TrunkChannelMap.group_label(("ifg", "pass")) == "full"
    assert TrunkChannelMap.group_label(("pass",))       == "pass"
    assert TrunkChannelMap.group_label(("ifg",))        == "ifg"


def test_group_label_rejects_unknown_groups():
    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.group_label(("dem",))


def test_group_label_rejects_empty_selection():
    with pytest.raises(ValueError, match="empty trunk input"):
        TrunkChannelMap.group_label(())
