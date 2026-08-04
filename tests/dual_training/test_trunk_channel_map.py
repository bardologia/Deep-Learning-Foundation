from __future__ import annotations

import pytest

from configuration.dataset            import InputConfig, Representation
from pipelines.dual.training.pipeline import TrunkChannelMap


def test_full_stack_four_channels_maps_the_trailing_derived_block():
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("derived",)) == (5, 6, 7, 8)


def test_all_groups_select_the_whole_stack():
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("chan", "derived")) == tuple(range(9))


def test_pass_group_selects_amplitudes_only():
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("chan",)) == (0, 1, 2, 3, 4)


def test_primary_and_derived_only_maps_after_primary():
    input_config = InputConfig()

    assert TrunkChannelMap.resolve(input_config, in_channels=5, groups=("derived",)) == (1, 2, 3, 4)


def test_dem_bearing_stack_never_reaches_a_trunk():
    input_config         = InputConfig.full_stack()
    input_config.use_aux = True

    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("derived",))        == (5, 6, 7, 8)
    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("chan", "derived")) == tuple(range(9))


def test_dem_group_rejected():
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.resolve(input_config, in_channels=9, groups=("aux",))


def test_multi_channel_derived_representation_expands_the_block():
    input_config                        = InputConfig.full_stack()
    input_config.derived_representation = Representation.MAG_ANGLE

    assert TrunkChannelMap.resolve(input_config, in_channels=7, groups=("derived",)) == (3, 4, 5, 6)


def test_unknown_group_rejected():
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.resolve(input_config, in_channels=9, groups=("bogus",))


def test_empty_selection_rejected():
    input_config             = InputConfig.full_stack()
    input_config.use_derived = False

    with pytest.raises(ValueError, match="select no channels"):
        TrunkChannelMap.resolve(input_config, in_channels=5, groups=("derived",))


def test_inconsistent_channel_count_rejected():
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="unique track count"):
        TrunkChannelMap.resolve(input_config, in_channels=2, groups=("derived",))


def test_group_label_names_the_selection():
    assert TrunkChannelMap.group_label(("chan", "derived")) == "full"
    assert TrunkChannelMap.group_label(("derived", "chan")) == "full"
    assert TrunkChannelMap.group_label(("chan",))       == "chan"
    assert TrunkChannelMap.group_label(("derived",))        == "derived"


def test_group_label_rejects_unknown_groups():
    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.group_label(("aux",))


def test_group_label_rejects_empty_selection():
    with pytest.raises(ValueError, match="empty trunk input"):
        TrunkChannelMap.group_label(())
