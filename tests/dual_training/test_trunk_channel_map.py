"""Tests for mapping dual trunk input groups onto input stack channel indices.

Covers the amplitude, derived channel and aux field blocks for various input
configurations, multi-channel derived channel representations, the rejection of
unknown or empty selections, and group labelling.
"""

from __future__ import annotations

import pytest

from configuration.dataset            import InputConfig, Representation
from pipelines.dual.training.pipeline import TrunkChannelMap


def test_full_stack_four_tracks_maps_trailing_ifg_block():
    """The derived group maps to the trailing channels of a nine-channel full stack."""
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("derived",)) == (5, 6, 7, 8)


def test_all_groups_select_the_whole_stack():
    """Selecting both groups yields every channel of the stack."""
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("chan", "derived")) == tuple(range(9))


def test_pass_group_selects_amplitudes_only():
    """The pass group maps to the leading primary and secondary amplitude channels."""
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("chan",)) == (0, 1, 2, 3, 4)


def test_primary_and_ifgs_only_maps_after_primary():
    """Without secondaries the derived block starts right after the primary channel."""
    input_config = InputConfig()

    assert TrunkChannelMap.resolve(input_config, in_channels=5, groups=("derived",)) == (1, 2, 3, 4)


def test_dem_bearing_stack_routes_the_dem_only_where_requested():
    """The trailing aux channel reaches exactly the selections that name the dem group."""
    input_config         = InputConfig.full_stack()
    input_config.use_aux = True

    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("derived",))               == (5, 6, 7, 8)
    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("chan", "derived"))        == tuple(range(9))
    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("aux",))               == (9,)
    assert TrunkChannelMap.resolve(input_config, in_channels=10, groups=("chan", "derived", "aux")) == tuple(range(10))


def test_dem_group_without_dem_channel_contributes_nothing():
    """Selecting dem alongside other groups on a aux field-free stack changes nothing, alone it raises."""
    input_config = InputConfig.full_stack()

    assert TrunkChannelMap.resolve(input_config, in_channels=9, groups=("chan", "derived", "aux")) == tuple(range(9))

    with pytest.raises(ValueError, match="select no channels"):
        TrunkChannelMap.resolve(input_config, in_channels=9, groups=("aux",))


def test_multi_channel_ifg_representation_expands_the_block():
    """A two-channel derived representation widens the derived block accordingly."""
    input_config = InputConfig.full_stack()
    input_config.derived_representation = Representation.MAG_ANGLE

    assert TrunkChannelMap.resolve(input_config, in_channels=7, groups=("derived",)) == (3, 4, 5, 6)


def test_unknown_group_rejected():
    """An unrecognised group name raises ValueError."""
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.resolve(input_config, in_channels=9, groups=("bogus",))


def test_empty_selection_rejected():
    """A group that selects no channels in the current stack raises ValueError."""
    input_config = InputConfig.full_stack()
    input_config.use_derived = False

    with pytest.raises(ValueError, match="select no channels"):
        TrunkChannelMap.resolve(input_config, in_channels=5, groups=("derived",))


def test_inconsistent_channel_count_rejected():
    """A channel count inconsistent with any track count raises ValueError."""
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="unique track count"):
        TrunkChannelMap.resolve(input_config, in_channels=2, groups=("derived",))


def test_group_label_names_the_selection():
    """Group labels are order-independent, join partial selections and name the combined selection 'full'."""
    assert TrunkChannelMap.group_label(("chan", "derived", "aux")) == "full"
    assert TrunkChannelMap.group_label(("aux", "derived", "chan")) == "full"
    assert TrunkChannelMap.group_label(("chan", "derived"))        == "chan+derived"
    assert TrunkChannelMap.group_label(("derived", "chan"))        == "chan+derived"
    assert TrunkChannelMap.group_label(("chan",))              == "chan"
    assert TrunkChannelMap.group_label(("derived",))               == "derived"
    assert TrunkChannelMap.group_label(("aux",))               == "aux"


def test_group_label_rejects_unknown_groups():
    """Labelling an unrecognised group raises ValueError."""
    with pytest.raises(ValueError, match="Unknown input groups"):
        TrunkChannelMap.group_label(("kz",))


def test_group_label_rejects_empty_selection():
    """Labelling an empty group selection raises ValueError."""
    with pytest.raises(ValueError, match="empty trunk input"):
        TrunkChannelMap.group_label(())
