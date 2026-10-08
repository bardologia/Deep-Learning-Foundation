"""Tests for the model table of the profile-autoencoder run loader.

Covers the rows describing the scored curve set and the settings that selected it: the kept
active and empty counts, the activity threshold, the sampling fractions and seed, and the
effective batch size with its origin.
"""

from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

from pipelines.profile_autoencoder.inference.loader import ProfileAeRunLoader


class _Dataset:
    """Dataset stub of 150 curves, 120 active and 30 empty.

    Attributes:
        n_kept_active : Active curves kept.
        n_kept_empty  : Empty curves kept.
    """

    n_kept_active = 120
    n_kept_empty  = 30

    def __len__(self) -> int:
        """Returns the curve count."""
        return 150


def _config(batch_size):
    """Returns an inference-config stub with the sampling settings and the given batch-size override."""
    return SimpleNamespace(split="test", pixel_subsample=0.5, keep_empty_frac=0.05, seed=3, batch_size=batch_size)


def test_describe_lists_the_scored_curve_set_and_its_selection_settings():
    """The table names the kept active and empty curves, the threshold, the sampling fractions and the seed."""
    rows = ProfileAeRunLoader._describe(Path("/runs/ae/best_model.pt"), 16, 150, _Dataset(), _config(batch_size=None), 1024, 1e-3)

    assert rows["Curves"]             == "150 (120 active + 30 empty)"
    assert rows["Activity threshold"].startswith("0.001 ")
    assert rows["Pixel subsample"]    == "0.5 of the active pixels"
    assert rows["Keep empty frac"]    == "0.05 of the empty pixels"
    assert rows["Selection seed"]     == 3
    assert rows["Split"]              == "test"


def test_describe_states_where_the_batch_size_came_from():
    """The batch-size row says whether the inference override or the training batch size applied."""
    training = ProfileAeRunLoader._describe(Path("best_model.pt"), 16, 150, _Dataset(), _config(batch_size=None), 1024, 1e-3)
    override = ProfileAeRunLoader._describe(Path("best_model.pt"), 16, 150, _Dataset(), _config(batch_size=256),  256,  1e-3)

    assert training["Batch size"] == "1024 (training batch size)"
    assert override["Batch size"] == "256 (inference override)"
