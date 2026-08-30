"""Tests that autoencoder run directories carry their architecture name exactly once.

Covers the timestamped name a single-seed run resolves to, the suffix an explicit run name
contributes, and the seed-fanout parent directory name for both autoencoder families.
"""

from __future__ import annotations

import re

from configuration.training                          import ImageAeEntryConfig
from configuration.training.profile_autoencoder      import ProfileAeEntryConfig
from pipelines.image_autoencoder.training.pipeline   import SingleTrainRunner as ImageAeRunner
from pipelines.profile_autoencoder.training.pipeline import SingleTrainRunner as ProfileAeRunner


TIMESTAMPED = re.compile(r"^(?P<architecture>.+)_\d{8}_\d{6}$")


def test_profile_ae_single_seed_run_name_carries_the_architecture():
    """A single-seed profile run resolves to the architecture name plus a timestamp."""
    runner = ProfileAeRunner(ProfileAeEntryConfig(ae_model_name="mlp_ae"))

    match = TIMESTAMPED.match(runner._resolve_run_name())

    assert match is not None
    assert match.group("architecture") == "mlp_ae"


def test_image_ae_single_seed_run_name_carries_the_architecture():
    """A single-seed image run resolves to the architecture name plus a timestamp."""
    runner = ImageAeRunner(ImageAeEntryConfig(ae_model_name="conv2d_ae"))

    match = TIMESTAMPED.match(runner._resolve_run_name())

    assert match is not None
    assert match.group("architecture") == "conv2d_ae"


def test_explicit_run_name_is_appended_to_the_architecture():
    """An explicit run name replaces the timestamp and is appended to the architecture name."""
    runner = ProfileAeRunner(ProfileAeEntryConfig(ae_model_name="gru_ae", run_name="ablation"))

    assert runner._resolve_run_name() == "gru_ae_ablation"


def test_profile_ae_seed_fanout_directory_carries_the_architecture_once(tmp_path):
    """A profile seed fanout names its parent directory with the architecture and a timestamp, not twice over."""
    from pipelines.shared.training.seed_sweep import SeedFanoutScheduler

    config        = ProfileAeEntryConfig(ae_model_name="mlp_ae", seeds=[0, 1])
    config.logdir = tmp_path

    scheduler = SeedFanoutScheduler.for_runner(config, {}, tmp_path / "train.py", ProfileAeRunner)

    match = TIMESTAMPED.match(scheduler.run_dir.name)
    assert match is not None
    assert match.group("architecture") == "mlp_ae"


def test_image_ae_seed_fanout_directory_carries_the_architecture_once(tmp_path):
    """An image seed fanout names its parent directory with the architecture and a timestamp, not twice over."""
    from pipelines.shared.training.seed_sweep import SeedFanoutScheduler

    config        = ImageAeEntryConfig(ae_model_name="conv2d_ae", seeds=[0, 1])
    config.logdir = tmp_path

    scheduler = SeedFanoutScheduler.for_runner(config, {}, tmp_path / "train.py", ImageAeRunner)

    match = TIMESTAMPED.match(scheduler.run_dir.name)
    assert match is not None
    assert match.group("architecture") == "conv2d_ae"
