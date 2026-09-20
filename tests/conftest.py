"""Shared pytest fixtures, logging stubs and a recording TensorBoard writer for the whole test suite.

Session-scoped fixtures expose the gitignored test_data/ tree (preprocessed stacks,
metadata and extracted Gaussian parameters); they skip when that tree is absent.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.monitoring.logger import Logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["MLTEMPLATE_MACHINE_PROFILE"] = ""

from configuration.synthesis import SynthesisConfig

_SCENE_CONFIG = SynthesisConfig()


class SilentProgress:
    """Progress-bar stub that accepts task calls and does nothing.

    Stands in for the Rich progress handle yielded by Logger.track.
    """

    def add_task(self, *args, **kwargs):
        """Returns a fixed task identifier of zero."""
        return 0

    def advance(self, *args, **kwargs):
        """Ignores a progress advance."""
        pass

    def update(self, *args, **kwargs):
        """Ignores a progress update."""
        pass


class SilentLogger:
    """Logger stub whose reporting methods are all no-ops.

    Mirrors the public surface of tools.monitoring.logger.Logger so pipelines under
    test can log without producing output or files.
    """

    def section(self, *args, **kwargs):       pass
    def subsection(self, *args, **kwargs):    pass
    def debug(self, *args, **kwargs):         pass
    def info(self, *args, **kwargs):          pass
    def warning(self, *args, **kwargs):       pass
    def error(self, *args, **kwargs):         pass
    def critical(self, *args, **kwargs):      pass
    def ok(self, *args, **kwargs):            pass
    def render(self, *args, **kwargs):        pass
    def panel(self, *args, **kwargs):         pass
    def rule(self, *args, **kwargs):          pass
    def kv_table(self, *args, **kwargs):      pass
    def metrics_table(self, *args, **kwargs): pass
    def close(self, *args, **kwargs):         pass

    @contextlib.contextmanager
    def timer(self, label: str):
        """Yields control without timing the labelled block.

        Args:
            label : Ignored name of the timed section.
        """
        yield

    @contextlib.contextmanager
    def track(self, transient: bool = False):
        """Yields a SilentProgress in place of a live progress bar.

        Args:
            transient : Ignored flag for transient rendering.

        Yields:
            A progress stub that discards all task updates.
        """
        yield SilentProgress()

    progress_bar = track

    @contextlib.contextmanager
    def live_monitor(self, title: str = "Training Monitor"):
        """Yields a SilentProgress in place of the live training monitor.

        Args:
            title : Ignored monitor title.

        Yields:
            A progress stub that discards all task updates.
        """
        yield SilentProgress()


class RecordingWriter:
    """TensorBoard writer stub capturing every write.

    Attributes:
        scalars    : Tuples of (tag, value, step).
        histograms : Tuples of (tag, values array, step, bins).
        figures    : Tuples of (tag, figure, step, close).
        layouts    : Custom Scalars layouts written.
        flushed    : Number of flush calls.
        closed     : Number of close calls.
    """

    def __init__(self):
        """Creates the empty capture buffers and counters."""
        self.scalars    = []
        self.histograms = []
        self.figures    = []
        self.layouts    = []
        self.flushed    = 0
        self.closed     = 0

    def add_scalar(self, tag, value, step):
        """Records a scalar write."""
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, values, step, bins="auto"):
        """Records a histogram write, storing the values as an array."""
        self.histograms.append((tag, np.asarray(values), step, bins))

    def add_figure(self, tag, figure, step, close=True):
        """Records a figure write together with its close flag."""
        self.figures.append((tag, figure, step, close))

    def add_custom_scalars(self, layout):
        """Records a Custom Scalars layout write."""
        self.layouts.append(layout)

    def flush(self):
        """Counts a flush."""
        self.flushed += 1

    def close(self):
        """Counts a close."""
        self.closed += 1


class RecordingLogger(SilentLogger):
    """Logger stub that records the messages it is asked to emit.

    Attributes:
        messages : List of (level, message) pairs in emission order.
    """

    def __init__(self):
        """Initialises an empty message log."""
        self.messages = []

    def section(self, msg):
        """Records a section heading."""
        self.messages.append(("section", msg))

    def subsection(self, msg):
        """Records a subsection heading."""
        self.messages.append(("subsection", msg))

    def ok(self, msg):
        """Records a success message."""
        self.messages.append(("ok", msg))

    def info(self, msg):
        """Records an informational message."""
        self.messages.append(("info", msg))

    def warning(self, msg):
        """Records a warning message."""
        self.messages.append(("warning", msg))

    def error(self, msg):
        """Records an error message."""
        self.messages.append(("error", msg))

    def metrics_table(self, rows, columns=None, title=None):
        """Records only the title of a metrics table, discarding the rows."""
        self.messages.append(("metrics_table", title))


def make_file_logger(tmp_path: Path, name: str) -> Logger:
    """Builds a real Logger writing under a temporary directory.

    Args:
        tmp_path : Directory that receives the logs/ subdirectory.
        name     : Logger name used for the log file.

    Returns:
        A Logger configured to write inside tmp_path.
    """
    return Logger(log_dir=str(tmp_path / "logs"), name=name)


@pytest.fixture
def logger_stub() -> SilentLogger:
    """Returns a no-op logger stub."""
    return SilentLogger()


@pytest.fixture
def recording_logger() -> RecordingLogger:
    """Returns a logger stub that records emitted messages."""
    return RecordingLogger()


@pytest.fixture
def force_cpu(monkeypatch):
    """Patches torch.cuda.is_available to report no GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Returns the repository root directory."""
    return _REPO_ROOT


@pytest.fixture(scope="session")
def scene_config() -> SynthesisConfig:
    """Returns the session-wide synthetic scene configuration."""
    return _SCENE_CONFIG


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Generates the session's synthetic scene and returns its dataset root."""
    from pipelines.processing.synthesis.pipeline import SyntheticDatasetPipeline

    config = dataclasses.replace(_SCENE_CONFIG, output_directory=tmp_path_factory.mktemp("synthetic_scene") / "synthetic_small")

    SyntheticDatasetPipeline(config, logger=SilentLogger()).run()

    return config.output_directory


@pytest.fixture(scope="session")
def data_dir(test_data_dir) -> Path:
    """Returns the scene's data directory holding the preprocessed arrays."""
    return test_data_dir / "data"


@pytest.fixture(scope="session")
def meta_dir(test_data_dir) -> Path:
    """Returns the scene's meta directory holding dataset metadata."""
    return test_data_dir / "meta"


@pytest.fixture(scope="session")
def params_dir(test_data_dir) -> Path:
    """Returns the scene's extracted-parameter run directory."""
    return test_data_dir / "params" / _SCENE_CONFIG.params_run_name


@pytest.fixture(scope="session")
def dataset_json(data_dir) -> dict:
    """Returns the parsed dataset.json manifest."""
    return json.loads((data_dir / "dataset.json").read_text())


@pytest.fixture(scope="session")
def config_state_json(meta_dir) -> dict:
    """Returns the parsed config_state.json of the preprocessing run."""
    return json.loads((meta_dir / "config_state.json").read_text())


@pytest.fixture(scope="session")
def baselines_json(meta_dir) -> dict:
    """Returns the parsed baselines.json acquisition metadata."""
    return json.loads((meta_dir / "baselines.json").read_text())


@pytest.fixture(scope="session")
def param_extraction_meta(params_dir) -> dict:
    """Returns the parsed param_extraction_meta.json of the parameter run."""
    return json.loads((params_dir / "param_extraction_meta.json").read_text())


@pytest.fixture(scope="session")
def channel_labels(dataset_json) -> list:
    """Returns the ordered chan labels of the input stack."""
    return dataset_json["channel_labels"]


@pytest.fixture(scope="session")
def cube_full(data_dir) -> np.ndarray:
    """Returns the memory-mapped cube of shape (z_values, row, col)."""
    return np.load(data_dir / "cube_full.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def aux_full(data_dir) -> np.ndarray:
    """Returns the memory-mapped Aux of shape (row, col) in metres."""
    return np.load(data_dir / "aux_full.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def primary(data_dir) -> np.ndarray:
    """Returns the memory-mapped primary complex image of shape (rows, cols)."""
    return np.load(data_dir / "primary.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def secondaries(data_dir) -> np.ndarray:
    """Returns the memory-mapped secondary complex stack of shape (channels, rows, cols)."""
    return np.load(data_dir / "secondaries.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def derived(data_dir) -> np.ndarray:
    """Returns the memory-mapped derived stack of shape (tracks, row, col)."""
    return np.load(data_dir / "derived.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def parameters(params_dir) -> np.ndarray:
    """Returns the memory-mapped Gaussian parameters of shape (params, rows, cols)."""
    return np.load(params_dir / "parameters.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def fit_diagnostics(params_dir) -> dict:
    """Returns the fit diagnostics of the parameter run keyed by array name."""
    npz = np.load(params_dir / "fit_diagnostics.npz")
    return {k: npz[k] for k in npz.files}


@pytest.fixture(scope="session")
def track_profiles(data_dir) -> dict:
    """Returns the per-channel reflectivity profiles keyed by array name."""
    npz = np.load(data_dir / "track_profiles.npz", allow_pickle=True)
    return {k: npz[k] for k in npz.files}


@pytest.fixture
def small_window():
    """Returns a 32x32 row and column slice pair for cheap windowed reads."""
    return (slice(0, 32), slice(0, 32))
