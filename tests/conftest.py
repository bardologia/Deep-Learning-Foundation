
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

from configuration.generation import GenerationConfig

_SCENE_CONFIG = GenerationConfig(rows=96, cols=64)


class SilentProgress:

    def add_task(self, *args, **kwargs):
        return 0

    def advance(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class SilentLogger:

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
        yield

    @contextlib.contextmanager
    def track(self, transient: bool = False):
        yield SilentProgress()

    progress_bar = track

    @contextlib.contextmanager
    def live_monitor(self, title: str = "Training Monitor"):
        yield SilentProgress()


class RecordingWriter:

    def __init__(self):
        self.scalars    = []
        self.histograms = []
        self.figures    = []
        self.layouts    = []
        self.flushed    = 0
        self.closed     = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, values, step, bins="auto"):
        self.histograms.append((tag, np.asarray(values), step, bins))

    def add_figure(self, tag, figure, step, close=True):
        self.figures.append((tag, figure, step, close))

    def add_custom_scalars(self, layout):
        self.layouts.append(layout)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed += 1


class RecordingLogger(SilentLogger):

    def __init__(self):
        self.messages = []

    def section(self, msg):
        self.messages.append(("section", msg))

    def subsection(self, msg):
        self.messages.append(("subsection", msg))

    def ok(self, msg):
        self.messages.append(("ok", msg))

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def error(self, msg):
        self.messages.append(("error", msg))

    def metrics_table(self, rows, columns=None, title=None):
        self.messages.append(("metrics_table", title))


def make_file_logger(tmp_path: Path, name: str) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name=name)


@pytest.fixture
def logger_stub() -> SilentLogger:
    return SilentLogger()


@pytest.fixture
def recording_logger() -> RecordingLogger:
    return RecordingLogger()


@pytest.fixture
def force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def scene_config() -> GenerationConfig:
    return _SCENE_CONFIG


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    from pipelines.processing.generation import DatasetGenerator

    config = dataclasses.replace(_SCENE_CONFIG, output_directory=tmp_path_factory.mktemp("synthetic_scene") / "synthetic")

    DatasetGenerator(config, logger=SilentLogger()).run()

    return config.output_directory


@pytest.fixture(scope="session")
def dataset_json(test_data_dir) -> dict:
    return json.loads((test_data_dir / "dataset.json").read_text())


@pytest.fixture(scope="session")
def inputs(test_data_dir) -> np.ndarray:
    return np.load(test_data_dir / "inputs.npy", mmap_mode="r")


@pytest.fixture(scope="session")
def targets(test_data_dir) -> np.ndarray:
    return np.load(test_data_dir / "targets.npy", mmap_mode="r")
