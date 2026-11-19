
from __future__ import annotations

import pytest

from pipelines.shared.dataset.dataset_queue import DatasetQueueResolver


def test_queue_lists_every_dataset_directory_in_order(tmp_path):
    for name in ("run_b", "run_a"):
        (tmp_path / name / "data").mkdir(parents=True)

    assert DatasetQueueResolver(tmp_path, []).resolve() == [tmp_path / "run_a", tmp_path / "run_b"]


def test_queue_refuses_a_base_path_without_datasets(tmp_path):
    with pytest.raises(FileNotFoundError, match="holds no dataset directory"):
        DatasetQueueResolver(tmp_path, []).resolve()


def test_queue_refuses_an_entry_without_data(tmp_path):
    (tmp_path / "logs").mkdir()

    with pytest.raises(NotADirectoryError, match="logs"):
        DatasetQueueResolver(tmp_path, []).resolve()
