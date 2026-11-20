
from __future__ import annotations


from pipelines.shared.dataset.dataset_queue import DatasetQueueResolver


def test_queue_lists_every_dataset_directory_in_order(tmp_path):
    for name in ("run_b", "run_a"):
        (tmp_path / name / "data").mkdir(parents=True)

    assert DatasetQueueResolver(tmp_path, []).resolve() == [tmp_path / "run_a", tmp_path / "run_b"]
