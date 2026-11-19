
from __future__ import annotations

import multiprocessing
import os
import socket
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from pipelines.backbone.training.trainer import Trainer
from tools.monitoring.logger             import Logger
from tools.training.distributed          import DistributedContext

from tests.backbone_training._helpers import identity_normalizer, tiny_model, tiny_trainer_config, x_axis_numpy


pytestmark = pytest.mark.usefixtures("force_cpu")

N_BATCHES  = 4
BATCH_SIZE = 2
HW         = 8


@pytest.fixture(autouse=True)
def _fresh_context():
    DistributedContext.reset()
    yield
    DistributedContext.reset()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _dataset() -> TensorDataset:
    generator = torch.Generator().manual_seed(7)
    n         = N_BATCHES * BATCH_SIZE

    return TensorDataset(torch.randn(n, 2, HW, HW, generator=generator), torch.randn(n, 6, HW, HW, generator=generator))


def _loader(rank: int | None) -> DataLoader:
    dataset = _dataset()

    if rank is not None:
        dataset = Subset(dataset, [index for index in range(len(dataset)) if (index // BATCH_SIZE) % 2 == rank])

    return DataLoader(dataset, batch_size=BATCH_SIZE)


def _trainer(run_dir) -> Trainer:
    torch.manual_seed(0)

    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    logger           = Logger(log_dir=str(run_dir / "logs"), name="ddp_trainer", level="ERROR")

    trainer = Trainer(model, model_cfg, x_axis_numpy(), tiny_trainer_config(n_gaussians=2), run_dir, logger, norm_stats=identity_normalizer(6), emit_docs=False)
    trainer.warmup.resolve(200)
    return trainer


def _rank_worker(rank: int, port: int, run_root: str, results: multiprocessing.Queue) -> None:
    torch.set_num_threads(1)

    if rank >= 0:
        os.environ.update(DistributedContext.environment(rank=rank, world_size=2, master_port=port))
    DistributedContext.reset()

    trainer = _trainer(Path(run_root) / f"worker{rank}")
    shard   = rank if rank >= 0 else None

    evaluated = trainer.evaluate(_loader(shard), epoch=0, stage="val")
    trained   = trainer.train_epoch(_loader(shard), epoch=0)

    results.put((rank, evaluated["avg_loss"], evaluated["num_batches"], float(trained)))

    if rank >= 0:
        torch.distributed.destroy_process_group()


def test_two_rank_backbone_group_reports_group_wide_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    port    = _free_port()
    spawner = multiprocessing.get_context("spawn")
    results = spawner.Queue()
    workers = [spawner.Process(target=_rank_worker, args=(rank, port, str(tmp_path / "workers"), results)) for rank in (-1, 0, 1)]

    for worker in workers:
        worker.start()

    observations = {}
    for _ in workers:
        rank, *values = results.get(timeout=300)
        observations[rank] = values

    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0

    reference_loss, reference_batches, _reference_train = observations[-1]

    for rank in (0, 1):
        val_loss, num_batches, train_loss = observations[rank]

        assert num_batches == reference_batches == N_BATCHES
        assert val_loss    == pytest.approx(reference_loss, rel=1e-5)
        assert train_loss  == pytest.approx(observations[0][2], rel=1e-5)
