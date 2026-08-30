"""Tests for the data-parallel DistributedContext, inert and in a live two-rank group."""

from __future__ import annotations

import multiprocessing
import socket

import pytest
import torch
import torch.distributed as dist

from tools.training.distributed import DistributedContext


@pytest.fixture(autouse=True)
def _fresh_context():
    """Clears the cached context before and after every test."""
    DistributedContext.reset()
    yield
    DistributedContext.reset()


def _free_port() -> int:
    """Returns a currently free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_current_without_environment_is_inert():
    """A process without the fan-out variables gets a single-rank inert context."""
    context = DistributedContext.current()

    assert context.active is False
    assert context.rank == 0
    assert context.world_size == 1


def test_current_parses_environment(monkeypatch):
    """The fan-out variables produce an active context with the stamped coordinates."""
    monkeypatch.setenv(DistributedContext.RANK_VAR,  "1")
    monkeypatch.setenv(DistributedContext.WORLD_VAR, "4")
    monkeypatch.setenv(DistributedContext.PORT_VAR,  "29123")

    context = DistributedContext.current()

    assert context.active is True
    assert (context.rank, context.world_size, context.master_port) == (1, 4, 29123)


def test_current_rejects_partial_environment(monkeypatch):
    """A rank variable without its world size or port is a stamping bug and raises."""
    monkeypatch.setenv(DistributedContext.RANK_VAR, "1")

    with pytest.raises(ValueError, match="stamps all of them together"):
        DistributedContext.current()


def test_invalid_coordinates_raise():
    """A rank outside the world size is rejected."""
    with pytest.raises(ValueError, match="invalid distributed coordinates"):
        DistributedContext(rank=2, world_size=2, master_addr="127.0.0.1", master_port=1)


def test_environment_round_trips(monkeypatch):
    """The stamped child environment parses back into the same coordinates."""
    for key, value in DistributedContext.environment(rank=2, world_size=3, master_port=29500).items():
        monkeypatch.setenv(key, value)

    context = DistributedContext.current()

    assert (context.rank, context.world_size, context.master_port) == (2, 3, 29500)


def test_shard_batch_divides_and_validates():
    """The global batch divides across ranks and a remainder is rejected."""
    context = DistributedContext(rank=0, world_size=4, master_addr="127.0.0.1", master_port=1)

    assert context.shard_batch(256) == 64

    with pytest.raises(ValueError, match="does not divide"):
        context.shard_batch(250)


def test_initialise_rejects_an_unmasked_cuda_index():
    """Collectives run on the single visible device, so a raw CUDA index is refused."""
    context = DistributedContext(rank=0, world_size=2, master_addr="127.0.0.1", master_port=1)

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        context.initialise(torch.device("cuda", 2))


def test_inert_context_operations_are_identities():
    """Every collective helper of an inert context is a passthrough."""
    context = DistributedContext.current()
    model   = torch.nn.Linear(2, 1, bias=False)

    assert context.shard_batch(5) == 5
    assert context.sync_window(model, True) is True
    assert context.sync_window(model, False) is False
    assert context.mean_scalar(3.5) == 3.5
    assert context.sum_scalars(1.0, 2.0) == (1.0, 2.0)


def _rank_worker(rank: int, world: int, port: int, results: multiprocessing.Queue) -> None:
    """Exercises broadcast, window sync and scalar reductions as one rank of a Gloo group.

    Args:
        rank    : Zero-based rank of this worker.
        world   : Number of ranks in the group.
        port    : Rendezvous port shared by the group.
        results : Queue collecting the per-rank observations.
    """
    context = DistributedContext(rank=rank, world_size=world, master_addr="127.0.0.1", master_port=port)
    context.initialise(torch.device("cpu"))

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(float(rank + 1))
    context.broadcast_parameters(model)
    weight_after_broadcast = float(model.weight.mean())

    model.weight.grad = torch.full_like(model.weight, 2.0 * (rank + 1))
    full_step          = context.sync_window(model, True)
    full_gradient      = float(model.weight.grad.mean())

    model.weight.grad = torch.full_like(model.weight, 6.0) if rank == 0 else None
    partial_step      = context.sync_window(model, rank == 0)
    partial_gradient  = float(model.weight.grad.mean())

    model.weight.grad = None
    empty_step        = context.sync_window(model, False)

    mean = context.mean_scalar(float(rank))
    sums = context.sum_scalars(1.0, float(rank))

    results.put((rank, weight_after_broadcast, full_step, full_gradient, partial_step, partial_gradient, empty_step, mean, sums))
    dist.destroy_process_group()


def test_two_rank_group_stays_in_lockstep():
    """A live two-rank Gloo group broadcasts, averages gradients and reduces scalars.

    The full window averages the two ranks' gradients, the partial window
    divides by the single contributing rank while still stepping everywhere,
    the empty window steps nowhere, and both ranks compute identical reduced
    scalars.
    """
    port    = _free_port()
    spawner = multiprocessing.get_context("fork")
    results = spawner.Queue()
    workers = [spawner.Process(target=_rank_worker, args=(rank, 2, port, results)) for rank in range(2)]

    for worker in workers:
        worker.start()

    observations = {}
    for _ in workers:
        rank, *values = results.get(timeout=120)
        observations[rank] = values

    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    for rank in (0, 1):
        weight, full_step, full_gradient, partial_step, partial_gradient, empty_step, mean, sums = observations[rank]

        assert weight == pytest.approx(1.0)
        assert full_step is True
        assert full_gradient == pytest.approx(3.0)
        assert partial_step is True
        assert partial_gradient == pytest.approx(6.0)
        assert empty_step is False
        assert mean == pytest.approx(0.5)
        assert sums == pytest.approx((2.0, 1.0))
