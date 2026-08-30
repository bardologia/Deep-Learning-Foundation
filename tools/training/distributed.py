"""Synchronous data-parallel coordination for multi-GPU training ranks.

`DistributedContext` describes one rank of a data-parallel training group. The
context is parsed once from the environment variables stamped by the DDP
fan-out; a process launched without them gets an inert context whose collective
methods are identity operations, so single-process training paths are
untouched. Active ranks join a `torch.distributed` process group and keep their
model replicas in lockstep by broadcasting the initial parameters, averaging
gradients across ranks at every optimizer window, and reducing evaluation
losses so checkpointing and early stopping decide identically everywhere.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


class DistributedContext:
    """One rank of a synchronous data-parallel training group.

    Attributes:
        RANK_VAR     : Environment variable naming this process's rank.
        WORLD_VAR    : Environment variable naming the number of ranks.
        ADDR_VAR     : Environment variable naming the rendezvous host.
        PORT_VAR     : Environment variable naming the rendezvous port.
        INIT_TIMEOUT : Rendezvous and collective timeout, generous enough to cover the per-rank dataset build and overfit gate that run before the first collective.
        rank         : Zero-based rank of this process.
        world_size   : Number of ranks in the group; 1 means inert.
        master_addr  : Rendezvous host of the process group.
        master_port  : Rendezvous port of the process group.
        device       : Device collectives run on, set at initialisation.
    """

    RANK_VAR  = "MLTEMPLATE_DDP_RANK"
    WORLD_VAR = "MLTEMPLATE_DDP_WORLD_SIZE"
    ADDR_VAR  = "MLTEMPLATE_DDP_MASTER_ADDR"
    PORT_VAR  = "MLTEMPLATE_DDP_MASTER_PORT"

    INIT_TIMEOUT = timedelta(hours=2)

    _current = None

    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int) -> None:
        """Stores the group coordinates without touching the process group.

        Args:
            rank        : Zero-based rank of this process.
            world_size  : Number of ranks in the group.
            master_addr : Rendezvous host of the process group.
            master_port : Rendezvous port of the process group.

        Raises:
            ValueError : If the rank does not fall inside the world size.
        """
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid distributed coordinates: rank {rank} of world size {world_size}")

        self.rank        = int(rank)
        self.world_size  = int(world_size)
        self.master_addr = master_addr
        self.master_port = int(master_port)
        self.device      = None

    @property
    def active(self) -> bool:
        """Whether this process is one rank of a multi-rank group."""
        return self.world_size > 1

    @classmethod
    def current(cls) -> "DistributedContext":
        """Returns the process-wide context parsed once from the environment.

        A process whose environment carries no rank variable gets an inert
        single-rank context.

        Raises:
            ValueError : If the rank variable is set but the world size or port variables are missing.
        """
        if cls._current is None:
            rank = os.environ.get(cls.RANK_VAR)

            if rank is None:
                cls._current = cls(rank=0, world_size=1, master_addr="127.0.0.1", master_port=0)
            else:
                world = os.environ.get(cls.WORLD_VAR)
                port  = os.environ.get(cls.PORT_VAR)
                if world is None or port is None:
                    raise ValueError(f"{cls.RANK_VAR} is set but {cls.WORLD_VAR} or {cls.PORT_VAR} is missing; the DDP fan-out stamps all of them together")

                cls._current = cls(rank=int(rank), world_size=int(world), master_addr=os.environ.get(cls.ADDR_VAR, "127.0.0.1"), master_port=int(port))

        return cls._current

    @classmethod
    def reset(cls) -> None:
        """Clears the cached context so the next `current()` re-reads the environment."""
        cls._current = None

    @classmethod
    def environment(cls, rank: int, world_size: int, master_port: int, master_addr: str = "127.0.0.1") -> dict:
        """Returns the environment variables that make a child process this rank.

        Args:
            rank        : Zero-based rank of the child.
            world_size  : Number of ranks in the group.
            master_port : Rendezvous port shared by the group.
            master_addr : Rendezvous host shared by the group.

        Returns:
            Mapping of the four context variables to their string values.
        """
        return {
            cls.RANK_VAR  : str(rank),
            cls.WORLD_VAR : str(world_size),
            cls.ADDR_VAR  : master_addr,
            cls.PORT_VAR  : str(master_port),
        }

    def shard_batch(self, batch_size: int) -> int:
        """Returns the per-rank micro-batch preserving the configured global batch.

        Args:
            batch_size : Global batch size from the training configuration.

        Returns:
            The per-rank batch size, unchanged when the context is inert.

        Raises:
            ValueError : If the global batch does not divide evenly across ranks.
        """
        if not self.active:
            return batch_size

        if batch_size % self.world_size != 0:
            raise ValueError(f"batch_size {batch_size} does not divide across {self.world_size} DDP ranks; pick a multiple of the rank count so the global batch and the LR rule are preserved exactly")

        return batch_size // self.world_size

    def initialise(self, device: torch.device) -> None:
        """Joins the process group, choosing the backend from the device type.

        Collectives run on ``cuda:0``, which is the rank's own GPU only because
        the fan-out masks that GPU through ``CUDA_VISIBLE_DEVICES`` before torch
        is imported, so every rank process sees exactly one device. A caller
        that instead pins a raw device index is rejected rather than silently
        reducing on the wrong GPU.

        Idempotent: an inert context and an already-initialised group both
        return immediately.

        Args:
            device : Device this rank trains on; CUDA selects the NCCL backend, anything else selects Gloo.

        Raises:
            ValueError : If a CUDA device carries an index other than zero, which means the rank's GPU was not masked to the single visible device.
        """
        if device.type == "cuda" and device.index not in (None, 0):
            raise ValueError(f"DDP rank {self.rank} was given {device}, but collectives run on the single visible device: mask the rank's GPU through CUDA_VISIBLE_DEVICES before importing torch instead of pinning a device index")

        if not self.active or dist.is_initialized():
            self.device = torch.device("cuda", 0) if device.type == "cuda" else torch.device("cpu")
            return

        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend     = backend,
            init_method = f"tcp://{self.master_addr}:{self.master_port}",
            rank        = self.rank,
            world_size  = self.world_size,
            timeout     = self.INIT_TIMEOUT,
        )

        self.device = torch.device("cuda", 0) if device.type == "cuda" else torch.device("cpu")

    def broadcast_parameters(self, model: torch.nn.Module) -> None:
        """Copies rank zero's parameters and buffers onto every other rank.

        Args:
            model : Model whose tensors are overwritten in place on ranks > 0.
        """
        if not self.active:
            return

        with torch.no_grad():
            for tensor in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(tensor.data, src=0)

    def sync_window(self, model: torch.nn.Module, has_grads: bool) -> bool:
        """Averages the accumulated gradients across the ranks that produced any.

        Every rank must call this once per optimizer window. A rank whose whole
        window was skipped for non-finite losses contributes zeros and is
        excluded from the divisor, and every rank leaves with the same verdict
        on whether the group takes the optimizer step, so replicas never
        diverge.

        Args:
            model     : Model whose parameter gradients are averaged in place.
            has_grads : Whether this rank accumulated any gradients this window.

        Returns:
            Whether any rank in the group accumulated gradients; `has_grads`
            unchanged when the context is inert.
        """
        if not self.active:
            return has_grads

        flag = torch.tensor([1.0 if has_grads else 0.0], device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)

        contributors = float(flag.item())
        if contributors == 0.0:
            return False

        for parameter in model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)

            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(contributors)

        return True

    def mean_scalar(self, value: float) -> float:
        """Returns the mean of a scalar across all ranks, the value itself when inert."""
        if not self.active:
            return value

        tensor = torch.tensor([float(value)], device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        return float(tensor.item()) / self.world_size

    def sum_scalars(self, *values: float) -> tuple:
        """Returns the elementwise sum of scalars across all ranks.

        Args:
            *values : Per-rank scalars, reduced position by position.

        Returns:
            Tuple of the summed scalars, the inputs unchanged when inert.
        """
        if not self.active:
            return tuple(values)

        tensor = torch.tensor([float(value) for value in values], device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        return tuple(float(entry) for entry in tensor.tolist())
