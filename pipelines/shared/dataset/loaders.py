"""Construction of the train/val/test DataLoader triple shared by every pipeline."""

from __future__ import annotations

from typing import Iterator, Tuple

from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from tools.monitoring.logger       import Logger
from tools.runtime.reproducibility import Reproducibility
from tools.training.distributed    import DistributedContext


class EvaluationShardSampler(Sampler):
    """Strided shard of a dataset for one DDP rank, visiting every sample once.

    `DistributedSampler` pads the short shards with duplicated samples so all
    ranks see the same number of batches, which the gradient-synchronised
    training loop needs but which would double-count those samples in the
    reduced evaluation loss. Evaluation reduces once after its loop, so unequal
    shard lengths are harmless and the reduced loss equals the single-process
    value.

    Attributes:
        indices : Dataset indices this rank evaluates, in ascending order.
    """

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int) -> None:
        """Takes every `num_replicas`-th index starting at the rank's offset.

        Args:
            dataset      : Dataset being sharded.
            num_replicas : Number of ranks the dataset is split over.
            rank         : Zero-based rank this shard belongs to.
        """
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self) -> Iterator[int]:
        """Yields this rank's dataset indices."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Returns the number of samples in this rank's shard."""
        return len(self.indices)


class Loader:
    """Builds seeded DataLoaders with a shared worker and memory configuration."""

    @staticmethod
    def build(
        train_dataset   : Dataset,
        val_dataset     : Dataset,
        test_dataset    : Dataset,
        batch_size      : int,
        num_workers     : int,
        logger          : Logger,
        pin_memory      : bool = True,
        shuffle_train   : bool = True,
        seed            : int  = 0,
        prefetch_factor : int  = 8,
        section_label   : str  = "[Loaders]",
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Builds the train, validation and test loaders and logs their sizes.

        Args:
            train_dataset   : Training split dataset.
            val_dataset     : Validation split dataset.
            test_dataset    : Test split dataset.
            batch_size      : Samples per batch, shared by all three loaders.
            num_workers     : Worker processes per loader; zero loads in the main process and disables prefetching and persistent workers.
            logger          : Logger the loader configuration and split sizes are written to.
            pin_memory      : Whether batches are placed in pinned host memory.
            shuffle_train   : Whether the training loader shuffles each epoch.
            seed            : Base seed for the shuffling generator and worker seeding.
            prefetch_factor : Batches prefetched per worker.
            section_label   : Section heading used in the log.

        Returns:
            Triple of (train, validation, test) loaders. Only the training loader
            shuffles and drops a partial final batch. Inside an active DDP group
            the given batch size is divided by the rank count so the global batch
            is preserved, and all three splits are sharded across the ranks: the
            training split with a DistributedSampler whose shards are padded to
            equal length, the evaluation splits with an EvaluationShardSampler
            that visits every sample exactly once.

        Raises:
            ValueError : The training loader would yield zero batches, so the run would take no gradient step; or the batch size does not divide evenly across the DDP ranks.
        """
        context = DistributedContext.current()

        if context.active:
            global_batch = batch_size
            batch_size   = context.shard_batch(batch_size)

        logger.section(section_label)
        logger.kv_table({
            "Batch size"         : f"{batch_size} per rank ({global_batch} global over {context.world_size} DDP ranks)" if context.active else batch_size,
            "DDP sharding"       : f"rank {context.rank + 1}/{context.world_size} (all splits sharded; train shards padded to equal length, val/test shards unpadded)" if context.active else "off",
            "Num workers"        : num_workers,
            "Pin memory"         : pin_memory,
            "Prefetch factor"    : prefetch_factor if num_workers > 0 else f"{prefetch_factor} (inactive: num_workers=0)",
            "Persistent workers" : num_workers > 0,
            "Worker seeding"     : f"base seed {seed} + worker id (augmenter RNG reseeded per worker)" if num_workers > 0 else "main process",
            "Shuffle train"      : shuffle_train,
            "Drop last"          : "train only",
            "Seed"               : seed,
        })

        worker_init = Reproducibility.worker_init(seed) if num_workers > 0 else None

        _base = dict(
            batch_size         = batch_size,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = prefetch_factor if num_workers > 0 else None,
            worker_init_fn     = worker_init,
        )

        if context.active:
            _shard = dict(num_replicas=context.world_size, rank=context.rank)

            train_loader = DataLoader(train_dataset, sampler = DistributedSampler(train_dataset, shuffle=shuffle_train, seed=seed, **_shard), drop_last = True,  generator = Reproducibility.generator(seed), **_base)
            val_loader   = DataLoader(val_dataset,   sampler = EvaluationShardSampler(val_dataset,  **_shard),                               drop_last = False, **_base)
            test_loader  = DataLoader(test_dataset,  sampler = EvaluationShardSampler(test_dataset, **_shard),                               drop_last = False, **_base)
        else:
            train_loader = DataLoader(train_dataset, shuffle = shuffle_train, drop_last = True,  generator = Reproducibility.generator(seed), **_base)
            val_loader   = DataLoader(val_dataset,   shuffle = False,         drop_last = False, **_base)
            test_loader  = DataLoader(test_dataset,  shuffle = False,         drop_last = False, **_base)

        if len(train_loader) == 0:
            raise ValueError(f"Train loader yields zero batches: {len(train_dataset)} training samples with batch size {batch_size} and drop_last=True. Lower the batch size or widen the training split; training on an empty loader would produce a run with no gradient steps.")

        logger.kv_table({
            "Train samples" : len(train_dataset),
            "Val samples"   : len(val_dataset),
            "Test samples"  : len(test_dataset),
            "Train batches" : f"{len(train_loader)} (drop_last floors a partial final batch)",
            "Val batches"   : len(val_loader),
            "Test batches"  : len(test_loader),
        })

        return train_loader, val_loader, test_loader
