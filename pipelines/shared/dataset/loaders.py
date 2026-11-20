
from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from tools.monitoring.logger       import Logger
from tools.runtime.reproducibility import Reproducibility


class Loader:

    @staticmethod
    def build(
        train_dataset   : Dataset,
        val_dataset     : Dataset,
        test_dataset    : Dataset,
        batch_size      : int,
        num_workers     : int,
        logger          : Logger,
        seed            : int  = 0,
        prefetch_factor : int  = 8,
        section_label   : str  = "[Loaders]",
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        drop_last = len(train_dataset) > batch_size

        logger.section(section_label)
        logger.kv_table({
            "Batch size"         : batch_size,
            "Num workers"        : num_workers,
            "Prefetch factor"    : prefetch_factor if num_workers > 0 else f"{prefetch_factor} (inactive: num_workers=0)",
            "Persistent workers" : num_workers > 0,
            "Worker seeding"     : f"base seed {seed} + worker id (augmenter RNG reseeded per worker)" if num_workers > 0 else "main process",
            "Drop last"          : f"train only, {drop_last} (a train split of one batch or less keeps its partial batch)",
            "Seed"               : seed,
        })

        worker_init = Reproducibility.worker_init(seed) if num_workers > 0 else None

        _base = dict(
            batch_size         = batch_size,
            num_workers        = num_workers,
            pin_memory         = True,
            persistent_workers = num_workers > 0,
            prefetch_factor    = prefetch_factor if num_workers > 0 else None,
            worker_init_fn     = worker_init,
        )

        train_loader = DataLoader(train_dataset, shuffle = True,  drop_last = drop_last, generator = Reproducibility.generator(seed), **_base)
        val_loader   = DataLoader(val_dataset,   shuffle = False, drop_last = False,     **_base)
        test_loader  = DataLoader(test_dataset,  shuffle = False, drop_last = False,     **_base)

        logger.kv_table({
            "Train samples" : len(train_dataset),
            "Val samples"   : len(val_dataset),
            "Test samples"  : len(test_dataset),
            "Train batches" : len(train_loader),
            "Val batches"   : len(val_loader),
            "Test batches"  : len(test_loader),
        })

        return train_loader, val_loader, test_loader
