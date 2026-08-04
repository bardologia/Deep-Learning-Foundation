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
        pin_memory      : bool = True,
        shuffle_train   : bool = True,
        seed            : int  = 0,
        prefetch_factor : int  = 8,
        section_label   : str  = "[Loaders]",
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        logger.section(section_label)
        logger.kv_table({
            "Batch size"         : batch_size,
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

        train_loader = DataLoader(train_dataset, shuffle = shuffle_train, drop_last = True,  generator = Reproducibility.generator(seed), **_base)
        val_loader   = DataLoader(val_dataset,   shuffle = False,         drop_last = False, **_base)
        test_loader  = DataLoader(test_dataset,  shuffle = False,         drop_last = False, **_base)

        logger.kv_table({
            "Train samples" : len(train_dataset),
            "Val samples"   : len(val_dataset),
            "Test samples"  : len(test_dataset),
            "Train batches" : f"{len(train_loader)} (drop_last floors a partial final batch)",
            "Val batches"   : len(val_loader),
            "Test batches"  : len(test_loader),
        })

        return train_loader, val_loader, test_loader
