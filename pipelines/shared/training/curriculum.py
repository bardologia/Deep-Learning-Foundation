"""Curriculum swap controller shared by the backbone and JEPA trainers.

Performs the one-off swap from the warmup loss stage to the complete stage at
the configured epoch, optionally resetting the optimization state so the second
phase starts clean.
"""

from __future__ import annotations


class CurriculumController:
    """Performs the one-off swap from the warmup to the complete loss stage.

    Besides replacing the loss configuration, the swap optionally resets early
    stopping, the learning-rate schedule, the warmup and the optimizer moments,
    so the second phase starts from a clean optimization state.

    Attributes:
        curriculum       : Curriculum configuration holding the swap epoch and reset flags.
        complete_stage   : Loss configuration installed at the swap.
        criterion        : Loss whose stage configuration is replaced.
        early_stopping   : Early-stopping state reset at the swap.
        lr_scheduler     : Learning-rate schedule optionally restarted at the swap.
        warmup           : Learning-rate warmup optionally restarted at the swap.
        optimizer        : Optimizer whose per-parameter state may be cleared.
        update_optimizer : Callable applying new learning rates to the optimizer.
        logger           : Logger recording what the swap changed.
    """

    def __init__(self, curriculum, complete_stage, criterion, early_stopping, lr_scheduler, warmup, optimizer, update_optimizer, logger):
        """Stores the training components the curriculum swap acts on.

        Args:
            curriculum       : Configuration exposing ``enabled``, ``swap_epoch`` and the ``reset_lr``/``reset_warmup``/``reset_optimizer`` flags.
            complete_stage   : Loss configuration passed to the criterion's ``set_curriculum`` at the swap.
            criterion        : Loss whose stage configuration is replaced.
            early_stopping   : Early-stopping state reset at the swap.
            lr_scheduler     : Learning-rate schedule optionally restarted.
            warmup           : Learning-rate warmup optionally restarted.
            optimizer        : Optimizer whose per-parameter state may be cleared.
            update_optimizer : Callable applying new learning rates.
            logger           : Logger recording what the swap changed.
        """
        self.curriculum       = curriculum
        self.complete_stage   = complete_stage
        self.criterion        = criterion
        self.early_stopping   = early_stopping
        self.lr_scheduler     = lr_scheduler
        self.warmup           = warmup
        self.optimizer        = optimizer
        self.update_optimizer = update_optimizer
        self.logger           = logger

    def maybe_swap(self, epoch: int) -> bool:
        """Swaps to the complete loss stage when `epoch` is the configured swap epoch.

        Args:
            epoch : Zero-based epoch about to start.

        Returns:
            True when the swap was performed, False otherwise.
        """
        lc = self.curriculum
        if not (lc.enabled and epoch == lc.swap_epoch):
            return False

        self.logger.section(f"[Curriculum Loss Swap @ epoch {epoch + 1}]")
        self.criterion.set_curriculum(self.complete_stage)
        self.logger.subsection("Loss config replaced with the complete stage.")

        self.early_stopping.reset()
        self.logger.subsection("Early stopping reset.")

        if lc.reset_lr:
            self.lr_scheduler.reset(epoch_offset=epoch)
            self.logger.subsection(f"LR scheduler reset (epoch offset = {epoch}).")

        if lc.reset_warmup:
            self.warmup.reset()
            self.logger.subsection(f"Warmup reset ({self.warmup.warmup_steps} steps).")

        if lc.reset_optimizer:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    state = self.optimizer.state.get(p)
                    if state:
                        state.clear()

            self.logger.subsection("Optimizer state (Adam moments) cleared.")

        if lc.reset_lr or lc.reset_warmup:
            self.update_optimizer(self.lr_scheduler.effective_lrs())
            self.logger.subsection(f"Optimizer LR set to the scheduled warmup-adjusted value (factor={self.warmup.factor():.4f}) for swap epoch.")

        return True
