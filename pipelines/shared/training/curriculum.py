
from __future__ import annotations


class CurriculumController:

    def __init__(self, curriculum, complete_stage, criterion, early_stopping, lr_scheduler, warmup, optimizer, update_optimizer, logger):
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
