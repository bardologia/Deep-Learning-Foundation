from __future__ import annotations


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker

        self.patience  = self.config.early_stopping.patience
        self.min_delta = self.config.early_stopping.min_delta

        self.logger.section("[Early Stopping]")
        self.logger.kv_table({
            "Patience"     : f"{self.patience} validation rounds (not epochs; validation may run every N epochs)",
            "Min Delta"    : self.min_delta,
            "Restore best" : self.config.early_stopping.restore_best,
        })

        self.best_loss  = None
        self.counter    = 0
        self.best_epoch = -1
        self.triggered  = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss  = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter    = 0
            stop            = False

        else:
            self.counter += 1
            stop          = (self.counter >= self.patience)

        self.tracker.log_scalar("controls/early_stop_counter", self.counter, epoch)

        if stop:
            self.triggered = True
            self.logger.warning(f"Early stopping triggered at epoch {epoch + 1}: {self.counter} validation rounds without improvement over min_delta={self.min_delta}. Best epoch (by the min_delta rule) was {self.best_epoch + 1}.")

        return stop

    def reset(self) -> None:
        self.best_loss  = None
        self.counter    = 0
        self.best_epoch = -1
        self.triggered  = False

    def state_dict(self) -> dict:
        return {
            "best_loss"  : self.best_loss,
            "counter"    : self.counter,
            "best_epoch" : self.best_epoch,
            "triggered"  : self.triggered,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_loss  = state["best_loss"]
        self.counter    = int(state["counter"])
        self.best_epoch = int(state["best_epoch"])
        self.triggered  = bool(state["triggered"])
