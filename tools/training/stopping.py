"""Patience-based early stopping driven by the validation loss."""

from __future__ import annotations


class EarlyStopping:
    """Stops training after a number of validation rounds without a min_delta improvement.

    Patience counts validation rounds rather than epochs, so a run validating
    every N epochs waits patience*N epochs before stopping.

    Attributes:
        config     : Training configuration exposing an early_stopping block.
        logger     : Logger for the configuration table and the trigger warning.
        tracker    : Experiment tracker receiving the patience counter.
        patience   : Validation rounds tolerated without improvement.
        min_delta  : Improvement in validation loss required to reset the counter.
        best_loss  : Best validation loss under the min_delta rule, None before the first round.
        counter    : Validation rounds since the last improvement.
        best_epoch : Zero-based epoch of the best round, -1 before the first.
        triggered  : Whether stopping has fired.
    """

    def __init__(self, config, logger, tracker):
        """Reads the patience settings from the config and reports them.

        Args:
            config  : Training configuration exposing an early_stopping block.
            logger  : Logger for the configuration table.
            tracker : Experiment tracker for the patience counter.
        """
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
        """Records one validation round and reports whether training should stop.

        Args:
            val_loss : Validation loss of this round.
            epoch    : Zero-based epoch index of this round.

        Returns:
            True when patience is exhausted and training should stop.
        """
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
        """Clears the patience counter and best loss, as at a curriculum stage change."""
        self.best_loss  = None
        self.counter    = 0
        self.best_epoch = -1
        self.triggered  = False

    def state_dict(self) -> dict:
        """Returns the best loss, patience counter, best epoch and trigger flag."""
        return {
            "best_loss"  : self.best_loss,
            "counter"    : self.counter,
            "best_epoch" : self.best_epoch,
            "triggered"  : self.triggered,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores the patience bookkeeping from a saved state dictionary."""
        self.best_loss  = state["best_loss"]
        self.counter    = int(state["counter"])
        self.best_epoch = int(state["best_epoch"])
        self.triggered  = bool(state["triggered"])
