"""
Callback system for monitoring and intervention during SOM training
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SOM


class Callback(ABC):
    """Abstract base class for callbacks"""

    @abstractmethod
    def on_epoch_begin(self, epoch: int, som: "SOM") -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, som: "SOM", metrics: Dict) -> None:
        pass

    @abstractmethod
    def on_training_begin(self, som: "SOM") -> None:
        pass

    @abstractmethod
    def on_training_end(self, som: "SOM") -> None:
        pass


class CheckpointCallback(Callback):
    """ARCHITECTURE FIX #4: Callback for checkpointing during training"""

    def __init__(self, checkpoint_dir: str, interval: int = 100):
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_epoch_begin(self, epoch: int, som: "SOM") -> None:
        pass

    def on_epoch_end(self, epoch: int, som: "SOM", metrics: Dict) -> None:
        if epoch % self.interval == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl"
            )
            try:
                som.save(checkpoint_path)
                if som.verbose:
                    print(f"Checkpoint saved: {checkpoint_path}")
            except (IOError, OSError) as e:
                if som.verbose:
                    print(f"Warning: Failed to save checkpoint: {e}")

    def on_training_begin(self, som: "SOM") -> None:
        pass

    def on_training_end(self, som: "SOM") -> None:
        final_path = os.path.join(self.checkpoint_dir, "final_model.pkl")
        try:
            som.save(final_path)
        except (IOError, OSError) as e:
            if som.verbose:
                print(f"Warning: Failed to save final model: {e}")


class EarlyStoppingCallback(Callback):
    """ARCHITECTURE FIX #9: Callback for custom early stopping logic"""

    def __init__(
        self, monitor: str = "qe", patience: int = 10, min_delta: float = 1e-4
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.wait = 0

    def on_epoch_begin(self, epoch: int, som: "SOM") -> None:
        pass

    def on_epoch_end(self, epoch: int, som: "SOM", metrics: Dict) -> None:
        current_value = metrics.get(self.monitor, float("inf"))
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                som.stop_training = True
                if som.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")

    def on_training_begin(self, som: "SOM") -> None:
        self.best_value = float("inf")
        self.wait = 0

    def on_training_end(self, som: "SOM") -> None:
        pass
