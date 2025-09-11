"""
Visualization utilities for SOM
"""

import numpy as np
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Set matplotlib backend to Agg (non-interactive) before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from .core import SOM


def ensure_plots_dir(save_path: str) -> str:
    """Ensure plots directory exists and return full path"""
    if not os.path.isabs(save_path):
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        return str(plots_dir / save_path)
    return save_path


class SOMVisualizer:
    """Visualization utilities for SOM analysis"""

    @staticmethod
    def plot_quantization_error(
        som: "SOM", show_plot: bool = True, save_path: str = "qe_plot.png"
    ):
        """
        Plot quantization error over training epochs

        Args:
            som: Trained SOM instance
            show_plot: Whether to display the plot
            save_path: Path to save the plot image (None to skip saving)
        """
        if not hasattr(som, "metadata") or "training_history" not in som.metadata:
            if som.verbose:
                print("No training history available for plotting")
            return

        history = som.metadata["training_history"]
        if not history:
            if som.verbose:
                print("No training history data available")
            return

        epochs = [h["epoch"] for h in history]
        qe_values = [h["metrics"]["qe"] for h in history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, qe_values, "b-", linewidth=2, label="Quantization Error")
        plt.xlabel("Epoch")
        plt.ylabel("Quantization Error")
        plt.title("Quantization Error vs Training Epochs")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_path:
            full_path = ensure_plots_dir(save_path)
            plt.savefig(full_path, dpi=300, bbox_inches="tight")
            if som.verbose:
                print(f"QE plot saved to {full_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def visualize_weights(
        som: "SOM", show_plot: bool = True, save_path: str = "som_weights.png"
    ):
        """
        Visualize SOM weights as an image

        Args:
            som: Trained SOM instance
            show_plot: Whether to display the visualization
            save_path: Path to save the visualization (None to skip saving)
        """
        weights = som.get_weights()

        # Handle different numbers of features
        if weights.shape[2] == 1:
            # Single feature - show as grayscale
            img = weights[:, :, 0]
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap="viridis", interpolation="nearest")
            plt.colorbar(label="Weight Value")
            plt.title("SOM Weight Visualization (Single Feature)")
        elif weights.shape[2] == 2:
            # Two features - show as 2D color map
            img = np.zeros((weights.shape[0], weights.shape[1], 3))
            img[:, :, 0] = weights[:, :, 0]  # Red channel
            img[:, :, 1] = weights[:, :, 1]  # Green channel
            img[:, :, 2] = 0.5  # Blue channel (constant)

            plt.figure(figsize=(8, 8))
            plt.imshow(img, interpolation="nearest")
            plt.title("SOM Weight Visualization (2 Features as RG)")
        elif weights.shape[2] >= 3:
            # Three or more features - show first 3 as RGB
            img = weights[:, :, :3]
            # Normalize to [0, 1] range
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            plt.figure(figsize=(8, 8))
            plt.imshow(img, interpolation="nearest")
            plt.title("SOM Weight Visualization (First 3 Features as RGB)")

        plt.axis("off")

        if save_path:
            full_path = ensure_plots_dir(save_path)
            plt.savefig(full_path, dpi=300, bbox_inches="tight")
            if som.verbose:
                print(f"SOM visualization saved to {full_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_training_progress(
        som: "SOM", show_plot: bool = True, save_path: str = "training_progress.png"
    ):
        """
        Plot comprehensive training progress including QE, sigma, and alpha

        Args:
            som: Trained SOM instance
            show_plot: Whether to display the plot
            save_path: Path to save the plot image (None to skip saving)
        """
        if not hasattr(som, "metadata") or "training_history" not in som.metadata:
            if som.verbose:
                print("No training history available for plotting")
            return

        history = som.metadata["training_history"]
        if not history:
            if som.verbose:
                print("No training history data available")
            return

        epochs = [h["epoch"] for h in history]
        qe_values = [h["metrics"]["qe"] for h in history]
        sigma_values = [h["sigma"] for h in history]
        alpha_values = [h["alpha"] for h in history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Quantization Error
        ax1.plot(epochs, qe_values, "b-", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Quantization Error")
        ax1.set_title("Quantization Error")
        ax1.grid(True, alpha=0.3)

        # Sigma decay
        ax2.plot(epochs, sigma_values, "r-", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Sigma")
        ax2.set_title("Neighborhood Radius Decay")
        ax2.grid(True, alpha=0.3)

        # Alpha decay
        ax3.plot(epochs, alpha_values, "g-", linewidth=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Alpha")
        ax3.set_title("Learning Rate Decay")
        ax3.grid(True, alpha=0.3)

        # Combined view
        ax4_twin = ax4.twinx()
        ax4.plot(epochs, qe_values, "b-", linewidth=2, label="QE")
        ax4_twin.plot(epochs, sigma_values, "r-", linewidth=2, label="Sigma")
        ax4_twin.plot(epochs, alpha_values, "g-", linewidth=2, label="Alpha")

        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Quantization Error", color="b")
        ax4_twin.set_ylabel("Parameters", color="r")
        ax4.set_title("Combined Training Progress")
        ax4.grid(True, alpha=0.3)

        # Add legends
        ax4.legend(loc="upper left")
        ax4_twin.legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            full_path = ensure_plots_dir(save_path)
            plt.savefig(full_path, dpi=300, bbox_inches="tight")
            if som.verbose:
                print(f"Training progress plot saved to {full_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()
