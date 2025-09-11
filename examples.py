"""
Example usage of the SOM package
"""

import numpy as np
import os
import shutil
from som import SOM, SOMConfig, InitStrategy, DistanceMetric


def run_examples():
    """Run comprehensive examples of SOM usage"""

    # Set random seed
    np.random.seed(42)

    # Example 1: Basic usage with new architecture and visualization
    print("Example 1: Basic SOM with visualization")
    data = np.random.random((100, 3)).astype(np.float32)

    # Configuration management
    config = SOMConfig(
        width=20,
        height=20,
        n_features=3,
        n_iterations=200,
        init_strategy=InitStrategy.PCA,
        distance_metric=DistanceMetric.EUCLIDEAN,
        seed=42,
    )

    # Create and train SOM
    som = SOM(config)
    som.fit(data)

    # Plot quantization error over epochs and visualize weights
    print("\nPlotting quantization error and visualizing weights...")
    som.plot_quantization_error(show_plot=False).visualize_weights(show_plot=False)

    # Save model
    som.save("trained_som.pkl")
    print("Model saved!")

    # Get comprehensive info
    info = som.get_info()
    print(f"Total epochs trained: {info['total_epochs']}")
    print(f"Total samples seen: {info['total_samples']}")

    # Example 2: Load and continue training
    print("\nExample 2: Incremental training")

    # Load saved model
    som_loaded = SOM.load("trained_som.pkl")
    print("Model loaded!")

    # Continue training with new data
    new_data = np.random.random((50, 3)).astype(np.float32)
    som_loaded.fit(new_data, n_iterations=100)
    print(
        f"Additional training complete. Total epochs: {som_loaded.metadata['total_epochs']}"
    )

    # Plot updated quantization error
    som_loaded.plot_quantization_error(
        show_plot=True, save_path="qe_plot_continued.png"
    )

    # Example 3: Different initialization strategies with visualization
    print("\nExample 3: Comparing initialization strategies")

    strategies = [InitStrategy.RANDOM, InitStrategy.PCA, InitStrategy.LINEAR]

    for i, strategy in enumerate(strategies):
        config = SOMConfig(
            width=10,
            height=10,
            n_features=3,
            n_iterations=100,
            init_strategy=strategy,
            seed=42,
        )
        som = SOM(config, verbose=False)
        som.fit(data)
        qe = som.quantization_error(data)
        print(f"{strategy.value}: QE = {qe:.4f}")

        # Save visualization for each strategy
        som.visualize_weights(
            show_plot=False, save_path=f"som_{strategy.value.lower()}.png"
        )

    # Example 4: Different distance metrics
    print("\nExample 4: Comparing distance metrics")

    metrics = [
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
        DistanceMetric.COSINE,
    ]

    for metric in metrics:
        config = SOMConfig(
            width=10,
            height=10,
            n_features=3,
            n_iterations=100,
            distance_metric=metric,
            seed=42,
        )
        som = SOM(config, verbose=False)
        som.fit(data)
        qe = som.quantization_error(data)
        print(f"{metric.value}: QE = {qe:.4f}")

    # Example 5: Testing with different feature counts and visualizations
    print("\nExample 5: Testing with different feature counts")
    for n_features in [1, 2, 3, 5]:
        test_data = np.random.random((50, n_features)).astype(np.float32)
        config = SOMConfig(
            width=8,
            height=8,
            n_features=n_features,
            n_iterations=50,
            init_strategy=InitStrategy.LINEAR,
            seed=42,
        )
        som = SOM(config, verbose=False)
        som.fit(test_data)
        print(f"n_features={n_features}: Training successful")

        # Visualize each case
        som.visualize_weights(
            show_plot=False, save_path=f"som_{n_features}features.png"
        )

    print("\nAll visualizations have been saved as PNG files:")
    print("- qe_plot.png: Quantization error over epochs")
    print("- som_weights.png: Main SOM weight visualization")
    print("- qe_plot_continued.png: QE after continued training")
    print("- som_random.png, som_pca.png, som_linear.png: Different init strategies")
    print("- som_1features.png through som_5features.png: Different feature counts")

    # Clean up
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    if os.path.exists("trained_som.pkl"):
        os.remove("trained_som.pkl")


if __name__ == "__main__":
    run_examples()
