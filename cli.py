"""
Command Line Interface for Kohonen SOM with observability
"""

import argparse
import json
import os
import sys
import structlog
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from som import (
    SOM,
    SOMConfig,
    InitStrategy,
    DistanceMetric,
    LearningMode,
    DecaySchedule,
    setup_logging,
    trace_operation,
)

# Initialize observability
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=False,  # Use console format for CLI
)

logger = structlog.get_logger()


def load_data(file_path: str, format: str = "auto") -> np.ndarray:
    """Load data from various formats"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Auto-detect format if not specified
    if format == "auto":
        format = path.suffix.lower()

    try:
        if format in [".csv", "csv"]:
            df = pd.read_csv(file_path)
            return df.select_dtypes(include=[np.number]).values.astype(np.float32)
        elif format in [".json", "json"]:
            with open(file_path, "r") as f:
                data = json.load(f)
            return np.array(data, dtype=np.float32)
        elif format in [".npy", "npy"]:
            return np.load(file_path).astype(np.float32)
        elif format in [".npz", "npz"]:
            loaded = np.load(file_path)
            # Use first array if multiple arrays in npz
            key = list(loaded.keys())[0]
            return loaded[key].astype(np.float32)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}")


def save_model(som: SOM, output_path: str) -> None:
    """Save trained SOM model"""
    try:
        som.save(output_path)
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}", file=sys.stderr)
        sys.exit(1)


def train_command(args) -> None:
    """Train a SOM model"""
    print(f"Loading data from: {args.input}")
    try:
        data = load_data(args.input, args.format)
        print(f"Data shape: {data.shape}")

        # Create configuration
        config = SOMConfig(
            width=args.width,
            height=args.height,
            n_features=data.shape[1],
            n_iterations=args.iterations,
            initial_alpha=args.learning_rate,
            initial_sigma=args.sigma,
            init_strategy=InitStrategy(args.init_strategy),
            distance_metric=DistanceMetric(args.distance_metric),
            learning_mode=LearningMode(args.learning_mode),
            seed=args.seed,
        )

        print(f"Training SOM: {args.width}x{args.height}, {args.iterations} iterations")
        print(f"Initialization: {args.init_strategy}, Distance: {args.distance_metric}")

        # Create and train SOM
        som = SOM(config, verbose=args.verbose)
        som.fit(data)

        # Calculate metrics
        qe = som.quantization_error(data)
        te = som.topographic_error(data)

        print(f"Training completed!")
        print(f"Quantization Error: {qe:.4f}")
        print(f"Topographic Error: {te:.4f}")

        # Save model
        save_model(som, args.output)

        # Save visualization if requested
        if args.visualize:
            viz_path = args.output.replace(".pkl", "_weights.png")
            som.visualize_weights(show_plot=False, save_path=viz_path)
            print(f"Weights visualization saved to: {viz_path}")

            qe_path = args.output.replace(".pkl", "_qe.png")
            som.plot_quantization_error(show_plot=False, save_path=qe_path)
            print(f"QE plot saved to: {qe_path}")

    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)


def predict_command(args) -> None:
    """Make predictions with a trained SOM"""
    print(f"Loading model from: {args.model}")
    try:
        som = SOM.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from: {args.input}")
    try:
        data = load_data(args.input, args.format)
        print(f"Data shape: {data.shape}")

        # Make predictions
        print("Making predictions...")
        predictions = som.predict(data)
        coordinates = som.transform(data)

        # Save results
        results = {
            "bmu_indices": predictions.tolist(),
            "coordinates": coordinates.tolist(),
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Predictions saved to: {args.output}")
        print(f"Predicted {len(predictions)} samples")

    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)


def visualize_command(args) -> None:
    """Visualize a trained SOM"""
    print(f"Loading model from: {args.model}")
    try:
        som = SOM.load(args.model)

        # Generate visualizations
        if args.type in ["weights", "all"]:
            weights_path = args.output or "som_weights.png"
            som.visualize_weights(show_plot=False, save_path=weights_path)
            print(f"Weights visualization saved to: {weights_path}")

        if args.type in ["training", "all"]:
            training_path = (
                args.output.replace(".png", "_training.png")
                if args.output
                else "som_training.png"
            )
            som.plot_quantization_error(show_plot=False, save_path=training_path)
            print(f"Training progress saved to: {training_path}")

    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def info_command(args) -> None:
    """Show information about a trained SOM"""
    print(f"Loading model from: {args.model}")
    try:
        som = SOM.load(args.model)

        info = som.get_info()

        print("\n=== SOM Model Information ===")
        print(f"Shape: {info['shape'][0]}x{info['shape'][1]}")
        print(f"Features: {info['n_features']}")
        print(f"Total Neurons: {info['n_neurons']}")
        print(f"Total Epochs: {info['total_epochs']}")
        print(f"Total Samples Seen: {info['total_samples']}")

        print("\n=== Configuration ===")
        config = info["config"]
        for key, value in config.items():
            if key not in ["weight_bounds"]:  # Skip verbose settings
                print(f"{key}: {value}")

    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Kohonen Self-Organizing Map CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new SOM model")
    train_parser.add_argument("input", help="Input data file")
    train_parser.add_argument(
        "--output", "-o", default="trained_som.pkl", help="Output model file"
    )
    train_parser.add_argument("--width", type=int, default=20, help="SOM width")
    train_parser.add_argument("--height", type=int, default=20, help="SOM height")
    train_parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of training iterations"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Initial learning rate"
    )
    train_parser.add_argument(
        "--sigma", type=float, help="Initial neighborhood radius (auto if not set)"
    )
    train_parser.add_argument(
        "--init-strategy",
        choices=["random", "pca", "sample", "linear"],
        default="random",
        help="Weight initialization strategy",
    )
    train_parser.add_argument(
        "--distance-metric",
        choices=["euclidean", "manhattan", "cosine", "chebyshev"],
        default="euclidean",
        help="Distance metric",
    )
    train_parser.add_argument(
        "--learning-mode",
        choices=["online", "batch", "mini_batch"],
        default="online",
        help="Learning mode",
    )
    train_parser.add_argument(
        "--format",
        choices=["auto", "csv", "json", "npy", "npz"],
        default="auto",
        help="Input data format",
    )
    train_parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility"
    )
    train_parser.add_argument(
        "--visualize", action="store_true", help="Save visualizations"
    )
    train_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Make predictions with trained model"
    )
    predict_parser.add_argument("model", help="Trained model file")
    predict_parser.add_argument("input", help="Input data file")
    predict_parser.add_argument(
        "--output", "-o", default="predictions.json", help="Output predictions file"
    )
    predict_parser.add_argument(
        "--format",
        choices=["auto", "csv", "json", "npy", "npz"],
        default="auto",
        help="Input data format",
    )

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize trained model")
    viz_parser.add_argument("model", help="Trained model file")
    viz_parser.add_argument("--output", "-o", help="Output image file")
    viz_parser.add_argument(
        "--type",
        choices=["weights", "training", "all"],
        default="weights",
        help="Visualization type",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="Trained model file")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "visualize":
        visualize_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "version":
        print("Kohonen SOM CLI v0.1.0")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
