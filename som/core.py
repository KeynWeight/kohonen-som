"""
Core SOM implementation
"""

import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any, Tuple, Union
from sklearn.neighbors import KDTree, BallTree
from tqdm import tqdm

from .config import (
    SOMConfig,
    Topology,
    DecaySchedule,
    LearningMode,
    DistanceMetric,
    InitStrategy,
)
from .callbacks import Callback, CheckpointCallback
from .distance import DistanceCalculator
from .visualization import SOMVisualizer


def ensure_models_dir(filepath: str) -> str:
    """Ensure models directory exists and return full path"""
    if not os.path.isabs(filepath):
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        return str(models_dir / filepath)
    return filepath


class SOM:
    """
    Self-Organizing Map with comprehensive architecture improvements

    Fixes all 23 original issues plus 9 architecture improvements
    """

    # Constants for numerical stability and magic numbers
    UNDERFLOW_PROTECTION = -50  # Prevent exp() underflow
    NUMERICAL_EPSILON = 1e-8  # Small value for division safety
    COSINE_EPSILON = 1e-8  # Small value for cosine distance normalization
    HEXAGONAL_ADJACENCY_TOLERANCE = 1.1  # Tolerance for hexagonal adjacency

    def __init__(self, config: SOMConfig, verbose: bool = True):
        """
        Initialize SOM with configuration

        Args:
            config: SOMConfig object with all parameters
            verbose: Whether to print training progress
        """
        self.config = config
        self.verbose = verbose

        # Issue #11: Set random seed for reproducibility (use local RNG)
        if config.seed is not None:
            self.rng = np.random.RandomState(config.seed)
        else:
            self.rng = np.random.RandomState()

        # Initialize core attributes
        self.n_neurons = config.width * config.height
        self.weights_flat = None
        self.tree = None  # Changed from kdtree to tree (can be KDTree or BallTree)
        self.neuron_coords = None

        # ARCHITECTURE FIX #7: Store comprehensive metadata
        self.metadata = {
            "creation_time": datetime.now().isoformat(),
            "training_history": [],
            "total_epochs": 0,
            "total_samples_seen": 0,
            "config": config.to_dict(),
        }

        # Training state for incremental training (ARCHITECTURE FIX #4)
        self.training_state = {
            "epoch": 0,
            "best_error": float("inf"),
            "no_improvement_count": 0,
            "prev_update": None,
        }

        # Callbacks list (ARCHITECTURE FIX #9)
        self.callbacks: List[Callback] = []

        # Control flag for early stopping
        self.stop_training = False

        # Distance calculator based on metric (ARCHITECTURE FIX #5)
        self.distance_func = self._get_distance_function()

        # Initialize weights and structure
        self._initialize_structure()

    def _get_distance_function(self) -> Callable:
        """ARCHITECTURE FIX #5: Get appropriate distance function"""
        metric_map = {
            DistanceMetric.EUCLIDEAN: DistanceCalculator.euclidean,
            DistanceMetric.MANHATTAN: DistanceCalculator.manhattan,
            DistanceMetric.COSINE: DistanceCalculator.cosine,
            DistanceMetric.CHEBYSHEV: DistanceCalculator.chebyshev,
        }
        return metric_map[self.config.distance_metric]

    def _create_tree(self):
        """FIX: Create KDTree or BallTree with appropriate metric.
        For COSINE we don't build a tree (BallTree doesn't accept 'cosine'),
        instead we keep a normalized weights cache for brute-force queries.
        """
        # Clear any previous normalized cache
        self._normalized_weights = None

        if self.config.distance_metric == DistanceMetric.EUCLIDEAN:
            self.tree = KDTree(self.weights_flat)
        elif self.config.distance_metric == DistanceMetric.MANHATTAN:
            self.tree = BallTree(self.weights_flat, metric="manhattan")
        elif self.config.distance_metric == DistanceMetric.CHEBYSHEV:
            self.tree = BallTree(self.weights_flat, metric="chebyshev")
        elif self.config.distance_metric == DistanceMetric.COSINE:
            # BallTree/KDTree don't support 'cosine' reliably -> use brute-force on normalized vectors
            # Store normalized weights for fast dot-product based cosine distance calculations
            norms = np.linalg.norm(self.weights_flat, axis=1, keepdims=True) + 1e-8
            self._normalized_weights = (self.weights_flat / norms).astype(
                self.config.dtype
            )
            # keep tree = None to signal brute-force path in _query_tree
            self.tree = None
        else:
            # Fallback
            self.tree = KDTree(self.weights_flat)

    def _query_tree(self, data: np.ndarray, k: int = 1):
        """Query tree with appropriate preprocessing for cosine distance.
        Returns (distances, indices) shapes: (n_samples, k)
        """
        if self.config.distance_metric == DistanceMetric.COSINE:
            # Use brute-force cosine distance on normalized vectors
            # Normalize data
            data = np.asarray(data, dtype=self.config.dtype)
            norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
            normalized_data = data / norms

            # Ensure normalized_weights is available (created in _create_tree)
            if getattr(self, "_normalized_weights", None) is None:
                # If weights changed and tree wasn't rebuilt, compute normalized weights now
                self._normalized_weights = self.weights_flat / (
                    np.linalg.norm(self.weights_flat, axis=1, keepdims=True) + 1e-8
                )

            # Dot product -> cosine similarity; distance = 1 - similarity
            # shape (n_samples, n_neurons)
            sim = normalized_data @ self._normalized_weights.T
            dists = 1.0 - sim

            # For k == 1 return min; else return top-k
            if k == 1:
                idx = np.argmin(dists, axis=1)
                min_dists = dists[np.arange(dists.shape[0]), idx]
                return min_dists.reshape(-1, 1), idx.reshape(-1, 1)
            else:
                # Partial selection for speed, then sort
                idx_part = np.argpartition(dists, kth=k - 1, axis=1)[
                    :, :k
                ]  # (n_samples, k)
                # Now sort those k per-row by actual distance
                row_idx = np.arange(dists.shape[0])[:, None]
                sorted_order = np.argsort(dists[row_idx, idx_part], axis=1)
                idx_sorted = idx_part[row_idx, sorted_order]
                dists_sorted = dists[row_idx, idx_sorted]
                return dists_sorted, idx_sorted
        else:
            # Use tree-based query (KDTree / BallTree)
            if self.tree is None:
                # safety: (re)create tree if missing
                self._create_tree()
            # sklearn tree.query returns (distances, indices)
            return self.tree.query(data, k=k)

    def _initialize_structure(self):
        """Initialize neuron coordinates and weights"""
        # OPTIMIZATION: Create neuron coordinates instead of 4D distance matrix
        self.neuron_coords = self._create_neuron_coordinates()

        # Initialize weights if not already present (for new training)
        if self.weights_flat is None:
            self._initialize_weights()

    def _create_neuron_coordinates(self) -> np.ndarray:
        """
        OPTIMIZATION: Create 2D coordinates for all neurons
        Issue #15: Support different topologies
        """
        if self.config.topology == Topology.HEXAGONAL:
            coords = []
            for i in range(self.config.width):
                for j in range(self.config.height):
                    # Proper hexagonal grid: offset every other row in x-direction
                    x = i + (0.5 if j % 2 else 0)
                    y = j * np.sqrt(3) / 2  # Vertical spacing for equilateral triangles
                    coords.append([x, y])
            return np.array(coords, dtype=self.config.dtype)
        else:
            coords = np.array(
                [
                    [i, j]
                    for i in range(self.config.width)
                    for j in range(self.config.height)
                ],
                dtype=self.config.dtype,
            )
            return coords

    def _initialize_weights(self, data: Optional[np.ndarray] = None):
        """
        ARCHITECTURE FIX #6: Multiple initialization strategies
        """
        if self.config.init_strategy == InitStrategy.RANDOM:
            # Original random initialization
            self.weights_flat = self.rng.random(
                (self.n_neurons, self.config.n_features)
            ).astype(self.config.dtype)

        elif self.config.init_strategy == InitStrategy.PCA and data is not None:
            # FIX: Handle cases with fewer dimensions
            from sklearn.decomposition import PCA

            n_components = min(2, data.shape[1], self.config.n_features)

            if n_components < 2:
                # Handle 1D PCA case
                pca = PCA(n_components=1)
                pca.fit(data)

                # Create 1D gradient along principal component
                x_range = np.linspace(-3, 3, self.n_neurons)
                points = x_range.reshape(-1, 1)
                self.weights_flat = pca.inverse_transform(points).astype(
                    self.config.dtype
                )
            else:
                # Original 2D PCA logic
                pca = PCA(n_components=2)
                pca.fit(data)

                # Create grid along principal components
                x_range = np.linspace(-3, 3, self.config.width)
                y_range = np.linspace(-3, 3, self.config.height)

                self.weights_flat = np.zeros(
                    (self.n_neurons, self.config.n_features), dtype=self.config.dtype
                )

                idx = 0
                for i in range(self.config.width):
                    for j in range(self.config.height):
                        point = np.array([x_range[i], y_range[j]])
                        self.weights_flat[idx] = pca.inverse_transform(
                            point.reshape(1, -1)
                        )
                        idx += 1

            # Normalize to data range
            self.weights_flat = np.clip(self.weights_flat, 0, 1)

        elif self.config.init_strategy == InitStrategy.SAMPLE and data is not None:
            # Initialize with random samples from data
            indices = self.rng.choice(len(data), self.n_neurons, replace=True)
            self.weights_flat = data[indices].copy().astype(self.config.dtype)

        elif self.config.init_strategy == InitStrategy.LINEAR:
            # FIX: Handle arbitrary number of features
            self.weights_flat = np.zeros(
                (self.n_neurons, self.config.n_features), dtype=self.config.dtype
            )
            for idx in range(self.n_neurons):
                x = idx // self.config.height
                y = idx % self.config.height

                values = []
                # Create gradients that work even with width=1 or height=1
                if self.config.width > 1:
                    x_val = x / (self.config.width - 1)  # 0 to 1 range
                else:
                    x_val = 0.5  # Middle value for single column

                if self.config.height > 1:
                    y_val = y / (self.config.height - 1)  # 0 to 1 range
                else:
                    y_val = 0.5  # Middle value for single row

                for f in range(self.config.n_features):
                    if f == 0:
                        values.append(x_val)
                    elif f == 1:
                        values.append(y_val)
                    else:
                        # Create additional gradients for extra features
                        diagonal_val = (x_val + y_val + f * 0.1) / (2 + f * 0.1)
                        values.append(diagonal_val)

                self.weights_flat[idx] = values
        else:
            # Fallback to random
            self.weights_flat = self.rng.random(
                (self.n_neurons, self.config.n_features)
            ).astype(self.config.dtype)

        # Issue #23: Ensure weights are within bounds
        self.weights_flat = np.clip(
            self.weights_flat,
            self.config.weight_bounds[0],
            self.config.weight_bounds[1],
        )

    def _compute_neuron_distances(self, bmu_coord: np.ndarray) -> np.ndarray:
        """
        OPTIMIZATION: Compute distances from BMU to all neurons
        Issue #15: Handle different topologies
        """
        if self.config.topology == Topology.TOROIDAL:
            # Toroidal wrap-around distance
            dx = np.abs(self.neuron_coords[:, 0] - bmu_coord[0])
            dy = np.abs(self.neuron_coords[:, 1] - bmu_coord[1])
            dx = np.minimum(dx, self.config.width - dx)
            dy = np.minimum(dy, self.config.height - dy)
            distances = np.sqrt(dx**2 + dy**2)
        else:
            # Standard distance (works for rectangular and hexagonal)
            distances = np.linalg.norm(self.neuron_coords - bmu_coord, axis=1)

        # FIX: Ensure consistent dtype
        return distances.astype(self.config.dtype)

    def _get_neighborhood_mask(
        self, bmu_idx: int, sigma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        OPTIMIZATION: Get mask of neurons within neighborhood radius
        Issues #13, #14: Cutoff radius for efficiency
        """
        cutoff_radius = self.config.cutoff_factor * sigma
        bmu_coord = self.neuron_coords[bmu_idx]

        distances = self._compute_neuron_distances(bmu_coord)
        mask = distances <= cutoff_radius

        return mask, distances[mask]

    def _calculate_neighborhood(
        self, distances: np.ndarray, sigma: float
    ) -> np.ndarray:
        """
        Issues #13, #14, #22: Gaussian with cutoff and underflow protection
        """
        exponent = -(distances**2) / (2 * (sigma**2))
        exponent = np.maximum(exponent, -50)  # Issue #22: Prevent underflow
        return np.exp(exponent).astype(self.config.dtype)

    def _get_decay_value(
        self, t: int, t_max: int, initial: float, final: float, schedule: DecaySchedule
    ) -> float:
        """
        Issues #18, #19, #20: Flexible decay schedules with warm-up
        """
        if t < self.config.warmup_steps:
            return initial

        effective_t = t - self.config.warmup_steps
        effective_max = t_max - self.config.warmup_steps

        if effective_max <= 0:
            return final

        if schedule == DecaySchedule.LINEAR:
            return initial - (initial - final) * (effective_t / effective_max)
        elif schedule == DecaySchedule.INVERSE:
            return initial / (1 + effective_t / effective_max)
        elif schedule == DecaySchedule.COSINE:
            return (
                final
                + (initial - final)
                * (1 + np.cos(np.pi * effective_t / effective_max))
                / 2
            )
        elif schedule == DecaySchedule.STEP:
            drops = effective_t // 100
            return initial * (0.5**drops)
        else:  # EXPONENTIAL
            if initial > 0 and final > 0:
                decay_rate = -np.log(final / initial) / effective_max
                return initial * np.exp(-decay_rate * effective_t)
            return final

    def fit(
        self,
        data: np.ndarray,
        n_iterations: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> "SOM":
        """
        Train the SOM on data

        ARCHITECTURE FIX #4: Supports incremental training
        ARCHITECTURE FIX #9: Supports callbacks

        Args:
            data: Input data of shape (n_samples, n_features)
            n_iterations: Number of iterations (uses config if None)
            callbacks: List of callback objects

        Returns:
            self for method chaining
        """
        # Validate input data structure first
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(f"Input data must be 2D array, got {data.ndim}D")

        if data.shape[0] == 0:
            raise ValueError("Input data is empty")

        if data.shape[1] != self.config.n_features:
            raise ValueError(
                f"Expected {self.config.n_features} features, got {data.shape[1]}"
            )

        # Then validate data content
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or infinite values")

        # Issue #8: Normalize data
        data_normalized = self._normalize_data(data)

        # ARCHITECTURE FIX #6: Initialize weights with data if needed
        # Initialize weights if they don't exist, or if using data-dependent initialization
        should_initialize = self.weights_flat is None or (
            self.config.init_strategy in [InitStrategy.PCA, InitStrategy.SAMPLE]
            and self.metadata["total_epochs"] == 0
        )  # Use total_epochs instead of current epoch
        if should_initialize:
            self._initialize_weights(data_normalized)

        # Setup iterations
        if n_iterations is None:
            n_iterations = self.config.n_iterations

        # Setup callbacks (ARCHITECTURE FIX #9)
        self.callbacks = callbacks or []
        if self.config.checkpoint_interval:
            self.callbacks.append(
                CheckpointCallback(
                    self.config.checkpoint_dir, self.config.checkpoint_interval
                )
            )

        # Call training begin callbacks
        for callback in self.callbacks:
            callback.on_training_begin(self)

        # Main training loop
        epochs_completed = self._train_loop(data_normalized, n_iterations)

        # Call training end callbacks
        for callback in self.callbacks:
            callback.on_training_end(self)

        # Update metadata (ARCHITECTURE FIX #7)
        self.metadata["total_epochs"] += epochs_completed
        self.metadata["total_samples_seen"] += len(data) * epochs_completed
        self.metadata["last_training"] = datetime.now().isoformat()

        return self

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Issue #8: Normalize input data with proper handling of constant features"""
        data = data.astype(self.config.dtype)
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_range = data_max - data_min

        # For constant features (range=0), keep the original value but normalize to weight bounds center
        constant_mask = data_range == 0
        if np.any(constant_mask):
            # Set constant features to the center of weight bounds
            bounds_center = (
                self.config.weight_bounds[0] + self.config.weight_bounds[1]
            ) / 2
            normalized_data = (data - data_min) / np.where(
                data_range == 0, 1, data_range
            )
            normalized_data[:, constant_mask] = bounds_center
            return normalized_data
        else:
            # Handle division by zero for constant features
            safe_range = np.where(data_range == 0, 1.0, data_range)
            normalized = (data - data_min) / safe_range

            # For constant features, set to middle of [0,1] range
            constant_mask = data_range == 0
            if np.any(constant_mask):
                normalized[:, constant_mask] = 0.5

            return normalized

    def _train_loop(self, data: np.ndarray, n_iterations: int) -> int:
        """Main training loop with all optimizations"""
        # FIX: Initialize tree with correct metric
        self._create_tree()

        # Setup progress bar (Issue #10)
        iterator = range(
            self.training_state["epoch"], self.training_state["epoch"] + n_iterations
        )
        if self.verbose:
            iterator = tqdm(iterator, desc="Training SOM")

        # FIX: Initialize momentum tracking consistently
        if self.training_state["prev_update"] is None:
            self.training_state["prev_update"] = np.zeros_like(self.weights_flat)

        epochs_completed = 0
        for t in iterator:
            # Epoch begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(t, self)

            # Check early stopping flag
            if self.stop_training:
                if self.verbose:
                    print(f"Training stopped at epoch {t}")
                break

            # Get current parameters (Issues #18, #19, #20)
            sigma = self._get_decay_value(
                t,
                self.config.n_iterations,
                self.config.initial_sigma,
                self.config.min_sigma,
                self.config.sigma_decay,
            )
            alpha = self._get_decay_value(
                t,
                self.config.n_iterations,
                self.config.initial_alpha,
                self.config.min_alpha,
                self.config.alpha_decay,
            )

            # Issue #7: Shuffle data
            shuffled_indices = self.rng.permutation(len(data))
            shuffled_data = data[shuffled_indices]

            # Process based on learning mode (Issue #21)
            epoch_metrics = self._process_epoch(shuffled_data, sigma, alpha)

            # FIX: Update tree periodically with correct metric
            if (t + 1) % self.config.kdtree_rebuild_interval == 0:
                self._create_tree()

            # Check convergence (Issues #16, #17)
            if (
                self.config.early_stopping
                and t % self.config.convergence_check_interval == 0
            ):
                if self._check_convergence(epoch_metrics):
                    if self.verbose:
                        print(f"\nConverged after {t+1} iterations")
                    break

            # Update display
            if self.verbose:
                iterator.set_postfix(
                    {
                        "QE": f"{epoch_metrics.get('qe', 0):.4f}",
                        "σ": f"{sigma:.3f}",
                        "α": f"{alpha:.4f}",
                    }
                )

            # Store metrics (ARCHITECTURE FIX #7)
            self.metadata["training_history"].append(
                {"epoch": t, "metrics": epoch_metrics, "sigma": sigma, "alpha": alpha}
            )

            # Epoch end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(t, self, epoch_metrics)

            # Update training state
            self.training_state["epoch"] = t + 1
            epochs_completed += 1

        return epochs_completed

    def _process_epoch(self, data: np.ndarray, sigma: float, alpha: float) -> Dict:
        """Process one epoch of training"""
        metrics = {"qe": 0}

        if self.config.learning_mode == LearningMode.BATCH:
            metrics = self._process_batch(data, sigma, alpha)
        elif self.config.learning_mode == LearningMode.MINI_BATCH:
            metrics = self._process_mini_batch(data, sigma, alpha)
        else:  # ONLINE
            metrics = self._process_online(data, sigma, alpha)

        # Issue #23: Clip weights
        self.weights_flat = np.clip(
            self.weights_flat,
            self.config.weight_bounds[0],
            self.config.weight_bounds[1],
        )

        return metrics

    def _process_batch(self, data: np.ndarray, sigma: float, alpha: float) -> Dict:
        """OPTIMIZATION: Vectorized batch processing"""
        batch_updates = np.zeros_like(self.weights_flat)

        # Find all BMUs at once
        distances, bmu_indices = self._query_tree(data, k=1)
        bmu_indices = bmu_indices.flatten()
        total_error = np.sum(distances**2)

        # Group by BMU for efficiency
        unique_bmus = np.unique(bmu_indices)
        for bmu_idx in unique_bmus:
            sample_mask = bmu_indices == bmu_idx
            samples_for_bmu = data[sample_mask]

            # OPTIMIZATION: Windowed update
            neighbor_mask, neighbor_distances = self._get_neighborhood_mask(
                bmu_idx, sigma
            )

            if np.any(neighbor_mask):
                theta = self._calculate_neighborhood(neighbor_distances, sigma)
                affected_indices = np.where(neighbor_mask)[0]

                for sample in samples_for_bmu:
                    batch_updates[affected_indices] += theta[:, np.newaxis] * (
                        sample - self.weights_flat[affected_indices]
                    )

        # Apply updates
        batch_updates /= len(data)

        # Issue #24: FIX - Consistent momentum application
        if self.config.momentum > 0:
            batch_updates = (
                self.config.momentum * self.training_state["prev_update"]
                + (1 - self.config.momentum) * batch_updates
            )
            self.training_state["prev_update"] = batch_updates.copy()

        self.weights_flat += alpha * batch_updates

        return {"qe": total_error / len(data)}

    def _process_mini_batch(self, data: np.ndarray, sigma: float, alpha: float) -> Dict:
        """Process mini-batches"""
        n_batches = (len(data) + self.config.batch_size - 1) // self.config.batch_size
        total_error = 0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(data))
            batch_data = data[start_idx:end_idx]

            # Process similar to batch mode
            batch_metrics = self._process_batch(batch_data, sigma, alpha)
            total_error += batch_metrics["qe"] * len(batch_data)

        return {"qe": total_error / len(data)}

    def _process_online(self, data: np.ndarray, sigma: float, alpha: float) -> Dict:
        """Process samples one by one"""
        total_error = 0

        for sample in data:
            # Find BMU
            distance, bmu_idx = self._query_tree(sample.reshape(1, -1), k=1)
            bmu_idx = bmu_idx[0, 0]
            total_error += distance[0, 0] ** 2

            # OPTIMIZATION: Windowed update
            neighbor_mask, neighbor_distances = self._get_neighborhood_mask(
                bmu_idx, sigma
            )

            if np.any(neighbor_mask):
                theta = self._calculate_neighborhood(neighbor_distances, sigma)
                affected_indices = np.where(neighbor_mask)[0]

                update = theta[:, np.newaxis] * (
                    sample - self.weights_flat[affected_indices]
                )

                # FIX: Use consistent momentum approach (same as batch mode)
                if self.config.momentum > 0:
                    # Apply momentum to affected indices only
                    prev_update = self.training_state["prev_update"][affected_indices]
                    update = (
                        self.config.momentum * prev_update
                        + (1 - self.config.momentum) * update
                    )
                    # Update momentum state for affected indices
                    self.training_state["prev_update"][affected_indices] = update

                self.weights_flat[affected_indices] += alpha * update

        return {"qe": total_error / len(data)}

    def _check_convergence(self, metrics: Dict) -> bool:
        """Issues #16, #17: Check for convergence"""
        current_error = metrics.get("qe", float("inf"))

        if (
            abs(self.training_state["best_error"] - current_error)
            < self.config.convergence_tolerance
        ):
            self.training_state["no_improvement_count"] += 1
            if self.training_state["no_improvement_count"] >= self.config.patience:
                return True
        else:
            self.training_state["no_improvement_count"] = 0
            if current_error < self.training_state["best_error"]:
                self.training_state["best_error"] = current_error

        return False

    def _check_trained(self):
        """Check if SOM has been trained, raise informative error if not"""
        if self.weights_flat is None:
            raise RuntimeError("SOM has not been trained yet. Call fit() first.")
        if self.tree is None:
            # Try to rebuild tree if weights exist but tree is missing
            self._create_tree()

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Find BMU indices for input data"""
        self._check_trained()

        # FIX: Check for NaN/inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or infinite values")

        data_normalized = self._normalize_data(data)
        _, bmu_indices = self._query_tree(data_normalized, k=1)
        return bmu_indices.flatten()

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        FIX: Transform data to 2D grid coordinates using neuron_coords directly
        """
        self._check_trained()
        bmu_indices = self.predict(data)
        # Directly use the neuron coordinates instead of recalculating
        return self.neuron_coords[bmu_indices]

    def get_weights(self) -> np.ndarray:
        """Get weights in grid format"""
        self._check_trained()
        return self.weights_flat.reshape(
            self.config.width, self.config.height, self.config.n_features
        )

    def quantization_error(self, data: np.ndarray) -> float:
        """Calculate quantization error for data"""
        self._check_trained()

        # FIX: Check for NaN/inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or infinite values")

        data_normalized = self._normalize_data(data)
        distances, _ = self._query_tree(data_normalized, k=1)
        return np.mean(distances**2)

    def topographic_error(self, data: np.ndarray) -> float:
        """
        FIX: Calculate topographic error accounting for different topologies
        """
        self._check_trained()

        # FIX: Check for NaN/inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or infinite values")

        data_normalized = self._normalize_data(data)

        # Validate we can find 2 neighbors
        k = min(2, self.n_neurons)
        if k < 2:
            return 0.0  # No topographic error possible with < 2 neurons

        distances, indices = self._query_tree(data_normalized, k=k)

        # Handle case where query returns fewer neighbors than requested
        if indices.shape[1] < 2:
            return 0.0  # Not enough neighbors for topographic error

        errors = 0
        for i in range(len(indices)):
            idx1, idx2 = indices[i, 0], indices[i, 1]
            coord1 = self.neuron_coords[idx1]
            coord2 = self.neuron_coords[idx2]

            # Calculate distance based on topology
            if self.config.topology == Topology.HEXAGONAL:
                # For hexagonal, adjacent means distance <= 1
                distance = np.linalg.norm(coord1 - coord2)
                if distance > 1.1:  # Not adjacent (with small tolerance)
                    errors += 1
            elif self.config.topology == Topology.TOROIDAL:
                # For toroidal, check wrap-around distance
                dx = np.abs(coord1[0] - coord2[0])
                dy = np.abs(coord1[1] - coord2[1])
                dx = min(dx, self.config.width - dx)
                dy = min(dy, self.config.height - dy)
                distance = np.sqrt(dx**2 + dy**2)
                if distance > np.sqrt(2):  # Not adjacent
                    errors += 1
            else:  # RECTANGULAR
                # Check if BMU and 2nd BMU are adjacent
                distance = np.linalg.norm(coord1 - coord2)
                if distance > np.sqrt(2):  # Not adjacent
                    errors += 1

        return errors / len(data)

    # ARCHITECTURE FIX #3: Save/Load functionality with error handling
    def save(self, filepath: str):
        """Save trained model to file"""
        full_path = ensure_models_dir(filepath)

        save_data = {
            "config": self.config.to_dict(),
            "weights": self.weights_flat,
            "metadata": self.metadata,
            "training_state": self.training_state,
        }

        try:
            with open(full_path, "wb") as f:
                pickle.dump(save_data, f)

            if self.verbose:
                print(f"Model saved to {full_path}")
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save model to {full_path}: {e}")

    @classmethod
    def load(cls, filepath: str) -> "SOM":
        """Load trained model from file"""
        full_path = ensure_models_dir(filepath)

        try:
            with open(full_path, "rb") as f:
                save_data = pickle.load(f)
        except (IOError, OSError) as e:
            raise IOError(f"Failed to load model from {full_path}: {e}")

        # Reconstruct SOM
        config = SOMConfig.from_dict(save_data["config"])
        som = cls(config, verbose=False)
        som.weights_flat = save_data["weights"]
        som.metadata = save_data["metadata"]
        som.training_state = save_data["training_state"]

        # FIX: Rebuild tree with correct metric
        som._create_tree()

        return som

    # ARCHITECTURE FIX #7: Get comprehensive info
    def get_info(self) -> Dict:
        """Get comprehensive information about the SOM"""
        return {
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "shape": (self.config.width, self.config.height),
            "n_neurons": self.n_neurons,
            "n_features": self.config.n_features,
            "total_epochs": self.metadata["total_epochs"],
            "total_samples": self.metadata["total_samples_seen"],
        }

    # Visualization methods using the new visualizer
    def plot_quantization_error(self, show_plot=True, save_path="qe_plot.png"):
        """Plot quantization error over training epochs"""
        SOMVisualizer.plot_quantization_error(self, show_plot, save_path)
        return self

    def visualize_weights(self, show_plot=True, save_path="som_weights.png"):
        """Visualize SOM weights as an image"""
        SOMVisualizer.visualize_weights(self, show_plot, save_path)
        return self

    def plot_training_progress(self, show_plot=True, save_path="training_progress.png"):
        """Plot comprehensive training progress"""
        SOMVisualizer.plot_training_progress(self, show_plot, save_path)
        return self
