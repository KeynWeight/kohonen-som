"""
Pytest configuration and fixtures for SOM tests
"""

import pytest
import numpy as np
from som import SOM, SOMConfig, InitStrategy, DistanceMetric, LearningMode


@pytest.fixture
def sample_data():
    """Generate sample 2D data for testing"""
    np.random.seed(42)
    return np.random.random((50, 3)).astype(np.float32)


@pytest.fixture
def small_data():
    """Generate small dataset for quick tests"""
    np.random.seed(42)
    return np.random.random((10, 2)).astype(np.float32)


@pytest.fixture
def basic_config():
    """Basic SOM configuration for testing"""
    return SOMConfig(width=5, height=5, n_features=3, n_iterations=10, seed=42)


@pytest.fixture
def minimal_config():
    """Minimal SOM configuration for quick tests"""
    return SOMConfig(width=3, height=3, n_features=2, n_iterations=5, seed=42)


@pytest.fixture
def trained_som(basic_config, sample_data):
    """Pre-trained SOM for testing"""
    som = SOM(basic_config, verbose=False)
    som.fit(sample_data)
    return som


@pytest.fixture
def all_init_strategies():
    """All initialization strategies for testing"""
    return [
        InitStrategy.RANDOM,
        InitStrategy.PCA,
        InitStrategy.SAMPLE,
        InitStrategy.LINEAR,
    ]


@pytest.fixture
def all_distance_metrics():
    """All distance metrics for testing"""
    return [
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
        DistanceMetric.COSINE,
        DistanceMetric.CHEBYSHEV,
    ]


@pytest.fixture
def all_learning_modes():
    """All learning modes for testing"""
    return [LearningMode.ONLINE, LearningMode.BATCH, LearningMode.MINI_BATCH]
