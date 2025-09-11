"""
Distance calculation utilities for SOM
"""

import numpy as np


class DistanceCalculator:
    """Calculate distances using different metrics"""

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a - b, axis=-1)

    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(a - b), axis=-1)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Cosine distance = 1 - cosine similarity
        dot_product = np.sum(a * b, axis=-1)
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        similarity = dot_product / (norm_a * norm_b + 1e-8)
        return 1 - similarity

    @staticmethod
    def chebyshev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.max(np.abs(a - b), axis=-1)
