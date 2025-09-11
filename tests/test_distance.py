"""
Tests for distance calculation utilities
"""

import pytest
import numpy as np
from som.distance import DistanceCalculator


@pytest.mark.unit
class TestDistanceCalculator:
    """Test distance calculation methods"""

    @pytest.mark.unit
    def test_euclidean_distance(self):
        a = np.array([[0, 0]])
        b = np.array([[1, 0]])

        distance = DistanceCalculator.euclidean(a, b)
        expected = np.array([1.0])

        np.testing.assert_array_almost_equal(distance, expected)

        # Test another case
        a = np.array([[1, 1]])
        b = np.array([[0, 1]])

        distance = DistanceCalculator.euclidean(a, b)
        expected = np.array([1.0])

        np.testing.assert_array_almost_equal(distance, expected)

    @pytest.mark.unit
    def test_manhattan_distance(self):
        a = np.array([[0, 0]])
        b = np.array([[1, 0]])

        distance = DistanceCalculator.manhattan(a, b)
        expected = np.array([1.0])

        np.testing.assert_array_almost_equal(distance, expected)

        # Test another case
        a = np.array([[1, 1]])
        b = np.array([[0, 0]])

        distance = DistanceCalculator.manhattan(a, b)
        expected = np.array([2.0])

        np.testing.assert_array_almost_equal(distance, expected)

    @pytest.mark.unit
    def test_chebyshev_distance(self):
        a = np.array([[0, 0], [1, 1]])
        b = np.array([[1, 0], [0, 1]])

        distances = DistanceCalculator.chebyshev(a, b)
        expected = np.array([1.0, 1.0])

        np.testing.assert_array_almost_equal(distances, expected)

    @pytest.mark.unit
    def test_cosine_distance(self):
        # Test orthogonal vectors (cosine distance = 1)
        a = np.array([[1, 0]])
        b = np.array([[0, 1]])

        distance = DistanceCalculator.cosine(a, b)
        np.testing.assert_array_almost_equal(distance, [1.0])

        # Test identical vectors (cosine distance = 0)
        a = np.array([[1, 1]])
        b = np.array([[1, 1]])

        distance = DistanceCalculator.cosine(a, b)
        np.testing.assert_array_almost_equal(distance, [0.0], decimal=7)

    @pytest.mark.unit
    def test_cosine_distance_parallel(self):
        # Test parallel vectors (cosine distance = 0)
        a = np.array([[2, 2]])
        b = np.array([[1, 1]])

        distance = DistanceCalculator.cosine(a, b)
        np.testing.assert_array_almost_equal(distance, [0.0], decimal=7)

    @pytest.mark.unit
    def test_cosine_distance_anti_parallel(self):
        # Test anti-parallel vectors (cosine distance = 2)
        a = np.array([[1, 1]])
        b = np.array([[-1, -1]])

        distance = DistanceCalculator.cosine(a, b)
        np.testing.assert_array_almost_equal(distance, [2.0], decimal=7)

    @pytest.mark.unit
    def test_distance_shapes(self):
        """Test that all distance functions handle different array shapes"""
        # 1D arrays
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        for dist_func in [
            DistanceCalculator.euclidean,
            DistanceCalculator.manhattan,
            DistanceCalculator.cosine,
            DistanceCalculator.chebyshev,
        ]:
            result = dist_func(a.reshape(1, -1), b.reshape(1, -1))
            assert result.shape == (1,)

        # 2D arrays
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        for dist_func in [
            DistanceCalculator.euclidean,
            DistanceCalculator.manhattan,
            DistanceCalculator.cosine,
            DistanceCalculator.chebyshev,
        ]:
            result = dist_func(a, b)
            assert result.shape == (2,)

    @pytest.mark.unit
    def test_zero_vectors_cosine(self):
        """Test cosine distance with zero vectors"""
        a = np.array([[0, 0]])
        b = np.array([[1, 1]])

        # Should handle zero vectors gracefully
        distance = DistanceCalculator.cosine(a, b)
        assert not np.isnan(distance[0])
        assert not np.isinf(distance[0])
