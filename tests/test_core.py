"""
Tests for core SOM functionality
"""

import pytest
import numpy as np
import tempfile
import os
from som import SOM, SOMConfig, InitStrategy, DistanceMetric, LearningMode


@pytest.mark.unit
class TestSOMInitialization:
    """Test SOM initialization"""

    @pytest.mark.unit
    def test_som_creation(self, basic_config):
        som = SOM(basic_config, verbose=False)
        assert som.config == basic_config
        assert som.n_neurons == 25  # 5x5
        assert som.weights_flat is not None
        assert som.neuron_coords is not None

    @pytest.mark.unit
    def test_som_with_different_init_strategies(self, sample_data):
        for strategy in [InitStrategy.RANDOM, InitStrategy.LINEAR]:
            config = SOMConfig(
                width=3, height=3, n_features=3, init_strategy=strategy, seed=42
            )
            som = SOM(config, verbose=False)
            assert som.weights_flat is not None
            assert som.weights_flat.shape == (9, 3)

    @pytest.mark.unit
    def test_som_pca_initialization(self, sample_data):
        config = SOMConfig(
            width=3, height=3, n_features=3, init_strategy=InitStrategy.PCA, seed=42
        )
        som = SOM(config, verbose=False)
        som._initialize_weights(sample_data)
        assert som.weights_flat is not None
        assert som.weights_flat.shape == (9, 3)

    @pytest.mark.unit
    def test_som_sample_initialization(self, sample_data):
        config = SOMConfig(
            width=3, height=3, n_features=3, init_strategy=InitStrategy.SAMPLE, seed=42
        )
        som = SOM(config, verbose=False)
        som._initialize_weights(sample_data)
        assert som.weights_flat is not None
        assert som.weights_flat.shape == (9, 3)


@pytest.mark.integration
class TestSOMTraining:
    """Test SOM training functionality"""

    @pytest.mark.integration
    def test_basic_training(self, minimal_config, small_data):
        som = SOM(minimal_config, verbose=False)
        som.fit(small_data)

        assert som.metadata["total_epochs"] == 5
        assert som.metadata["total_samples_seen"] == 50  # 10 samples * 5 iterations
        assert len(som.metadata["training_history"]) == 5

    @pytest.mark.slow
    @pytest.mark.integration
    def test_training_with_different_learning_modes(self, small_data):
        for mode in [LearningMode.ONLINE, LearningMode.BATCH, LearningMode.MINI_BATCH]:
            config = SOMConfig(
                width=3,
                height=3,
                n_features=2,
                n_iterations=3,
                learning_mode=mode,
                seed=42,
            )
            som = SOM(config, verbose=False)
            som.fit(small_data)
            assert som.metadata["total_epochs"] == 3

    @pytest.mark.slow
    @pytest.mark.integration
    def test_training_with_different_distance_metrics(self, small_data):
        for metric in [
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.MANHATTAN,
            DistanceMetric.COSINE,
            DistanceMetric.CHEBYSHEV,
        ]:
            config = SOMConfig(
                width=3,
                height=3,
                n_features=2,
                n_iterations=2,
                distance_metric=metric,
                seed=42,
            )
            som = SOM(config, verbose=False)
            som.fit(small_data)
            assert som.metadata["total_epochs"] == 2

    @pytest.mark.integration
    def test_incremental_training(self, minimal_config, small_data):
        som = SOM(minimal_config, verbose=False)

        # First training
        som.fit(small_data)
        assert som.metadata["total_epochs"] == 5

        # Second training
        som.fit(small_data, n_iterations=3)
        assert som.metadata["total_epochs"] == 8


@pytest.mark.integration
class TestSOMPrediction:
    """Test SOM prediction functionality"""

    @pytest.mark.integration
    def test_predict(self, trained_som, sample_data):
        predictions = trained_som.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(0 <= p < trained_som.n_neurons for p in predictions)

    @pytest.mark.integration
    def test_transform(self, trained_som, sample_data):
        coordinates = trained_som.transform(sample_data)
        assert coordinates.shape == (len(sample_data), 2)

    @pytest.mark.integration
    def test_quantization_error(self, trained_som, sample_data):
        qe = trained_som.quantization_error(sample_data)
        assert isinstance(qe, float)
        assert qe >= 0

    @pytest.mark.integration
    def test_topographic_error(self, trained_som, sample_data):
        te = trained_som.topographic_error(sample_data)
        assert isinstance(te, float)
        assert 0 <= te <= 1


@pytest.mark.integration
class TestSOMUtilities:
    """Test SOM utility functions"""

    @pytest.mark.integration
    def test_get_weights(self, trained_som):
        weights = trained_som.get_weights()
        assert weights.shape == (5, 5, 3)  # width x height x features

    @pytest.mark.integration
    def test_get_info(self, trained_som):
        info = trained_som.get_info()
        assert "config" in info
        assert "metadata" in info
        assert "shape" in info
        assert info["n_neurons"] == 25

    @pytest.mark.slow
    @pytest.mark.integration
    def test_save_load(self, trained_som):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            filepath = f.name

        try:
            # Save model
            trained_som.save(filepath)
            assert os.path.exists(filepath)

            # Load model
            loaded_som = SOM.load(filepath)

            # Check that loaded model has same properties
            assert loaded_som.config.width == trained_som.config.width
            assert loaded_som.config.height == trained_som.config.height
            assert loaded_som.n_neurons == trained_som.n_neurons
            np.testing.assert_array_equal(
                loaded_som.weights_flat, trained_som.weights_flat
            )

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


@pytest.mark.unit
class TestSOMErrorHandling:
    """Test error handling"""

    @pytest.mark.unit
    def test_invalid_data_shapes(self, basic_config):
        som = SOM(basic_config, verbose=False)

        # Wrong number of features
        with pytest.raises(ValueError, match="Expected 3 features"):
            som.fit(np.random.random((10, 2)))

        # 1D data
        with pytest.raises(ValueError, match="must be 2D array"):
            som.fit(np.random.random(10))

        # Empty data
        with pytest.raises(ValueError, match="empty"):
            som.fit(np.array([]).reshape(0, 3))

    @pytest.mark.unit
    def test_nan_inf_data(self, basic_config):
        som = SOM(basic_config, verbose=False)

        # NaN data
        nan_data = np.random.random((10, 3))
        nan_data[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            som.fit(nan_data)

        # Inf data
        inf_data = np.random.random((10, 3))
        inf_data[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite"):
            som.fit(inf_data)

    @pytest.mark.unit
    def test_predict_without_training(self, basic_config):
        som = SOM(basic_config, verbose=False)
        som.weights_flat = None  # Simulate untrained state

        with pytest.raises(RuntimeError, match="not been trained"):
            som.predict(np.random.random((5, 3)))

    @pytest.mark.unit
    def test_invalid_file_operations(self, trained_som):
        # Test save to invalid path
        with pytest.raises(IOError):
            trained_som.save("/invalid/path/model.pkl")

        # Test load from non-existent file
        with pytest.raises(IOError):
            SOM.load("/non/existent/file.pkl")
