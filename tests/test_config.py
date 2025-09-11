"""
Tests for configuration classes and enums
"""

import pytest
import numpy as np
from som.config import (
    SOMConfig,
    Topology,
    DecaySchedule,
    LearningMode,
    DistanceMetric,
    InitStrategy,
)


@pytest.mark.unit
class TestEnums:
    """Test enum classes"""

    @pytest.mark.unit
    def test_topology_values(self):
        assert Topology.RECTANGULAR.value == "rectangular"
        assert Topology.HEXAGONAL.value == "hexagonal"
        assert Topology.TOROIDAL.value == "toroidal"

    @pytest.mark.unit
    def test_decay_schedule_values(self):
        assert DecaySchedule.EXPONENTIAL.value == "exponential"
        assert DecaySchedule.LINEAR.value == "linear"
        assert DecaySchedule.INVERSE.value == "inverse"
        assert DecaySchedule.COSINE.value == "cosine"
        assert DecaySchedule.STEP.value == "step"

    @pytest.mark.unit
    def test_learning_mode_values(self):
        assert LearningMode.ONLINE.value == "online"
        assert LearningMode.BATCH.value == "batch"
        assert LearningMode.MINI_BATCH.value == "mini_batch"

    @pytest.mark.unit
    def test_distance_metric_values(self):
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.MANHATTAN.value == "manhattan"
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.CHEBYSHEV.value == "chebyshev"

    @pytest.mark.unit
    def test_init_strategy_values(self):
        assert InitStrategy.RANDOM.value == "random"
        assert InitStrategy.PCA.value == "pca"
        assert InitStrategy.SAMPLE.value == "sample"
        assert InitStrategy.LINEAR.value == "linear"


@pytest.mark.unit
class TestSOMConfig:
    """Test SOMConfig class"""

    @pytest.mark.unit
    def test_basic_config_creation(self):
        config = SOMConfig(width=10, height=10, n_features=3)
        assert config.width == 10
        assert config.height == 10
        assert config.n_features == 3
        assert config.n_iterations == 1000  # default

    @pytest.mark.unit
    def test_auto_sigma_calculation(self):
        config = SOMConfig(width=20, height=15, n_features=3)
        # Should be max(width, height) / 2
        assert config.initial_sigma == 10.0

    @pytest.mark.unit
    def test_manual_sigma_setting(self):
        config = SOMConfig(width=10, height=10, n_features=3, initial_sigma=5.0)
        assert config.initial_sigma == 5.0

    @pytest.mark.unit
    def test_to_dict_conversion(self):
        config = SOMConfig(
            width=5,
            height=5,
            n_features=2,
            topology=Topology.HEXAGONAL,
            distance_metric=DistanceMetric.COSINE,
        )
        config_dict = config.to_dict()

        assert config_dict["width"] == 5
        assert config_dict["height"] == 5
        assert config_dict["topology"] == "hexagonal"  # Enum converted to string
        assert config_dict["distance_metric"] == "cosine"

    @pytest.mark.unit
    def test_from_dict_conversion(self):
        config_dict = {
            "width": 8,
            "height": 8,
            "n_features": 3,
            "topology": "toroidal",
            "distance_metric": "manhattan",
            "init_strategy": "pca",
        }
        config = SOMConfig.from_dict(config_dict)

        assert config.width == 8
        assert config.height == 8
        assert config.topology == Topology.TOROIDAL
        assert config.distance_metric == DistanceMetric.MANHATTAN
        assert config.init_strategy == InitStrategy.PCA

    @pytest.mark.unit
    def test_round_trip_conversion(self):
        original_config = SOMConfig(
            width=6,
            height=4,
            n_features=5,
            topology=Topology.HEXAGONAL,
            distance_metric=DistanceMetric.CHEBYSHEV,
            init_strategy=InitStrategy.LINEAR,
        )

        config_dict = original_config.to_dict()
        restored_config = SOMConfig.from_dict(config_dict)

        assert original_config.width == restored_config.width
        assert original_config.height == restored_config.height
        assert original_config.topology == restored_config.topology
        assert original_config.distance_metric == restored_config.distance_metric
        assert original_config.init_strategy == restored_config.init_strategy

    @pytest.mark.unit
    def test_default_values(self):
        config = SOMConfig(width=10, height=10)

        assert config.n_features == 3
        assert config.n_iterations == 1000
        assert config.learning_mode == LearningMode.ONLINE
        assert config.batch_size == 32
        assert config.topology == Topology.RECTANGULAR
        assert config.distance_metric == DistanceMetric.EUCLIDEAN
        assert config.init_strategy == InitStrategy.RANDOM

    @pytest.mark.unit
    def test_bounds_validation(self):
        config = SOMConfig(width=5, height=5, weight_bounds=(0.0, 1.0))
        assert config.weight_bounds == (0.0, 1.0)

    @pytest.mark.unit
    def test_seed_setting(self):
        config = SOMConfig(width=5, height=5, seed=123)
        assert config.seed == 123
