"""
Configuration classes and enums for SOM
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict
import numpy as np


class Topology(Enum):
    """Issue #15: Support different grid topologies"""

    RECTANGULAR = "rectangular"
    HEXAGONAL = "hexagonal"
    TOROIDAL = "toroidal"


class DecaySchedule(Enum):
    """Issue #18 & #19: Different decay schedules for learning rate"""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    INVERSE = "inverse"
    COSINE = "cosine"
    STEP = "step"


class LearningMode(Enum):
    """Issue #21: Different learning modes"""

    ONLINE = "online"
    BATCH = "batch"
    MINI_BATCH = "mini_batch"


class DistanceMetric(Enum):
    """ARCHITECTURE FIX #5: Support multiple distance metrics"""

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    CHEBYSHEV = "chebyshev"


class InitStrategy(Enum):
    """ARCHITECTURE FIX #6: Different initialization strategies"""

    RANDOM = "random"
    PCA = "pca"
    SAMPLE = "sample"
    LINEAR = "linear"


@dataclass
class SOMConfig:
    """Centralized configuration management for SOM parameters"""

    # Basic parameters
    width: int
    height: int
    n_features: int = 3
    n_iterations: int = 1000

    # Training parameters
    learning_mode: LearningMode = LearningMode.ONLINE
    batch_size: int = 32

    # Decay schedules (Issues #18, #19, #20)
    sigma_decay: DecaySchedule = DecaySchedule.EXPONENTIAL
    alpha_decay: DecaySchedule = DecaySchedule.EXPONENTIAL
    initial_sigma: Optional[float] = None  # Auto-calculated if None
    initial_alpha: float = 0.1
    min_sigma: float = 0.01
    min_alpha: float = 0.001
    warmup_steps: int = 0

    # Topology and distance (Issue #15, ARCHITECTURE FIX #5)
    topology: Topology = Topology.RECTANGULAR
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN

    # Optimization parameters
    cutoff_factor: float = 3.0  # Issues #13, #14
    momentum: float = 0.0  # Issue #24
    dtype: np.dtype = np.float32  # OPTIMIZATION

    # Convergence parameters (Issues #16, #17)
    early_stopping: bool = True
    convergence_tolerance: float = 1e-4
    patience: int = 10
    convergence_check_interval: int = 10

    # Performance parameters
    kdtree_rebuild_interval: int = (
        100  # Rebuild tree less frequently for better performance
    )

    # Initialization (ARCHITECTURE FIX #6)
    init_strategy: InitStrategy = InitStrategy.RANDOM

    # Persistence (ARCHITECTURE FIX #3, #4)
    checkpoint_interval: Optional[int] = None
    checkpoint_dir: str = "checkpoints"

    # Bounds (Issue #23)
    weight_bounds: Tuple[float, float] = (0.0, 1.0)

    # Reproducibility (Issue #11)
    seed: Optional[int] = None

    def __post_init__(self):
        """Auto-calculate initial sigma if not provided"""
        if self.initial_sigma is None:
            self.initial_sigma = max(self.width, self.height) / 2

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        config_dict = asdict(self)
        # Convert enums to strings
        for key, value in config_dict.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SOMConfig":
        """Create config from dictionary"""
        # Convert string back to enums
        enum_fields = {
            "topology": Topology,
            "sigma_decay": DecaySchedule,
            "alpha_decay": DecaySchedule,
            "learning_mode": LearningMode,
            "distance_metric": DistanceMetric,
            "init_strategy": InitStrategy,
        }
        for field_name, enum_class in enum_fields.items():
            if field_name in config_dict and isinstance(config_dict[field_name], str):
                config_dict[field_name] = enum_class(config_dict[field_name])
        return cls(**config_dict)
