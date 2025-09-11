"""
Kohonen Self-Organizing Map (SOM) Package

A comprehensive, production-ready implementation of Self-Organizing Maps
with support for different topologies, distance metrics, and training modes.
"""

from .core import SOM
from .config import (
    SOMConfig,
    Topology,
    DecaySchedule,
    LearningMode,
    DistanceMetric,
    InitStrategy,
)
from .callbacks import Callback, CheckpointCallback, EarlyStoppingCallback
from .observability import (
    setup_logging,
    setup_error_tracking,
    trace_operation,
    get_metrics,
    get_health_status,
    log_training_metrics,
    log_prediction_metrics,
    update_active_models_count,
    RequestTracingMiddleware,
)

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "SOM",
    "SOMConfig",
    "Topology",
    "DecaySchedule",
    "LearningMode",
    "DistanceMetric",
    "InitStrategy",
    "Callback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "setup_logging",
    "setup_error_tracking",
    "trace_operation",
    "get_metrics",
    "get_health_status",
    "log_training_metrics",
    "log_prediction_metrics",
    "update_active_models_count",
    "RequestTracingMiddleware",
]
