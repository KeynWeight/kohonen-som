"""
Observability infrastructure for Kohonen SOM
Provides structured logging, metrics, tracing, and error tracking
"""

import os
import time
import uuid
import psutil
import structlog
from typing import Dict, Any, Optional
from contextlib import contextmanager
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# Prometheus Metrics
REQUESTS_TOTAL = Counter(
    "kohonen_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "kohonen_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

TRAINING_DURATION = Histogram(
    "kohonen_training_duration_seconds",
    "SOM training duration in seconds",
    ["width", "height"],
)

TRAINING_ITERATIONS = Counter(
    "kohonen_training_iterations_total", "Total training iterations completed"
)

MODELS_CREATED = Counter(
    "kohonen_models_created_total", "Total number of models created"
)

MODELS_ACTIVE = Gauge("kohonen_models_active", "Number of currently stored models")

PREDICTION_REQUESTS = Counter("kohonen_predictions_total", "Total prediction requests")

SYSTEM_MEMORY_USAGE = Gauge(
    "kohonen_system_memory_usage_bytes", "System memory usage in bytes"
)

SYSTEM_CPU_USAGE = Gauge(
    "kohonen_system_cpu_usage_percent", "System CPU usage percentage"
)


class CorrelationIDProcessor:
    """Add correlation ID to log entries"""

    def __call__(self, logger, method_name, event_dict):
        # Add correlation ID if not present
        if "correlation_id" not in event_dict:
            event_dict["correlation_id"] = getattr(
                event_dict.get("request"), "correlation_id", "unknown"
            )
        return event_dict


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure structured logging with structlog"""

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        CorrelationIDProcessor(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )


def setup_error_tracking() -> None:
    """Configure error tracking (placeholder for future integrations)"""
    # For production, you could integrate with:
    # - Sentry (paid service)
    # - Self-hosted error tracking
    # - Cloud provider error tracking (AWS CloudWatch, etc.)
    pass


def get_correlation_id() -> str:
    """Generate a new correlation ID"""
    return str(uuid.uuid4())


@contextmanager
def trace_operation(operation_name: str, **extra_context):
    """Context manager for tracing operations with metrics and logging"""
    logger = structlog.get_logger()
    correlation_id = get_correlation_id()
    start_time = time.time()

    logger.info(
        "Operation started",
        operation=operation_name,
        correlation_id=correlation_id,
        **extra_context,
    )

    try:
        yield correlation_id
        duration = time.time() - start_time
        logger.info(
            "Operation completed",
            operation=operation_name,
            correlation_id=correlation_id,
            duration_seconds=duration,
            **extra_context,
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Operation failed",
            operation=operation_name,
            correlation_id=correlation_id,
            duration_seconds=duration,
            error=str(e),
            error_type=type(e).__name__,
            **extra_context,
        )
        raise


def update_system_metrics():
    """Update system-level metrics"""
    try:
        # Memory usage
        memory_info = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory_info.used)

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        SYSTEM_CPU_USAGE.set(cpu_percent)

    except Exception as e:
        logger = structlog.get_logger()
        logger.error("Failed to update system metrics", error=str(e))


def get_metrics() -> str:
    """Get Prometheus metrics in text format"""
    update_system_metrics()
    return generate_latest()


def log_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Log request metrics to Prometheus"""
    REQUESTS_TOTAL.labels(
        method=method, endpoint=endpoint, status_code=status_code
    ).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def log_training_metrics(width: int, height: int, duration: float, iterations: int):
    """Log training metrics to Prometheus"""
    TRAINING_DURATION.labels(width=str(width), height=str(height)).observe(duration)
    TRAINING_ITERATIONS.inc(iterations)
    MODELS_CREATED.inc()


def log_prediction_metrics():
    """Log prediction metrics to Prometheus"""
    PREDICTION_REQUESTS.inc()


def update_active_models_count(count: int):
    """Update the count of active models"""
    MODELS_ACTIVE.set(count)


class RequestTracingMiddleware:
    """Middleware to add correlation IDs to requests"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add correlation ID to request
            correlation_id = get_correlation_id()
            scope["correlation_id"] = correlation_id

            # Add to headers for client reference
            async def send_with_correlation_id(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"x-correlation-id"] = correlation_id.encode()
                    message["headers"] = list(headers.items())
                await send(message)

            await self.app(scope, receive, send_with_correlation_id)
        else:
            await self.app(scope, receive, send)


# Health check utilities
def get_health_status() -> Dict[str, Any]:
    """Get detailed health status"""
    try:
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage("/")

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "memory": {
                    "total": memory_info.total,
                    "available": memory_info.available,
                    "used": memory_info.used,
                    "percentage": memory_info.percent,
                },
                "cpu": {"usage_percent": psutil.cpu_percent(interval=1)},
                "disk": {
                    "total": disk_info.total,
                    "used": disk_info.used,
                    "free": disk_info.free,
                    "percentage": (disk_info.used / disk_info.total) * 100,
                },
            },
            "application": {
                "version": os.getenv("APP_VERSION", "unknown"),
                "environment": os.getenv("ENVIRONMENT", "development"),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "timestamp": time.time(), "error": str(e)}
