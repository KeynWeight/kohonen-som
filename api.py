"""
FastAPI web service for Kohonen SOM with observability
"""

import io
import json
import os
import time
import uuid
import structlog
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

# Set matplotlib backend to Agg (non-interactive) before importing pyplot
import matplotlib

matplotlib.use("Agg")

from som import (  # noqa: E402
    SOM,
    SOMConfig,
    InitStrategy,
    DistanceMetric,
    LearningMode,
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
from som.observability import log_request_metrics, CONTENT_TYPE_LATEST  # noqa: E402


def serialize_all(obj: Any) -> Any:
    """Recursively serialize any object to JSON-serializable format"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, type):
        # Convert Python types to strings
        return str(obj)
    elif isinstance(obj, (np.dtype, np.number, np.ndarray)):
        # Handle numpy types
        return str(obj)
    elif hasattr(obj, "__name__") and not isinstance(obj, (str, int, float, bool)):
        # Handle other non-serializable objects with names (enums, functions, etc.)
        return str(obj)
    elif isinstance(obj, dict):
        # Recursively serialize dictionaries
        return {k: serialize_all(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively serialize lists and tuples
        return [serialize_all(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Handle objects with __dict__ by converting to dict first
        return serialize_all(obj.__dict__)
    else:
        # Fallback: convert to string
        return str(obj)


# Pydantic models for API requests/responses
class SOMConfigRequest(BaseModel):
    """Configuration for SOM training"""

    width: int = Field(default=20, ge=1, le=1000)
    height: int = Field(default=20, ge=1, le=1000)
    n_features: Optional[int] = Field(default=None, ge=1)
    n_iterations: int = Field(default=1000, ge=1, le=10000)
    initial_alpha: float = Field(default=0.1, gt=0, le=1)
    initial_sigma: Optional[float] = Field(default=None, gt=0)
    min_alpha: float = Field(default=0.001, gt=0, le=1)
    min_sigma: float = Field(default=0.01, gt=0)
    init_strategy: InitStrategy = InitStrategy.RANDOM
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    learning_mode: LearningMode = LearningMode.ONLINE
    batch_size: int = Field(default=32, ge=1)
    seed: Optional[int] = Field(default=None, ge=0)
    early_stopping: bool = True
    convergence_tolerance: float = Field(default=1e-4, gt=0)
    patience: int = Field(default=10, ge=1)


class TrainingRequest(BaseModel):
    """Request to train a SOM"""

    data: List[List[float]] = Field(description="Training data as list of samples")
    config: SOMConfigRequest = Field(default_factory=SOMConfigRequest)


class TrainingResponse(BaseModel):
    """Response from SOM training"""

    model_id: str
    training_info: Dict[str, Any]
    quantization_error: float
    topographic_error: float
    message: str


class PredictionRequest(BaseModel):
    """Request for SOM predictions"""

    model_id: str
    data: List[List[float]]


class PredictionResponse(BaseModel):
    """Response from SOM predictions"""

    predictions: List[int]
    coordinates: List[List[float]]
    model_id: str


class ModelInfo(BaseModel):
    """Information about a SOM model"""

    model_id: str
    config: Dict[str, Any]
    shape: tuple
    n_features: int
    n_neurons: int
    total_epochs: int
    total_samples: int
    created_at: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    models_loaded: int


# Global storage for trained models (in production, use Redis/database)
models_storage: Dict[str, SOM] = {}

# Initialize observability
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=os.getenv("LOG_FORMAT", "json").lower() == "json",
)
setup_error_tracking()

logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="Kohonen SOM API",
    description="A REST API for training and using Self-Organizing Maps with observability",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Add observability middleware
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Add correlation ID to request
        request.state.correlation_id = correlation_id

        response = await call_next(request)

        # Log metrics
        duration = time.time() - start_time
        log_request_metrics(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration,
        )

        # Log request
        logger.info(
            "HTTP request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration_seconds=duration,
            correlation_id=correlation_id,
        )

        return response


app.add_middleware(RequestTracingMiddleware)
app.add_middleware(MetricsMiddleware)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "Kohonen SOM API", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed status"""
    try:
        health_status = get_health_status()
        health_status["models_loaded"] = len(models_storage)
        health_status["version"] = "0.1.0"

        # Update active models metric
        update_active_models_count(len(models_storage))

        logger.info("Health check requested", status=health_status["status"])
        return health_status

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "models_loaded": len(models_storage),
        }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    logger.debug("Metrics requested")
    return Response(content=get_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.post("/train", response_model=TrainingResponse)
async def train_som(request: TrainingRequest):
    """Train a new SOM model with observability"""

    with trace_operation(
        "som_training",
        width=request.config.width,
        height=request.config.height,
        data_shape=f"{len(request.data)}x{len(request.data[0]) if request.data else 0}",
    ) as correlation_id:
        try:
            logger.info("Starting SOM training", correlation_id=correlation_id)

            # Convert data to numpy array
            data = np.array(request.data, dtype=np.float32)

            if data.size == 0:
                raise HTTPException(status_code=400, detail="Empty data provided")

            if data.ndim != 2:
                raise HTTPException(
                    status_code=400, detail="Data must be 2-dimensional"
                )

            # Update config with actual number of features
            config_dict = request.config.model_dump()
            config_dict["n_features"] = data.shape[1]

            # Create SOM configuration
            som_config = SOMConfig(**config_dict)

            # Generate unique model ID
            model_id = str(uuid.uuid4())

            # Train SOM
            start_time = time.time()
            som = SOM(som_config, verbose=False)
            som.fit(data)
            training_duration = time.time() - start_time

            # Log training metrics
            log_training_metrics(
                width=som_config.width,
                height=som_config.height,
                duration=training_duration,
                iterations=som_config.n_iterations,
            )

            # Calculate metrics
            qe = float(som.quantization_error(data))
            te = float(som.topographic_error(data))

            # Store model
            models_storage[model_id] = som

            # Get training info (convert non-serializable types)
            info = som.get_info()
            # Serialize everything to handle all non-serializable types
            serialized_info = serialize_all(info)

            logger.info(
                "SOM training completed successfully",
                model_id=model_id,
                correlation_id=correlation_id,
                quantization_error=qe,
                topographic_error=te,
                training_duration=training_duration,
                data_samples=data.shape[0],
            )

            return TrainingResponse(
                model_id=model_id,
                training_info=serialized_info,
                quantization_error=qe,
                topographic_error=te,
                message=f"SOM trained successfully with {data.shape[0]} samples",
            )

        except ValueError as e:
            logger.error(
                "Training failed - invalid data",
                error=str(e),
                correlation_id=correlation_id,
            )
            raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
        except Exception as e:
            logger.error(
                "Training failed - unexpected error",
                error=str(e),
                correlation_id=correlation_id,
            )
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_som(request: PredictionRequest):
    """Make predictions with a trained SOM"""

    with trace_operation(
        "som_prediction",
        model_id=request.model_id,
        data_shape=f"{len(request.data)}x{len(request.data[0]) if request.data else 0}",
    ) as correlation_id:
        try:
            logger.info(
                "Starting SOM prediction",
                model_id=request.model_id,
                correlation_id=correlation_id,
            )

            # Log prediction metrics
            log_prediction_metrics()

            # Check if model exists
            if request.model_id not in models_storage:
                raise HTTPException(status_code=404, detail="Model not found")

            som = models_storage[request.model_id]

            # Convert data to numpy array
            data = np.array(request.data, dtype=np.float32)

            if data.size == 0:
                raise HTTPException(status_code=400, detail="Empty data provided")

            if data.ndim != 2:
                raise HTTPException(
                    status_code=400, detail="Data must be 2-dimensional"
                )

            if data.shape[1] != som.config.n_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {som.config.n_features} features, got {data.shape[1]}",
                )

            # Make predictions
            predictions = som.predict(data)
            coordinates = som.transform(data)

            logger.info(
                "SOM prediction completed successfully",
                model_id=request.model_id,
                correlation_id=correlation_id,
                data_samples=data.shape[0],
            )

            return PredictionResponse(
                predictions=predictions.tolist(),
                coordinates=coordinates.tolist(),
                model_id=request.model_id,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Prediction failed", error=str(e), correlation_id=correlation_id
            )
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=List[str])
async def list_models():
    """List all available models"""
    return list(models_storage.keys())


@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")

    som = models_storage[model_id]
    info = som.get_info()

    # Serialize config to handle non-serializable types
    config = serialize_all(info["config"])

    return ModelInfo(
        model_id=model_id,
        config=config,
        shape=info["shape"],
        n_features=info["n_features"],
        n_neurons=info["n_neurons"],
        total_epochs=info["total_epochs"],
        total_samples=info["total_samples"],
        created_at=som.metadata["creation_time"],
    )


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a specific model"""
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")

    del models_storage[model_id]
    return {"message": f"Model {model_id} deleted successfully"}


@app.get("/models/{model_id}/visualize")
async def visualize_model(model_id: str, type: str = "weights"):
    """Generate visualization for a model"""
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")

    if type not in ["weights", "training"]:
        raise HTTPException(
            status_code=400, detail="Type must be 'weights' or 'training'"
        )

    som = models_storage[model_id]

    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Use full path for both saving and returning
    viz_filename = f"{model_id}_{type}.png"
    viz_path = plots_dir / viz_filename

    try:
        if type == "weights":
            som.visualize_weights(show_plot=False, save_path=viz_filename)
        elif type == "training":
            som.plot_quantization_error(show_plot=False, save_path=viz_filename)

        return FileResponse(
            str(viz_path), media_type="image/png", filename=f"{model_id}_{type}.png"
        )
    except Exception as e:
        # Clean up visualization file if error occurs
        viz_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload data file and return parsed data info"""
    try:
        content = await file.read()

        # Try to parse as different formats
        if file.filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            data_array = np.array(data, dtype=np.float32)
        elif file.filename.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
            data_array = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Use CSV or JSON."
            )

        return {
            "filename": file.filename,
            "shape": data_array.shape,
            "data_preview": data_array[:5].tolist() if len(data_array) > 0 else [],
            "message": f"Data loaded successfully: {data_array.shape[0]} samples, {data_array.shape[1]} features",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


def main():
    """Run the FastAPI server"""
    import os
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RAILWAY_ENVIRONMENT") != "production"
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    main()
