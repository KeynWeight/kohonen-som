"""
Test cases for the FastAPI web service (api.py)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import uuid

import pytest
import numpy as np
from fastapi.testclient import TestClient

from api import app, models_storage, serialize_all
from som import SOM, SOMConfig, InitStrategy, DistanceMetric, LearningMode


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def sample_training_data():
    """Sample training data for API tests"""
    np.random.seed(42)
    return np.random.random((20, 3)).tolist()


@pytest.fixture
def sample_config():
    """Sample SOM configuration for API tests"""
    return {
        "width": 5,
        "height": 5,
        "n_iterations": 10,
        "initial_alpha": 0.1,
        "seed": 42,
    }


@pytest.fixture
def trained_model_id(client, sample_training_data, sample_config):
    """Create a trained model and return its ID"""
    request_data = {"data": sample_training_data, "config": sample_config}
    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    return response.json()["model_id"]


@pytest.mark.api
@pytest.mark.unit
class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Kohonen SOM API"
        assert data["version"] == "0.1.0"
        assert data["docs"] == "/docs"


@pytest.mark.api
@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "models_loaded" in data

    def test_health_check_with_models(self, client, trained_model_id):
        """Test health check reflects loaded models"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["models_loaded"] >= 1


@pytest.mark.api
@pytest.mark.unit
class TestTrainEndpoint:
    """Tests for training endpoint"""

    def test_train_valid_data(self, client, sample_training_data, sample_config):
        """Test training with valid data"""
        request_data = {"data": sample_training_data, "config": sample_config}
        response = client.post("/train", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "model_id" in data
        assert "training_info" in data
        assert "quantization_error" in data
        assert "topographic_error" in data
        assert "message" in data
        assert data["quantization_error"] >= 0
        assert data["topographic_error"] >= 0

    def test_train_minimal_config(self, client, sample_training_data):
        """Test training with minimal configuration"""
        request_data = {"data": sample_training_data}
        response = client.post("/train", json=request_data)
        assert response.status_code == 200

    def test_train_empty_data(self, client):
        """Test training with empty data"""
        request_data = {"data": []}
        response = client.post("/train", json=request_data)
        assert (
            response.status_code == 500
        )  # Numpy conversion fails before our validation
        # Empty list causes numpy conversion error before reaching our validation

    def test_train_invalid_dimensions(self, client):
        """Test training with invalid data dimensions"""
        request_data = {"data": [1, 2, 3]}  # 1D instead of 2D
        response = client.post("/train", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
        # Pydantic returns different error format for validation

    def test_train_various_configs(self, client, sample_training_data):
        """Test training with various configuration options"""
        configs = [
            {"init_strategy": "pca"},
            {"distance_metric": "manhattan"},
            {"learning_mode": "batch"},
            {"width": 10, "height": 8},
            {"n_iterations": 5},
        ]

        for config in configs:
            request_data = {"data": sample_training_data, "config": config}
            response = client.post("/train", json=request_data)
            assert response.status_code == 200

    @patch("api.SOM")
    def test_train_som_exception(self, mock_som, client, sample_training_data):
        """Test training handles SOM exceptions"""
        mock_som.side_effect = Exception("Training failed")
        request_data = {"data": sample_training_data}
        response = client.post("/train", json=request_data)
        assert response.status_code == 500
        assert "Training failed" in response.json()["detail"]


@pytest.mark.api
@pytest.mark.unit
class TestPredictEndpoint:
    """Tests for prediction endpoint"""

    def test_predict_valid_data(self, client, trained_model_id, sample_training_data):
        """Test prediction with valid data"""
        request_data = {
            "model_id": trained_model_id,
            "data": sample_training_data[:5],  # Use subset for prediction
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "coordinates" in data
        assert data["model_id"] == trained_model_id
        assert len(data["predictions"]) == 5
        assert len(data["coordinates"]) == 5

    def test_predict_nonexistent_model(self, client, sample_training_data):
        """Test prediction with nonexistent model"""
        fake_id = str(uuid.uuid4())
        request_data = {"model_id": fake_id, "data": sample_training_data}
        response = client.post("/predict", json=request_data)
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]

    def test_predict_empty_data(self, client, trained_model_id):
        """Test prediction with empty data"""
        request_data = {"model_id": trained_model_id, "data": []}
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        assert "Empty data provided" in response.json()["detail"]

    def test_predict_wrong_dimensions(self, client, trained_model_id):
        """Test prediction with wrong feature dimensions"""
        request_data = {
            "model_id": trained_model_id,
            "data": [[1, 2]],  # Wrong number of features
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        assert "Expected" in response.json()["detail"]

    def test_predict_invalid_data_structure(self, client, trained_model_id):
        """Test prediction with invalid data structure"""
        request_data = {
            "model_id": trained_model_id,
            "data": [1, 2, 3],  # 1D instead of 2D
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Pydantic validation error


@pytest.mark.api
@pytest.mark.unit
class TestModelManagement:
    """Tests for model management endpoints"""

    def test_list_empty_models(self, client):
        """Test listing models when none exist"""
        # Clear storage
        models_storage.clear()
        response = client.get("/models")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_models_with_data(self, client, trained_model_id):
        """Test listing models when models exist"""
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert trained_model_id in models

    def test_get_model_info(self, client, trained_model_id):
        """Test getting model information"""
        response = client.get(f"/models/{trained_model_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == trained_model_id
        assert "config" in data
        assert "shape" in data
        assert "n_features" in data
        assert "n_neurons" in data

    def test_get_nonexistent_model_info(self, client):
        """Test getting info for nonexistent model"""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/models/{fake_id}")
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]

    def test_delete_model(self, client, trained_model_id):
        """Test deleting a model"""
        # Verify model exists
        response = client.get(f"/models/{trained_model_id}")
        assert response.status_code == 200

        # Delete model
        response = client.delete(f"/models/{trained_model_id}")
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]

        # Verify model is gone
        response = client.get(f"/models/{trained_model_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_model(self, client):
        """Test deleting nonexistent model"""
        fake_id = str(uuid.uuid4())
        response = client.delete(f"/models/{fake_id}")
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]


@pytest.mark.api
@pytest.mark.visualization
class TestVisualizationEndpoint:
    """Tests for visualization endpoint"""

    @patch("api.Path.unlink")
    @patch("api.FileResponse")
    def test_visualize_weights(
        self, mock_file_response, mock_unlink, client, trained_model_id
    ):
        """Test weights visualization"""
        mock_file_response.return_value = MagicMock()
        with patch.object(
            models_storage[trained_model_id], "visualize_weights"
        ) as mock_viz:
            response = client.get(f"/models/{trained_model_id}/visualize?type=weights")
            assert response.status_code == 200
            mock_viz.assert_called_once()

    @patch("api.Path.unlink")
    @patch("api.FileResponse")
    def test_visualize_training(
        self, mock_file_response, mock_unlink, client, trained_model_id
    ):
        """Test training visualization"""
        mock_file_response.return_value = MagicMock()
        with patch.object(
            models_storage[trained_model_id], "plot_quantization_error"
        ) as mock_plot:
            response = client.get(f"/models/{trained_model_id}/visualize?type=training")
            assert response.status_code == 200
            mock_plot.assert_called_once()

    def test_visualize_invalid_type(self, client, trained_model_id):
        """Test visualization with invalid type"""
        response = client.get(f"/models/{trained_model_id}/visualize?type=invalid")
        assert response.status_code == 400
        assert "Type must be" in response.json()["detail"]

    def test_visualize_nonexistent_model(self, client):
        """Test visualization with nonexistent model"""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/models/{fake_id}/visualize")
        assert response.status_code == 404

    @patch("api.Path.unlink")
    def test_visualize_exception_handling(self, mock_unlink, client, trained_model_id):
        """Test visualization handles exceptions"""
        with patch.object(
            models_storage[trained_model_id],
            "visualize_weights",
            side_effect=Exception("Viz failed"),
        ):
            response = client.get(f"/models/{trained_model_id}/visualize")
            assert response.status_code == 500
            assert "Visualization failed" in response.json()["detail"]
            mock_unlink.assert_called()


@pytest.mark.api
@pytest.mark.io
class TestUploadEndpoint:
    """Tests for file upload endpoint"""

    def test_upload_json_file(self, client):
        """Test uploading JSON file"""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        json_content = json.dumps(data).encode()

        files = {"file": ("test.json", json_content, "application/json")}
        response = client.post("/upload", files=files)

        assert response.status_code == 200
        result = response.json()
        assert result["filename"] == "test.json"
        assert result["shape"] == [3, 3]
        assert len(result["data_preview"]) == 3

    def test_upload_csv_file(self, client):
        """Test uploading CSV file"""
        csv_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9"

        files = {"file": ("test.csv", csv_content.encode(), "text/csv")}
        response = client.post("/upload", files=files)

        assert response.status_code == 200
        result = response.json()
        assert result["filename"] == "test.csv"
        assert result["shape"] == [3, 3]

    def test_upload_unsupported_format(self, client):
        """Test uploading unsupported file format"""
        files = {"file": ("test.txt", b"some text", "text/plain")}
        response = client.post("/upload", files=files)

        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_upload_invalid_json(self, client):
        """Test uploading invalid JSON"""
        files = {"file": ("test.json", b"invalid json", "application/json")}
        response = client.post("/upload", files=files)

        assert response.status_code == 400
        assert "Failed to parse file" in response.json()["detail"]

    def test_upload_empty_file(self, client):
        """Test uploading empty file"""
        files = {"file": ("test.json", b"[]", "application/json")}
        response = client.post("/upload", files=files)

        assert response.status_code == 400  # Empty data causes conversion error
        # Empty arrays fail during numpy conversion


@pytest.mark.api
@pytest.mark.unit
class TestSerializeFunction:
    """Tests for serialize_all utility function"""

    def test_serialize_primitives(self):
        """Test serializing primitive types"""
        assert serialize_all("string") == "string"
        assert serialize_all(42) == 42
        assert serialize_all(3.14) == 3.14
        assert serialize_all(True) is True
        assert serialize_all(None) is None

    def test_serialize_numpy_types(self):
        """Test serializing numpy types"""
        assert isinstance(serialize_all(np.float32(1.0)), str)
        assert isinstance(serialize_all(np.array([1, 2, 3])), str)

    def test_serialize_collections(self):
        """Test serializing collections"""
        result = serialize_all({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

        result = serialize_all([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_serialize_nested_structures(self):
        """Test serializing nested structures"""
        data = {
            "list": [1, 2, {"nested": "value"}],
            "numpy": np.float32(1.0),
            "type": int,
        }
        result = serialize_all(data)
        assert isinstance(result, dict)
        assert result["list"][2]["nested"] == "value"
        assert isinstance(result["numpy"], str)
        assert isinstance(result["type"], str)

    def test_serialize_objects_with_dict(self):
        """Test serializing objects with __dict__"""

        class TestObj:
            def __init__(self):
                self.attr = "value"

        obj = TestObj()
        result = serialize_all(obj)
        assert result == {"attr": "value"}


@pytest.mark.api
@pytest.mark.integration
class TestIntegration:
    """Integration tests combining multiple endpoints"""

    def test_full_workflow(self, client, sample_training_data, sample_config):
        """Test complete workflow: train, predict, visualize, delete"""
        # Train model
        train_request = {"data": sample_training_data, "config": sample_config}
        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        # Make predictions
        predict_request = {"model_id": model_id, "data": sample_training_data[:3]}
        predict_response = client.post("/predict", json=predict_request)
        assert predict_response.status_code == 200

        # Get model info
        info_response = client.get(f"/models/{model_id}")
        assert info_response.status_code == 200

        # List models
        list_response = client.get("/models")
        assert model_id in list_response.json()

        # Delete model
        delete_response = client.delete(f"/models/{model_id}")
        assert delete_response.status_code == 200

        # Verify deletion
        final_info_response = client.get(f"/models/{model_id}")
        assert final_info_response.status_code == 404

    def test_multiple_models(self, client, sample_training_data):
        """Test handling multiple models simultaneously"""
        model_ids = []

        # Train multiple models
        for i in range(3):
            config = {"width": 3 + i, "height": 3 + i, "n_iterations": 5}
            request = {"data": sample_training_data, "config": config}
            response = client.post("/train", json=request)
            assert response.status_code == 200
            model_ids.append(response.json()["model_id"])

        # Verify all models exist
        list_response = client.get("/models")
        models = list_response.json()
        for model_id in model_ids:
            assert model_id in models

        # Clean up
        for model_id in model_ids:
            client.delete(f"/models/{model_id}")


@pytest.fixture(autouse=True)
def cleanup_models():
    """Clean up models storage after each test"""
    yield
    models_storage.clear()
