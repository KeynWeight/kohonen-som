# Kohonen Self-Organizing Map (SOM)

A Self-Organizing Map implementation with CLI and API interfaces featuring comprehensive visualization capabilities.

## 🚀 Live Demo

**Try the live API at: https://kohonen-som.onrender.com**

- API Documentation: https://kohonen-som.onrender.com/docs
- Health Check: https://kohonen-som.onrender.com/health

## Installation

```bash
uv sync
```

## Quick Start Examples

### CLI Workflow

```bash
# 1. Train model with visualization
uv run python cli.py train data/sample_data.csv --width 15 --height 15 --iterations 1000 --visualize --output my_model.pkl

# 2. Generate specific visualizations
uv run python cli.py visualize my_model.pkl --type weights --output weights.png
uv run python cli.py visualize my_model.pkl --type training --output training.png

# 3. Make predictions
uv run python cli.py predict my_model.pkl data/new_data.csv --output predictions.json

# 4. View model info
uv run python cli.py info my_model.pkl

# 5. Run examples (generates multiple plots)
uv run python examples.py
```

### API Workflow (Local Development)

```bash
# 1. Start server
uv run python api.py
# Server available at http://localhost:8000/docs

# 2. Train model and get model_id
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d "{\"data\": [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],[0.2,0.3,0.4],[0.5,0.6,0.7]], \"config\": {\"width\": 10, \"height\": 10, \"n_iterations\": 100}}"

# Response contains model_id like: "8ca65ab2-5764-4a61-94b9-248eb25d5b12"

# 3. List all models
curl "http://localhost:8000/models"

# 4. Get model information (replace with your model_id)
curl "http://localhost:8000/models/8ca65ab2-5764-4a61-94b9-248eb25d5b12"

# 5. Generate and download visualizations
# Note: Plots are automatically saved to plots/ directory on server
curl "http://localhost:8000/models/8ca65ab2-5764-4a61-94b9-248eb25d5b12/visualize?type=weights" --output weights_download.png
curl "http://localhost:8000/models/8ca65ab2-5764-4a61-94b9-248eb25d5b12/visualize?type=training" --output training_download.png

# 6. Make predictions
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"model_id\": \"8ca65ab2-5764-4a61-94b9-248eb25d5b12\", \"data\": [[0.1,0.2,0.3],[0.8,0.9,1.0]]}"

# 7. Upload data file (CSV or JSON)
curl -X POST "http://localhost:8000/upload" -F "file=@data/sample_data.csv"

# 8. Health check
curl "http://localhost:8000/health"

# 9. Delete model when done
curl -X DELETE "http://localhost:8000/models/8ca65ab2-5764-4a61-94b9-248eb25d5b12"
```

### API Workflow (Live Deployment)

```bash
# Use the live API at https://kohonen-som.onrender.com

# 1. Train model and get model_id
curl -X POST "https://kohonen-som.onrender.com/train" -H "Content-Type: application/json" -d "{\"data\": [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],[0.2,0.3,0.4],[0.5,0.6,0.7]], \"config\": {\"width\": 10, \"height\": 10, \"n_iterations\": 100}}"

# 2. List all models
curl "https://kohonen-som.onrender.com/models"

# 3. Get model information (replace with your model_id)
curl "https://kohonen-som.onrender.com/models/8ca65ab2-5764-4a61-94b9-248eb25d5b12"

# 4. Generate and download visualizations
curl "https://kohonen-som.onrender.com/models/8ca65ab2-5764-4a61-94b9-248eb25d5b12/visualize?type=weights" --output weights_download.png

# 5. Make predictions
curl -X POST "https://kohonen-som.onrender.com/predict" -H "Content-Type: application/json" -d "{\"model_id\": \"8ca65ab2-5764-4a61-94b9-248eb25d5b12\", \"data\": [[0.1,0.2,0.3],[0.8,0.9,1.0]]}"

# 6. Health check
curl "https://kohonen-som.onrender.com/health"
```

### Complete Windows Workflow Example

```bash
# Start fresh session
uv run python api.py
# Keep this terminal open, open new terminal for commands below

# Step 1: Train a model
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d "{\"data\": [[1.0,2.0],[3.0,4.0],[5.0,6.0],[2.0,3.0],[4.0,5.0]], \"config\": {\"width\": 5, \"height\": 5, \"n_iterations\": 50}}"

# Copy the model_id from response, example: "abc123-def456-789"

# Step 2: Check model info
curl "http://localhost:8000/models/abc123-def456-789"

# Step 3: Generate visualizations (plots saved to plots/ folder automatically)
curl "http://localhost:8000/models/abc123-def456-789/visualize?type=weights"
curl "http://localhost:8000/models/abc123-def456-789/visualize?type=training"

# Step 4: Make predictions
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"model_id\": \"abc123-def456-789\", \"data\": [[1.5,2.5],[4.5,5.5]]}"

# Step 5: List all files in plots directory
dir plots
```

## API Endpoints Reference

### Training & Models
- `POST /train` - Train new SOM model
- `GET /models` - List all trained models  
- `GET /models/{model_id}` - Get model information
- `DELETE /models/{model_id}` - Delete specific model

### Predictions
- `POST /predict` - Make predictions with trained model

### Visualizations
- `GET /models/{model_id}/visualize?type=weights` - Generate/download weight visualization
- `GET /models/{model_id}/visualize?type=training` - Generate/download training progress plot

### Utilities
- `POST /upload` - Upload and validate data files (CSV/JSON)
- `GET /health` - API health check
- `GET /` - API information

## CLI Commands Reference

### Training
```bash
uv run python cli.py train <data_file> [options]
  --width INT          SOM width (default: 20)
  --height INT         SOM height (default: 20) 
  --iterations INT     Training iterations (default: 1000)
  --learning-rate FLOAT Initial learning rate (default: 0.1)
  --visualize         Generate visualizations during training
  --output PATH       Save trained model to file
```

### Visualization
```bash
uv run python cli.py visualize <model_file> [options]
  --type TYPE         Visualization type: weights, training, all
  --output PATH       Output file path
```

### Prediction
```bash
uv run python cli.py predict <model_file> <data_file> [options]
  --output PATH       Save predictions to JSON file
```

### Model Info
```bash
uv run python cli.py info <model_file>
```

## Configuration Parameters

| Parameter | CLI Flag | API Field | Default | Description |
|-----------|----------|-----------|---------|-------------|
| **Grid Dimensions** |
| Width | `--width` | `width` | 20 | SOM grid width |
| Height | `--height` | `height` | 20 | SOM grid height |
| **Training** |
| Iterations | `--iterations` | `n_iterations` | 1000 | Number of training epochs |
| Learning Rate | `--learning-rate` | `initial_alpha` | 0.1 | Initial learning rate |
| Min Learning Rate | - | `min_alpha` | 0.001 | Minimum learning rate |
| Batch Size | - | `batch_size` | 32 | Training batch size |
| **Topology** |
| Initial Sigma | - | `initial_sigma` | auto | Initial neighborhood radius |
| Min Sigma | - | `min_sigma` | 0.01 | Minimum neighborhood radius |
| Distance Metric | - | `distance_metric` | euclidean | Distance calculation method |
| Init Strategy | - | `init_strategy` | random | Weight initialization method |
| **Advanced** |
| Early Stopping | - | `early_stopping` | true | Stop when converged |
| Convergence Tolerance | - | `convergence_tolerance` | 1e-4 | Convergence threshold |
| Patience | - | `patience` | 10 | Early stopping patience |
| Random Seed | - | `seed` | null | Reproducibility seed |

## Data Formats

### Supported Input Formats
- **CSV files**: Comma-separated numeric data
- **JSON arrays**: Nested arrays of numbers
- **Direct API**: JSON in request body

### CSV Example
```csv
0.1,0.2,0.3
0.4,0.5,0.6
0.7,0.8,0.9
1.0,1.1,1.2
```

### JSON Example
```json
[
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9],
  [1.0, 1.1, 1.2]
]
```

### API Request Example
```json
{
  "data": [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
  ],
  "config": {
    "width": 10,
    "height": 10,
    "n_iterations": 100,
    "initial_alpha": 0.1
  }
}
```

## Output Files

### Generated Visualizations
- **Weight Visualization**: `plots/{model_id}_weights.png` or custom name
- **Training Progress**: `plots/{model_id}_training.png` or custom name
- **All plots saved to**: `plots/` directory (auto-created)

### Model Files
- **Trained Models**: `.pkl` files in `models/` directory
- **Predictions**: JSON files with coordinates and cluster assignments

### Response Examples

#### Training Response
```json
{
  "model_id": "8ca65ab2-5764-4a61-94b9-248eb25d5b12",
  "quantization_error": 0.024,
  "topographic_error": 0.4,
  "message": "SOM trained successfully with 5 samples"
}
```

#### Prediction Response
```json
{
  "predictions": [0, 1, 2],
  "coordinates": [[2, 3], [1, 4], [0, 1]],
  "model_id": "8ca65ab2-5764-4a61-94b9-248eb25d5b12"
}
```
