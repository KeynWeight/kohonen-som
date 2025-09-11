"""
Test cases for the CLI module (cli.py)
"""

import json
import tempfile
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

import pytest
import numpy as np
import pandas as pd

from cli import (
    load_data,
    save_model,
    train_command,
    predict_command,
    visualize_command,
    info_command,
    main,
)
from som import SOM, SOMConfig


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n")
        return f.name


@pytest.fixture
def sample_json_file():
    """Create a temporary JSON file for testing"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return f.name


@pytest.fixture
def sample_npy_file():
    """Create a temporary NPY file for testing"""
    data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, data)
        return f.name


@pytest.fixture
def sample_npz_file():
    """Create a temporary NPZ file for testing"""
    data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, data=data)
        return f.name


@pytest.fixture
def trained_som_file():
    """Create a temporary trained SOM file for testing"""
    config = SOMConfig(width=3, height=3, n_features=3, n_iterations=5, seed=42)
    som = SOM(config, verbose=False)
    data = np.random.random((10, 3)).astype(np.float32)
    som.fit(data)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        som.save(f.name)
        return f.name


@pytest.mark.cli
@pytest.mark.io
class TestLoadData:
    """Tests for load_data function"""

    def test_load_csv_file(self, sample_csv_file):
        """Test loading CSV file"""
        data = load_data(sample_csv_file)
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 3)
        assert data.dtype == np.float32

    def test_load_csv_auto_format(self, sample_csv_file):
        """Test loading CSV with auto format detection"""
        data = load_data(sample_csv_file, "auto")
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 3)

    def test_load_json_file(self, sample_json_file):
        """Test loading JSON file"""
        data = load_data(sample_json_file)
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 3)
        assert data.dtype == np.float32

    def test_load_npy_file(self, sample_npy_file):
        """Test loading NPY file"""
        data = load_data(sample_npy_file)
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 3)
        assert data.dtype == np.float32

    def test_load_npz_file(self, sample_npz_file):
        """Test loading NPZ file"""
        data = load_data(sample_npz_file)
        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 3)
        assert data.dtype == np.float32

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file"""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent.csv")

    def test_load_unsupported_format(self, sample_csv_file):
        """Test loading with unsupported format"""
        with pytest.raises(ValueError, match="Unsupported format"):
            load_data(sample_csv_file, "txt")

    def test_load_invalid_csv(self):
        """Test loading invalid CSV file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("invalid,csv,content\nno,numbers,here")
            with pytest.raises(ValueError):
                load_data(f.name)

    def test_load_invalid_json(self):
        """Test loading invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json}")
            with pytest.raises(ValueError):
                load_data(f.name)


@pytest.mark.cli
@pytest.mark.io
class TestSaveModel:
    """Tests for save_model function"""

    def test_save_model_success(self, trained_som_file):
        """Test successful model saving"""
        som = SOM.load(trained_som_file)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            output_path = f.name

        with patch("builtins.print") as mock_print:
            save_model(som, output_path)
            mock_print.assert_called_with(f"Model saved to: {output_path}")

    def test_save_model_failure(self, trained_som_file):
        """Test model saving failure"""
        som = SOM.load(trained_som_file)

        with patch.object(som, "save", side_effect=Exception("Save failed")):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print") as mock_print:
                    save_model(som, "/invalid/path/model.pkl")
                    mock_print.assert_called()
                    mock_exit.assert_called_with(1)


@pytest.mark.cli
@pytest.mark.unit
class TestTrainCommand:
    """Tests for train_command function"""

    @pytest.fixture
    def train_args(self, sample_csv_file):
        """Create mock args for training"""
        args = MagicMock()
        args.input = sample_csv_file
        args.format = "auto"
        args.width = 5
        args.height = 5
        args.iterations = 10
        args.learning_rate = 0.1
        args.sigma = None
        args.init_strategy = "random"
        args.distance_metric = "euclidean"
        args.learning_mode = "online"
        args.seed = 42
        args.verbose = False
        args.output = "test_model.pkl"
        args.visualize = False
        return args

    def test_train_command_success(self, train_args):
        """Test successful training command"""
        with patch("cli.load_data") as mock_load:
            mock_load.return_value = np.random.random((10, 3)).astype(np.float32)

            with patch("cli.save_model") as mock_save:
                with patch("builtins.print") as mock_print:
                    train_command(train_args)
                    mock_save.assert_called_once()

                    # Check that training info was printed
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    assert any("Training completed!" in call for call in print_calls)

    def test_train_command_with_visualization(self, train_args):
        """Test training command with visualization"""
        train_args.visualize = True

        with patch("cli.load_data") as mock_load:
            mock_load.return_value = np.random.random((10, 3)).astype(np.float32)

            with patch("cli.save_model"):
                with patch("cli.SOM") as mock_som_class:
                    mock_som = MagicMock()
                    mock_som.quantization_error.return_value = 0.1
                    mock_som.topographic_error.return_value = 0.05
                    mock_som_class.return_value = mock_som

                    with patch("builtins.print") as mock_print:
                        train_command(train_args)

                        mock_som.visualize_weights.assert_called_once()
                        mock_som.plot_quantization_error.assert_called_once()

    def test_train_command_data_loading_error(self, train_args):
        """Test training command with data loading error"""
        with patch("cli.load_data", side_effect=Exception("Load failed")):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print") as mock_print:
                    train_command(train_args)
                    mock_exit.assert_called_with(1)


@pytest.mark.cli
@pytest.mark.unit
class TestPredictCommand:
    """Tests for predict_command function"""

    @pytest.fixture
    def predict_args(self, trained_som_file, sample_csv_file):
        """Create mock args for prediction"""
        args = MagicMock()
        args.model = trained_som_file
        args.input = sample_csv_file
        args.format = "auto"
        args.output = "predictions.json"
        return args

    def test_predict_command_success(self, predict_args):
        """Test successful prediction command"""
        with patch("cli.load_data") as mock_load:
            mock_load.return_value = np.random.random((5, 3)).astype(np.float32)

            with patch("cli.SOM.load") as mock_som_load:
                mock_som = MagicMock()
                mock_som.predict.return_value = np.array([0, 1, 2, 3, 4])
                mock_som.transform.return_value = np.random.random((5, 2))
                mock_som_load.return_value = mock_som

                with patch("builtins.open", mock_open()) as mock_file:
                    with patch("json.dump") as mock_json_dump:
                        with patch("builtins.print"):
                            predict_command(predict_args)
                            mock_json_dump.assert_called_once()

    def test_predict_command_model_loading_error(self, predict_args):
        """Test prediction command with model loading error"""
        with patch("cli.SOM.load", side_effect=Exception("Model load failed")):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print"):
                    predict_command(predict_args)
                    mock_exit.assert_called_with(1)

    def test_predict_command_data_loading_error(self, predict_args):
        """Test prediction command with data loading error"""
        with patch("cli.SOM.load") as mock_som_load:
            mock_som_load.return_value = MagicMock()

            with patch("cli.load_data", side_effect=Exception("Data load failed")):
                with patch("sys.exit") as mock_exit:
                    with patch("builtins.print"):
                        predict_command(predict_args)
                        mock_exit.assert_called_with(1)


@pytest.mark.cli
@pytest.mark.visualization
class TestVisualizeCommand:
    """Tests for visualize_command function"""

    @pytest.fixture
    def visualize_args(self, trained_som_file):
        """Create mock args for visualization"""
        args = MagicMock()
        args.model = trained_som_file
        args.output = None
        args.type = "weights"
        return args

    def test_visualize_weights(self, visualize_args):
        """Test weights visualization"""
        with patch("cli.SOM.load") as mock_som_load:
            mock_som = MagicMock()
            mock_som_load.return_value = mock_som

            with patch("builtins.print"):
                visualize_command(visualize_args)
                mock_som.visualize_weights.assert_called_once()

    def test_visualize_training(self, visualize_args):
        """Test training visualization"""
        visualize_args.type = "training"

        with patch("cli.SOM.load") as mock_som_load:
            mock_som = MagicMock()
            mock_som_load.return_value = mock_som

            with patch("builtins.print"):
                visualize_command(visualize_args)
                mock_som.plot_quantization_error.assert_called_once()

    def test_visualize_all(self, visualize_args):
        """Test all visualizations"""
        visualize_args.type = "all"

        with patch("cli.SOM.load") as mock_som_load:
            mock_som = MagicMock()
            mock_som_load.return_value = mock_som

            with patch("builtins.print"):
                visualize_command(visualize_args)
                mock_som.visualize_weights.assert_called_once()
                mock_som.plot_quantization_error.assert_called_once()

    def test_visualize_model_loading_error(self, visualize_args):
        """Test visualization with model loading error"""
        with patch("cli.SOM.load", side_effect=Exception("Model load failed")):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print"):
                    visualize_command(visualize_args)
                    mock_exit.assert_called_with(1)


@pytest.mark.cli
@pytest.mark.unit
class TestInfoCommand:
    """Tests for info_command function"""

    @pytest.fixture
    def info_args(self, trained_som_file):
        """Create mock args for info command"""
        args = MagicMock()
        args.model = trained_som_file
        return args

    def test_info_command_success(self, info_args):
        """Test successful info command"""
        mock_info = {
            "shape": (3, 3),
            "n_features": 3,
            "n_neurons": 9,
            "total_epochs": 5,
            "total_samples": 50,
            "config": {
                "width": 3,
                "height": 3,
                "n_features": 3,
                "weight_bounds": (-1, 1),  # This should be skipped
            },
        }

        with patch("cli.SOM.load") as mock_som_load:
            mock_som = MagicMock()
            mock_som.get_info.return_value = mock_info
            mock_som_load.return_value = mock_som

            with patch("builtins.print") as mock_print:
                info_command(info_args)

                # Check that info was printed
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("SOM Model Information" in call for call in print_calls)
                assert any("Shape: 3x3" in call for call in print_calls)

    def test_info_command_model_loading_error(self, info_args):
        """Test info command with model loading error"""
        with patch("cli.SOM.load", side_effect=Exception("Model load failed")):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print"):
                    info_command(info_args)
                    mock_exit.assert_called_with(1)


@pytest.mark.cli
@pytest.mark.unit
class TestMainFunction:
    """Tests for main function and argument parsing"""

    def test_main_no_args(self):
        """Test main function with no arguments"""
        with patch("sys.argv", ["cli.py"]):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print"):
                    main()
                    mock_exit.assert_called_with(1)

    def test_main_help(self):
        """Test main function with help argument"""
        with patch("sys.argv", ["cli.py", "--help"]):
            with patch("sys.exit") as mock_exit:
                try:
                    main()
                except SystemExit:
                    pass  # Expected behavior for help

    def test_main_train_command(self, sample_csv_file):
        """Test main function with train command"""
        args = [
            "cli.py",
            "train",
            sample_csv_file,
            "--width",
            "5",
            "--height",
            "5",
            "--iterations",
            "10",
        ]

        with patch("sys.argv", args):
            with patch("cli.train_command") as mock_train:
                main()
                mock_train.assert_called_once()

    def test_main_predict_command(self, trained_som_file, sample_csv_file):
        """Test main function with predict command"""
        args = ["cli.py", "predict", trained_som_file, sample_csv_file]

        with patch("sys.argv", args):
            with patch("cli.predict_command") as mock_predict:
                main()
                mock_predict.assert_called_once()

    def test_main_visualize_command(self, trained_som_file):
        """Test main function with visualize command"""
        args = ["cli.py", "visualize", trained_som_file]

        with patch("sys.argv", args):
            with patch("cli.visualize_command") as mock_visualize:
                main()
                mock_visualize.assert_called_once()

    def test_main_info_command(self, trained_som_file):
        """Test main function with info command"""
        args = ["cli.py", "info", trained_som_file]

        with patch("sys.argv", args):
            with patch("cli.info_command") as mock_info:
                main()
                mock_info.assert_called_once()

    def test_main_version_command(self):
        """Test main function with version command"""
        with patch("sys.argv", ["cli.py", "version"]):
            with patch("builtins.print") as mock_print:
                main()
                mock_print.assert_called_with("Kohonen SOM CLI v0.1.0")

    def test_main_invalid_command(self):
        """Test main function with invalid command"""
        with patch("sys.argv", ["cli.py", "invalid"]):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print"):
                    main()
                    mock_exit.assert_called_with(1)


@pytest.mark.cli
@pytest.mark.unit
class TestArgumentParsing:
    """Tests for command line argument parsing"""

    def test_train_arguments(self):
        """Test train command arguments"""
        args = [
            "cli.py",
            "train",
            "data.csv",
            "--output",
            "model.pkl",
            "--width",
            "10",
            "--height",
            "8",
            "--iterations",
            "500",
            "--learning-rate",
            "0.05",
            "--sigma",
            "2.0",
            "--init-strategy",
            "pca",
            "--distance-metric",
            "manhattan",
            "--learning-mode",
            "batch",
            "--format",
            "csv",
            "--seed",
            "123",
            "--visualize",
            "--verbose",
        ]

        with patch("sys.argv", args):
            with patch("cli.train_command") as mock_train:
                main()
                mock_train.assert_called_once()

                # Check that args were parsed correctly
                call_args = mock_train.call_args[0][0]
                assert call_args.output == "model.pkl"
                assert call_args.width == 10
                assert call_args.height == 8
                assert call_args.iterations == 500
                assert call_args.learning_rate == 0.05
                assert call_args.sigma == 2.0
                assert call_args.init_strategy == "pca"
                assert call_args.distance_metric == "manhattan"
                assert call_args.learning_mode == "batch"
                assert call_args.format == "csv"
                assert call_args.seed == 123
                assert call_args.visualize is True
                assert call_args.verbose is True

    def test_predict_arguments(self):
        """Test predict command arguments"""
        args = [
            "cli.py",
            "predict",
            "model.pkl",
            "data.csv",
            "--output",
            "results.json",
            "--format",
            "json",
        ]

        with patch("sys.argv", args):
            with patch("cli.predict_command") as mock_predict:
                main()
                mock_predict.assert_called_once()

                call_args = mock_predict.call_args[0][0]
                assert call_args.model == "model.pkl"
                assert call_args.input == "data.csv"
                assert call_args.output == "results.json"
                assert call_args.format == "json"

    def test_visualize_arguments(self):
        """Test visualize command arguments"""
        args = [
            "cli.py",
            "visualize",
            "model.pkl",
            "--output",
            "viz.png",
            "--type",
            "training",
        ]

        with patch("sys.argv", args):
            with patch("cli.visualize_command") as mock_visualize:
                main()
                mock_visualize.assert_called_once()

                call_args = mock_visualize.call_args[0][0]
                assert call_args.model == "model.pkl"
                assert call_args.output == "viz.png"
                assert call_args.type == "training"


@pytest.mark.cli
@pytest.mark.integration
class TestIntegration:
    """Integration tests for CLI functionality"""

    def test_full_workflow_simulation(self, sample_csv_file):
        """Test simulated full workflow"""
        # Mock the complete workflow without actual file I/O
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")

            # Simulate training
            train_args = [
                "cli.py",
                "train",
                sample_csv_file,
                "--output",
                model_path,
                "--width",
                "3",
                "--height",
                "3",
                "--iterations",
                "5",
            ]

            with patch("sys.argv", train_args):
                with patch("cli.load_data") as mock_load:
                    mock_load.return_value = np.random.random((10, 3)).astype(
                        np.float32
                    )
                    with patch("cli.save_model"):
                        with patch("builtins.print"):
                            main()

    def test_error_handling_chain(self):
        """Test error handling across different command failures"""
        # Test file not found error
        with patch("sys.argv", ["cli.py", "train", "nonexistent.csv"]):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print"):
                    main()
                    mock_exit.assert_called_with(1)

        # Test invalid model file
        with patch("sys.argv", ["cli.py", "info", "nonexistent.pkl"]):
            with patch("sys.exit") as mock_exit:
                with patch("builtins.print"):
                    main()
                    mock_exit.assert_called_with(1)


# Cleanup function for temporary files
def cleanup_temp_files():
    """Clean up any temporary files created during tests"""
    import glob

    temp_files = (
        glob.glob("test_*.pkl")
        + glob.glob("test_*.png")
        + glob.glob("predictions.json")
    )
    for file in temp_files:
        try:
            os.unlink(file)
        except FileNotFoundError:
            pass


@pytest.fixture(autouse=True)
def cleanup_files():
    """Auto cleanup after each test"""
    yield
    cleanup_temp_files()
