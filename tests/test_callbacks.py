"""
Tests for callback functionality
"""

import pytest
import tempfile
import os
import shutil
from som import SOM, SOMConfig
from som.callbacks import CheckpointCallback, EarlyStoppingCallback


@pytest.mark.integration
class TestCheckpointCallback:
    """Test checkpoint callback functionality"""

    @pytest.mark.unit
    def test_checkpoint_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(tmpdir, interval=2)
            assert callback.checkpoint_dir == tmpdir
            assert callback.interval == 2
            assert os.path.exists(tmpdir)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_checkpoint_saving(self, minimal_config, small_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(tmpdir, interval=2)

            som = SOM(minimal_config, verbose=False)
            som.fit(small_data, callbacks=[callback])

            # Check that checkpoints were created
            checkpoint_files = [
                f for f in os.listdir(tmpdir) if f.startswith("checkpoint_epoch_")
            ]
            assert len(checkpoint_files) > 0

            # Check final model was saved
            assert os.path.exists(os.path.join(tmpdir, "final_model.pkl"))


@pytest.mark.integration
class TestEarlyStoppingCallback:
    """Test early stopping callback functionality"""

    @pytest.mark.unit
    def test_early_stopping_creation(self):
        callback = EarlyStoppingCallback(monitor="qe", patience=5, min_delta=1e-3)
        assert callback.monitor == "qe"
        assert callback.patience == 5
        assert callback.min_delta == 1e-3
        assert callback.best_value == float("inf")
        assert callback.wait == 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_early_stopping_trigger(self, small_data):
        # Create config with many iterations but early stopping
        config = SOMConfig(width=3, height=3, n_features=2, n_iterations=100, seed=42)
        callback = EarlyStoppingCallback(monitor="qe", patience=3, min_delta=1e-6)

        som = SOM(config, verbose=False)
        som.fit(small_data, callbacks=[callback])

        # Should stop early due to convergence
        assert som.metadata["total_epochs"] < 100

    @pytest.mark.unit
    def test_early_stopping_reset_on_training_begin(self):
        callback = EarlyStoppingCallback(patience=5)
        callback.best_value = 0.5
        callback.wait = 3

        # Simulate training begin
        callback.on_training_begin(None)

        assert callback.best_value == float("inf")
        assert callback.wait == 0


@pytest.mark.integration
class TestCallbackIntegration:
    """Test integration of multiple callbacks"""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_multiple_callbacks(self, minimal_config, small_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_cb = CheckpointCallback(tmpdir, interval=2)
            early_stop_cb = EarlyStoppingCallback(patience=10)

            som = SOM(minimal_config, verbose=False)
            som.fit(small_data, callbacks=[checkpoint_cb, early_stop_cb])

            # Both callbacks should have been executed
            assert os.path.exists(os.path.join(tmpdir, "final_model.pkl"))
            assert (
                som.metadata["total_epochs"] <= 10
            )  # Either completed or stopped early
