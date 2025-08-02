"""
Integration Tests for Diabetes Prediction Pipeline

End-to-end tests for the complete MLOps pipeline including
data loading, preprocessing, training, and prediction.
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Add src to Python path (must be before local imports)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import pytest

from data.load_diabetes_data import main as load_data_main
from data.preprocess import DiabetesPreprocessor, preprocess_diabetes_data
from models.train_diabetes_model import DiabetesModelTrainer


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()

        # Create directory structure
        os.makedirs(os.path.join(temp_dir, "data", "raw"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "models", "trained"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "models", "artifacts"), exist_ok=True)

        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        yield temp_dir

        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    def test_data_loading_pipeline(self, temp_workspace):
        """Test data loading pipeline integration."""
        # Test data loading
        file_paths = load_data_main()

        # Verify files were created
        assert os.path.exists(file_paths["raw"])
        assert os.path.exists(file_paths["train"])
        assert os.path.exists(file_paths["test"])

        # Verify data quality
        raw_df = pd.read_csv(file_paths["raw"])
        train_df = pd.read_csv(file_paths["train"])
        test_df = pd.read_csv(file_paths["test"])

        # Check shapes
        assert len(raw_df) == len(train_df) + len(test_df)
        assert len(train_df) > len(test_df)  # 80/20 split

        # Check columns
        expected_columns = [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree_function",
            "age",
            "target",
        ]
        assert all(col in raw_df.columns for col in expected_columns)
        assert all(col in train_df.columns for col in expected_columns)
        assert all(col in test_df.columns for col in expected_columns)

    def test_preprocessing_pipeline(self, temp_workspace):
        """Test preprocessing pipeline integration."""
        # First load data
        file_paths = load_data_main()

        # Test preprocessing
        train_processed, test_processed, preprocessor = preprocess_diabetes_data(
            file_paths["train"], file_paths["test"]
        )

        # Verify preprocessing results
        assert isinstance(train_processed, pd.DataFrame)
        assert isinstance(test_processed, pd.DataFrame)
        assert isinstance(preprocessor, DiabetesPreprocessor)
        assert preprocessor.is_fitted

        # Check that features were engineered
        assert train_processed.shape[1] > 9  # More than original features
        assert test_processed.shape[1] == train_processed.shape[1]

        # Check no missing values in processed data
        assert not train_processed.drop("target", axis=1).isna().any().any()
        assert not test_processed.drop("target", axis=1).isna().any().any()

        # Verify preprocessor was saved
        preprocessor_path = "models/artifacts/diabetes_preprocessor.pkl"
        assert os.path.exists(preprocessor_path)

    @patch("models.train_diabetes_model.mlflow.set_experiment")
    @patch("models.train_diabetes_model.mlflow.set_tracking_uri")
    @patch("models.train_diabetes_model.mlflow.start_run")
    @patch("models.train_diabetes_model.mlflow.log_params")
    @patch("models.train_diabetes_model.mlflow.log_metric")
    @patch("models.train_diabetes_model.mlflow.sklearn.log_model")
    @patch("models.train_diabetes_model.mlflow.tracking.MlflowClient")
    def test_training_pipeline(
        self,
        mock_client_class,
        mock_log_model,
        mock_log_metric,
        mock_log_params,
        mock_start_run,
        mock_set_uri,
        mock_set_experiment,
        temp_workspace,
    ):
        """Test model training pipeline integration."""
        # Setup mocks
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None

        mock_client = MagicMock()
        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_client_class.return_value = mock_client

        # Load and preprocess data
        file_paths = load_data_main()
        train_processed, test_processed, preprocessor = preprocess_diabetes_data(
            file_paths["train"], file_paths["test"]
        )

        # Test training
        trainer = DiabetesModelTrainer(experiment_name="integration_test")

        # Train baseline models
        X_train = train_processed.drop("target", axis=1)
        y_train = train_processed["target"]
        X_test = test_processed.drop("target", axis=1)
        y_test = test_processed["target"]

        baseline_results, best_model = trainer.train_baseline_models(
            X_train, X_test, y_train, y_test
        )

        # Verify training results
        assert isinstance(baseline_results, dict)
        assert len(baseline_results) > 0
        assert best_model in baseline_results

        # Check that models were saved
        for model_name in baseline_results:
            model_path = f"models/trained/diabetes_{model_name}.pkl"
            assert os.path.exists(model_path)

        # Verify MLflow integration
        mock_set_experiment.assert_called()
        mock_log_params.assert_called()
        mock_log_metric.assert_called()
        mock_log_model.assert_called()

    def test_prediction_pipeline(self, temp_workspace):
        """Test prediction pipeline integration."""
        # Load and preprocess data
        file_paths = load_data_main()
        train_processed, test_processed, preprocessor = preprocess_diabetes_data(
            file_paths["train"], file_paths["test"]
        )

        # Train a simple model for prediction testing
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        X_train = train_processed.drop("target", axis=1)
        y_train = train_processed["target"]
        X_test = test_processed.drop("target", axis=1)
        # y_test unused in this test - would be used for evaluation

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        model_path = "models/trained/test_model.pkl"
        joblib.dump(model, model_path)

        # Test prediction on new data
        sample_data = X_test.head(5)

        # Preprocess and predict
        X_processed = preprocessor.transform(sample_data)
        predictions = model.predict(X_processed)
        prediction_probas = model.predict_proba(X_processed)

        # Verify predictions
        assert len(predictions) == 5
        assert len(prediction_probas) == 5
        assert all(pred in [0, 1] for pred in predictions)
        assert all(
            0 <= prob <= 1 for prob_row in prediction_probas for prob in prob_row
        )


class TestPipelineDataFlow:
    """Test data flow through the pipeline."""

    def test_data_consistency_through_pipeline(self):
        """Test that data maintains consistency through pipeline stages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create directory structure
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("models/artifacts", exist_ok=True)

            # Generate initial data
            file_paths = load_data_main()
            original_data = pd.read_csv(file_paths["raw"])

            # Process data
            train_processed, test_processed, _ = preprocess_diabetes_data(
                file_paths["train"], file_paths["test"]
            )

            # Verify data consistency (original files would be used for validation)
            # train_original = pd.read_csv(file_paths["train"])
            # test_original = pd.read_csv(file_paths["test"])

            # Check that target distribution is preserved
            original_target_dist = original_data["target"].value_counts(normalize=True)
            processed_target_dist = pd.concat(
                [train_processed["target"], test_processed["target"]]
            ).value_counts(normalize=True)

            # Allow for small differences due to processing
            for class_label in [0, 1]:
                assert (
                    abs(
                        original_target_dist[class_label]
                        - processed_target_dist[class_label]
                    )
                    < 0.05
                )

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results."""
        results1 = []
        results2 = []

        for run in range(2):
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)

                # Create directory structure
                os.makedirs("data/raw", exist_ok=True)
                os.makedirs("data/processed", exist_ok=True)
                os.makedirs("models/artifacts", exist_ok=True)

                # Run pipeline
                file_paths = load_data_main()
                train_processed, test_processed, _ = preprocess_diabetes_data(
                    file_paths["train"], file_paths["test"]
                )

                # Store results
                result = {
                    "train_shape": train_processed.shape,
                    "test_shape": test_processed.shape,
                    "train_target_mean": train_processed["target"].mean(),
                    "test_target_mean": test_processed["target"].mean(),
                }

                if run == 0:
                    results1.append(result)
                else:
                    results2.append(result)

        # Compare results
        assert results1[0]["train_shape"] == results2[0]["train_shape"]
        assert results1[0]["test_shape"] == results2[0]["test_shape"]
        assert (
            abs(results1[0]["train_target_mean"] - results2[0]["train_target_mean"])
            < 0.01
        )
        assert (
            abs(results1[0]["test_target_mean"] - results2[0]["test_target_mean"])
            < 0.01
        )


class TestPipelineErrorHandling:
    """Test pipeline error handling and recovery."""

    def test_missing_data_handling(self):
        """Test pipeline handles missing input data gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create directory structure
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("models/artifacts", exist_ok=True)

            # Test preprocessing with missing files
            with pytest.raises(FileNotFoundError):
                preprocess_diabetes_data(
                    "nonexistent_train.csv", "nonexistent_test.csv"
                )

    def test_corrupted_data_handling(self):
        """Test pipeline handles corrupted data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create directory structure
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("models/artifacts", exist_ok=True)

            # Create corrupted data files
            corrupted_data = pd.DataFrame(
                {"invalid_column": [1, 2, 3], "another_invalid": ["a", "b", "c"]}
            )

            train_path = "data/processed/corrupted_train.csv"
            test_path = "data/processed/corrupted_test.csv"

            corrupted_data.to_csv(train_path, index=False)
            corrupted_data.to_csv(test_path, index=False)

            # Test that preprocessing handles missing columns gracefully
            try:
                preprocess_diabetes_data(train_path, test_path)
            except Exception as e:
                # Should raise a meaningful error
                assert "target" in str(e).lower() or "column" in str(e).lower()


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_pipeline_execution_time(self):
        """Test that pipeline completes within reasonable time."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create directory structure
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("models/artifacts", exist_ok=True)

            start_time = time.time()

            # Run data loading and preprocessing
            file_paths = load_data_main()
            train_processed, test_processed, _ = preprocess_diabetes_data(
                file_paths["train"], file_paths["test"]
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete data pipeline within reasonable time
            assert execution_time < 30.0  # 30 seconds max

            # Verify results quality wasn't sacrificed for speed
            assert len(train_processed) > 0
            assert len(test_processed) > 0
            assert not train_processed.isna().all().any()

    def test_memory_usage(self):
        """Test pipeline memory usage is reasonable."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create directory structure
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("models/artifacts", exist_ok=True)

            # Run pipeline
            file_paths = load_data_main()
            train_processed, test_processed, _ = preprocess_diabetes_data(
                file_paths["train"], file_paths["test"]
            )

            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory

            # Memory increase should be reasonable for the dataset size
            assert memory_increase < 500  # Less than 500 MB increase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
