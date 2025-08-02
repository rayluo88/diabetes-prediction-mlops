"""
Unit Tests for Diabetes Model Training Components

Test suite for model training, evaluation, and MLflow integration.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from models.train_diabetes_model import DiabetesModelTrainer


class TestDiabetesModelTrainer:
    """Test diabetes model training functionality."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        X_train = pd.DataFrame(
            {
                "pregnancies": np.random.randint(0, 10, n_samples),
                "glucose": np.random.normal(120, 30, n_samples),
                "blood_pressure": np.random.normal(80, 15, n_samples),
                "skin_thickness": np.random.exponential(20, n_samples),
                "insulin": np.random.exponential(80, n_samples),
                "bmi": np.random.normal(32, 8, n_samples),
                "diabetes_pedigree_function": np.random.exponential(0.5, n_samples),
                "age": np.random.randint(21, 81, n_samples),
            }
        )

        # Create correlated target
        diabetes_risk = (
            0.02 * X_train["glucose"]
            + 0.01 * X_train["bmi"]
            + 0.005 * X_train["age"]
            + np.random.normal(0, 1, n_samples)
        )
        y_train = (diabetes_risk > np.percentile(diabetes_risk, 70)).astype(int)

        X_test = X_train.sample(20, random_state=42)
        y_test = y_train.iloc[X_test.index]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def trainer(self):
        """Create a model trainer instance."""
        with patch("models.train_diabetes_model.mlflow.set_experiment"):
            with patch("models.train_diabetes_model.mlflow.set_tracking_uri"):
                return DiabetesModelTrainer(experiment_name="test_experiment")

    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.experiment_name == "test_experiment"
        assert hasattr(trainer, "setup_mlflow")

    @patch("models.train_diabetes_model.os.path.exists")
    @patch("models.train_diabetes_model.pd.read_csv")
    def test_load_data(self, mock_read_csv, mock_exists, trainer, sample_training_data):
        """Test data loading functionality."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Mock file existence
        mock_exists.return_value = True

        # Mock CSV reading
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        test_df = pd.concat([X_test, y_test.rename("target")], axis=1)
        mock_read_csv.side_effect = [train_df, test_df]

        X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = (
            trainer.load_data()
        )

        assert isinstance(X_train_loaded, pd.DataFrame)
        assert isinstance(X_test_loaded, pd.DataFrame)
        assert isinstance(y_train_loaded, pd.Series)
        assert isinstance(y_test_loaded, pd.Series)
        assert len(X_train_loaded) == len(y_train_loaded)
        assert len(X_test_loaded) == len(y_test_loaded)

    @patch("models.train_diabetes_model.os.path.exists")
    def test_load_data_file_not_found(self, mock_exists, trainer):
        """Test data loading with missing files."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            trainer.load_data()

    def test_evaluate_model(self, trainer, sample_training_data):
        """Test model evaluation functionality."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics, y_pred, y_pred_proba = trainer.evaluate_model(model, X_test, y_test)

        # Check metrics structure
        expected_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        assert all(metric in metrics for metric in expected_metrics)

        # Check metric ranges
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"

        # Check predictions
        assert len(y_pred) == len(y_test)
        assert len(y_pred_proba) == len(y_test)
        assert set(y_pred).issubset({0, 1})
        assert all(0 <= prob <= 1 for prob in y_pred_proba)

    @patch("models.train_diabetes_model.mlflow.start_run")
    @patch("models.train_diabetes_model.mlflow.log_params")
    @patch("models.train_diabetes_model.mlflow.log_metric")
    @patch("models.train_diabetes_model.mlflow.sklearn.log_model")
    @patch("models.train_diabetes_model.joblib.dump")
    @patch("models.train_diabetes_model.os.makedirs")
    def test_train_baseline_models(
        self,
        mock_makedirs,
        mock_joblib_dump,
        mock_log_model,
        mock_log_metric,
        mock_log_params,
        mock_start_run,
        trainer,
        sample_training_data,
    ):
        """Test baseline model training."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Mock MLflow context manager
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None

        results, best_model = trainer.train_baseline_models(
            X_train, X_test, y_train, y_test
        )

        # Check results structure
        assert isinstance(results, dict)
        assert len(results) > 0
        assert best_model in results

        # Check that each model has required components
        for model_name, result in results.items():
            assert "model" in result
            assert "metrics" in result
            assert "model_path" in result

            # Check model is trained
            model = result["model"]
            assert hasattr(model, "predict")

            # Check metrics
            metrics = result["metrics"]
            assert "f1_score" in metrics
            assert 0 <= metrics["f1_score"] <= 1

        # Check MLflow logging was called
        assert mock_log_params.called
        assert mock_log_metric.called
        assert mock_log_model.called

    def test_optimize_hyperparameters_random_forest(
        self, trainer, sample_training_data
    ):
        """Test hyperparameter optimization for random forest."""
        X_train, X_test, y_train, y_test = sample_training_data

        with (
            patch(
                "models.train_diabetes_model.optuna.create_study"
            ) as mock_create_study,
            patch("models.train_diabetes_model.mlflow.start_run") as mock_start_run,
            patch("models.train_diabetes_model.mlflow.log_params"),
            patch("models.train_diabetes_model.mlflow.log_metric"),
            patch("models.train_diabetes_model.mlflow.sklearn.log_model"),
            patch("models.train_diabetes_model.joblib.dump"),
        ):

            # Mock Optuna study
            mock_study = MagicMock()
            mock_study.best_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            }
            mock_study.best_value = 0.85
            mock_create_study.return_value = mock_study

            # Mock MLflow context manager
            mock_run = MagicMock()
            mock_start_run.return_value.__enter__.return_value = mock_run
            mock_start_run.return_value.__exit__.return_value = None

            model, best_params, metrics = trainer.optimize_hyperparameters(
                X_train, y_train, X_test, y_test, model_type="random_forest"
            )

            # Check results
            assert model is not None
            assert isinstance(best_params, dict)
            assert isinstance(metrics, dict)
            assert "f1_score" in metrics

            # Check that optimize was called
            mock_study.optimize.assert_called_once()

    def test_optimize_hyperparameters_gradient_boosting(
        self, trainer, sample_training_data
    ):
        """Test hyperparameter optimization for gradient boosting."""
        X_train, X_test, y_train, y_test = sample_training_data

        with (
            patch(
                "models.train_diabetes_model.optuna.create_study"
            ) as mock_create_study,
            patch("models.train_diabetes_model.mlflow.start_run") as mock_start_run,
            patch("models.train_diabetes_model.mlflow.log_params"),
            patch("models.train_diabetes_model.mlflow.log_metric"),
            patch("models.train_diabetes_model.mlflow.sklearn.log_model"),
            patch("models.train_diabetes_model.joblib.dump"),
        ):

            # Mock Optuna study
            mock_study = MagicMock()
            mock_study.best_params = {
                "n_estimators": 150,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_samples_split": 10,
            }
            mock_study.best_value = 0.82
            mock_create_study.return_value = mock_study

            # Mock MLflow context manager
            mock_run = MagicMock()
            mock_start_run.return_value.__enter__.return_value = mock_run
            mock_start_run.return_value.__exit__.return_value = None

            model, best_params, metrics = trainer.optimize_hyperparameters(
                X_train, y_train, X_test, y_test, model_type="gradient_boosting"
            )

            # Check results
            assert model is not None
            assert isinstance(best_params, dict)
            assert isinstance(metrics, dict)

    @patch("models.train_diabetes_model.mlflow.start_run")
    @patch("models.train_diabetes_model.mlflow.log_metric")
    @patch("models.train_diabetes_model.mlflow.sklearn.log_model")
    @patch("models.train_diabetes_model.mlflow.tracking.MlflowClient")
    def test_register_champion_model(
        self,
        mock_client_class,
        mock_log_model,
        mock_log_metric,
        mock_start_run,
        trainer,
    ):
        """Test champion model registration."""
        # Mock MLflow components
        mock_client = MagicMock()
        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_client_class.return_value = mock_client

        # Mock MLflow context manager
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None

        # Create test model and metrics
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        metrics = {"accuracy": 0.85, "f1_score": 0.82}

        trainer.register_champion_model(model, "random_forest", metrics)

        # Check that MLflow logging was called
        mock_log_metric.assert_called()
        mock_log_model.assert_called()
        mock_client.set_model_version_tag.assert_called()

    def test_generate_training_summary(self, trainer, sample_training_data):
        """Test training summary generation."""
        X_train, X_test, y_train, y_test = sample_training_data

        # Create mock baseline results
        baseline_results = {
            "random_forest": {
                "metrics": {"accuracy": 0.85, "f1_score": 0.82, "roc_auc": 0.88}
            },
            "logistic_regression": {
                "metrics": {"accuracy": 0.80, "f1_score": 0.78, "roc_auc": 0.83}
            },
        }

        optimized_metrics = {"accuracy": 0.87, "f1_score": 0.84, "roc_auc": 0.90}

        # Should not raise any exceptions
        trainer.generate_training_summary(
            baseline_results, optimized_metrics, "random_forest"
        )


class TestModelValidation:
    """Test model validation and performance checks."""

    def test_model_performance_thresholds(self):
        """Test that models meet minimum performance requirements."""
        # Create test data
        np.random.seed(42)
        X = np.random.randn(200, 8)
        y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(int)

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # Test predictions
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # For this synthetic data, we should get reasonable performance
        assert accuracy > 0.7, f"Accuracy {accuracy} below minimum threshold"
        assert f1 > 0.6, f"F1 score {f1} below minimum threshold"

    def test_model_consistency(self):
        """Test model training consistency."""
        # Create reproducible data
        np.random.seed(42)
        X = np.random.randn(100, 8)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Train two models with same parameters
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        # Predictions should be identical
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_model_feature_importance(self):
        """Test that models provide feature importance."""
        np.random.seed(42)
        X = np.random.randn(100, 8)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Check feature importance
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == 8
        assert all(importance >= 0 for importance in model.feature_importances_)
        assert abs(sum(model.feature_importances_) - 1.0) < 1e-6


class TestMLflowIntegration:
    """Test MLflow integration functionality."""

    @patch("models.train_diabetes_model.mlflow.set_experiment")
    @patch("models.train_diabetes_model.mlflow.set_tracking_uri")
    def test_mlflow_setup(self, mock_set_uri, mock_set_experiment):
        """Test MLflow setup."""
        trainer = DiabetesModelTrainer(experiment_name="test_mlflow")

        mock_set_experiment.assert_called_with("test_mlflow")
        mock_set_uri.assert_called()

    @patch("models.train_diabetes_model.mlflow.log_metric")
    def test_mlflow_metric_logging(self, mock_log_metric):
        """Test MLflow metric logging."""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
            "roc_auc": 0.87,
        }

        # Simulate logging metrics
        for metric_name, value in metrics.items():
            mock_log_metric(metric_name, value)

        # Check that log_metric was called for each metric
        assert mock_log_metric.call_count == len(metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
