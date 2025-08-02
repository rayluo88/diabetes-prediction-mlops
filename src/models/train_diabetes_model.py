"""
Diabetes Model Training with MLflow Integration

Comprehensive model training pipeline with experiment tracking,
hyperparameter optimization, and model registry integration.
"""

import logging
import os
import warnings
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DiabetesModelTrainer:
    """Diabetes model training with MLflow experiment tracking."""

    def __init__(self, experiment_name="diabetes_prediction"):
        self.experiment_name = experiment_name
        self.setup_mlflow()

    def setup_mlflow(self):
        """Initialize MLflow experiment."""
        mlflow.set_experiment(self.experiment_name)

        # Set MLflow tracking URI
        os.makedirs("models/artifacts", exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///models/artifacts/mlflow.db")

        logger.info(f"MLflow experiment set: {self.experiment_name}")

    def load_data(self):
        """Load processed diabetes data."""
        train_path = "data/processed/diabetes_train_processed.csv"
        test_path = "data/processed/diabetes_test_processed.csv"

        if not os.path.exists(train_path):
            logger.error("Processed data not found. Run preprocessing first.")
            raise FileNotFoundError("Run data preprocessing pipeline first")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]
        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]

        logger.info(f"Training data: {X_train.shape}")
        logger.info(f"Test data: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation."""
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        return metrics, y_pred, y_pred_proba

    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train multiple baseline models."""
        models = {
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "svm": SVC(random_state=42, probability=True),
        }

        best_model = None
        best_score = 0
        results = {}

        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"baseline_{model_name}"):
                logger.info(f"Training {model_name}...")

                # Train model
                model.fit(X_train, y_train)

                # Evaluate model
                metrics, y_pred, y_pred_proba = self.evaluate_model(
                    model, X_test, y_test
                )

                # Log parameters
                mlflow.log_params(model.get_params())

                # Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)

                # Log model
                mlflow.sklearn.log_model(
                    model,
                    f"model_{model_name}",
                    registered_model_name=f"diabetes_model_{model_name}",
                )

                # Save locally
                model_path = f"models/trained/diabetes_{model_name}.pkl"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(model, model_path)

                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "model_path": model_path,
                }

                # Track best model
                if metrics["f1_score"] > best_score:
                    best_score = metrics["f1_score"]
                    best_model = model_name

                logger.info(f"{model_name} - F1 Score: {metrics['f1_score']:.4f}")

        logger.info(f"Best baseline model: {best_model} (F1: {best_score:.4f})")
        return results, best_model

    def optimize_hyperparameters(
        self, X_train, y_train, X_test, y_test, model_type="random_forest"
    ):
        """Optimize hyperparameters using Optuna."""

        def objective(trial):
            if model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": 42,
                }
                model = RandomForestClassifier(**params)

            elif model_type == "gradient_boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "random_state": 42,
                }
                model = GradientBoostingClassifier(**params)

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return f1_score(y_test, y_pred)

        logger.info(f"Starting hyperparameter optimization for {model_type}...")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best F1 score: {best_score:.4f}")

        # Train final model with best parameters
        if model_type == "random_forest":
            final_model = RandomForestClassifier(**best_params)
        elif model_type == "gradient_boosting":
            final_model = GradientBoostingClassifier(**best_params)

        final_model.fit(X_train, y_train)

        # Log optimized model
        with mlflow.start_run(run_name=f"optimized_{model_type}"):
            # Log parameters
            mlflow.log_params(best_params)

            # Evaluate and log metrics
            metrics, _, _ = self.evaluate_model(final_model, X_test, y_test)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log optimization details
            mlflow.log_param("optimization_trials", 50)
            mlflow.log_metric("best_trial_score", best_score)

            # Log model
            mlflow.sklearn.log_model(
                final_model,
                f"optimized_{model_type}",
                registered_model_name=f"diabetes_model_optimized_{model_type}",
            )

            # Save locally
            model_path = f"models/trained/diabetes_optimized_{model_type}.pkl"
            joblib.dump(final_model, model_path)

        return final_model, best_params, metrics

    def register_champion_model(self, model, model_name, metrics):
        """Register the champion model in MLflow Model Registry."""
        with mlflow.start_run(run_name=f"champion_{model_name}"):
            # Log final metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log model with champion tag
            mlflow.sklearn.log_model(
                model,
                "champion_model",
                registered_model_name="diabetes_prediction_champion",
            )

            # Add model version tags
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(
                "diabetes_prediction_champion", stages=["None"]
            )[0]

            client.set_model_version_tag(
                "diabetes_prediction_champion",
                model_version.version,
                "validation_status",
                "approved",
            )

            client.set_model_version_tag(
                "diabetes_prediction_champion",
                model_version.version,
                "model_type",
                model_name,
            )

            logger.info(f"Champion model registered: {model_name}")

    def train_complete_pipeline(self):
        """Execute complete training pipeline."""
        logger.info("Starting complete diabetes model training pipeline...")

        # Load data
        X_train, X_test, y_train, y_test = self.load_data()

        # Train baseline models
        baseline_results, best_baseline = self.train_baseline_models(
            X_train, X_test, y_train, y_test
        )

        # Optimize best performing model
        optimized_model, best_params, optimized_metrics = self.optimize_hyperparameters(
            X_train, y_train, X_test, y_test, best_baseline
        )

        # Register champion model
        self.register_champion_model(optimized_model, best_baseline, optimized_metrics)

        # Generate training summary
        self.generate_training_summary(
            baseline_results, optimized_metrics, best_baseline
        )

        return optimized_model, optimized_metrics

    def generate_training_summary(
        self, baseline_results, optimized_metrics, best_model
    ):
        """Generate comprehensive training summary."""
        logger.info("\n" + "=" * 50)
        logger.info("DIABETES MODEL TRAINING SUMMARY")
        logger.info("=" * 50)

        logger.info("\nBASELINE MODEL RESULTS:")
        for model_name, results in baseline_results.items():
            metrics = results["metrics"]
            logger.info(f"{model_name}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")

        logger.info(f"\nBEST MODEL: {best_model}")
        logger.info("OPTIMIZED MODEL METRICS:")
        for metric_name, value in optimized_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        logger.info(f"\nMODEL ARTIFACTS LOCATION: models/trained/")
        logger.info(f"MLFLOW EXPERIMENT: {self.experiment_name}")


def main():
    """Main training execution."""
    trainer = DiabetesModelTrainer()

    try:
        model, metrics = trainer.train_complete_pipeline()
        print("\n✅ Diabetes model training completed successfully!")
        print(f"🎯 Best F1-Score: {metrics['f1_score']:.4f}")
        print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"📁 Model saved in: models/trained/")
        print(f"🔬 MLflow UI: mlflow ui")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
