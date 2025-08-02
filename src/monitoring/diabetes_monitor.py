"""
Diabetes Model Monitoring with Evidently

Comprehensive monitoring for diabetes prediction model including
drift detection, bias monitoring, and automated alerting system.
"""

import json
import logging
import os
import smtplib
import warnings
from datetime import datetime, timedelta
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from typing import Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_suite import MetricSuite
from evidently.metrics import (
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
    ClassificationQualityByClass,
    ClassificationQualityMetric,
    ConflictPredictionMetric,
    ConflictTargetMetric,
    DatasetCorrelationsMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnsType,
    TestMeanInNSigmas,
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDriftedColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfRowsWithMissingValues,
    TestShareOfMissingValues,
    TestValueRange,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DiabetesModelMonitor:
    """Comprehensive diabetes model monitoring system."""

    def __init__(
        self,
        reference_data_path: str = "data/processed/diabetes_train_processed.csv",
        alert_email: str = None,
        alert_threshold: float = 0.3,
    ):
        """
        Initialize diabetes model monitor.

        Args:
            reference_data_path: Path to reference dataset
            alert_email: Email for alerts
            alert_threshold: Threshold for triggering alerts
        """
        self.reference_data_path = reference_data_path
        self.alert_email = alert_email
        self.alert_threshold = alert_threshold
        self.column_mapping = self._setup_column_mapping()
        self.monitoring_reports_dir = "models/monitoring"
        os.makedirs(self.monitoring_reports_dir, exist_ok=True)

        # Load reference data
        self.reference_data = self._load_reference_data()
        logger.info("Diabetes model monitor initialized")

    def _setup_column_mapping(self) -> ColumnMapping:
        """Set up column mapping for Evidently."""
        return ColumnMapping(
            prediction="diabetes_prediction",
            target="target",
            numerical_features=[
                "pregnancies",
                "glucose",
                "blood_pressure",
                "skin_thickness",
                "insulin",
                "bmi",
                "diabetes_pedigree_function",
                "age",
            ],
            categorical_features=[],
            datetime_features=[],
        )

    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference dataset for comparison."""
        if not os.path.exists(self.reference_data_path):
            logger.error(f"Reference data not found: {self.reference_data_path}")
            raise FileNotFoundError("Reference dataset required for monitoring")

        reference_df = pd.read_csv(self.reference_data_path)
        logger.info(f"Reference data loaded: {reference_df.shape}")
        return reference_df

    def generate_data_drift_report(self, current_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data drift report.

        Args:
            current_data: Current production data

        Returns:
            Dict: Drift analysis results
        """
        logger.info("Generating data drift report...")

        # Create data drift report
        data_drift_report = Report(
            metrics=[
                DatasetDriftMetric(),
                DatasetMissingValuesMetric(),
                DatasetCorrelationsMetric(),
            ]
        )

        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.monitoring_reports_dir, f"data_drift_report_{timestamp}.html"
        )
        data_drift_report.save_html(report_path)

        # Extract metrics
        report_dict = data_drift_report.as_dict()
        drift_metrics = self._extract_drift_metrics(report_dict)

        logger.info(f"Data drift report saved: {report_path}")
        return drift_metrics

    def generate_model_performance_report(
        self, current_data: pd.DataFrame, predictions: np.ndarray
    ) -> Dict:
        """
        Generate model performance monitoring report.

        Args:
            current_data: Current production data with targets
            predictions: Model predictions

        Returns:
            Dict: Performance analysis results
        """
        logger.info("Generating model performance report...")

        # Add predictions to current data
        current_data_with_preds = current_data.copy()
        current_data_with_preds["diabetes_prediction"] = predictions

        # Create performance report
        performance_report = Report(
            metrics=[
                ClassificationQualityMetric(),
                ClassificationClassBalance(),
                ClassificationConfusionMatrix(),
                ClassificationQualityByClass(),
                ConflictTargetMetric(),
                ConflictPredictionMetric(),
            ]
        )

        performance_report.run(
            reference_data=self.reference_data,
            current_data=current_data_with_preds,
            column_mapping=self.column_mapping,
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.monitoring_reports_dir, f"performance_report_{timestamp}.html"
        )
        performance_report.save_html(report_path)

        # Extract performance metrics
        report_dict = performance_report.as_dict()
        performance_metrics = self._extract_performance_metrics(report_dict)

        logger.info(f"Performance report saved: {report_path}")
        return performance_metrics

    def run_data_quality_tests(self, current_data: pd.DataFrame) -> Dict:
        """
        Run comprehensive data quality test suite.

        Args:
            current_data: Current production data

        Returns:
            Dict: Test results
        """
        logger.info("Running data quality tests...")

        # Create test suite
        data_quality_tests = TestSuite(
            tests=[
                TestNumberOfColumnsWithMissingValues(),
                TestNumberOfRowsWithMissingValues(),
                TestNumberOfConstantColumns(),
                TestNumberOfDuplicatedRows(),
                TestColumnsType(),
                TestShareOfMissingValues(),
                # Medical range validation tests
                TestValueRange(column_name="glucose", left=50, right=300),
                TestValueRange(column_name="blood_pressure", left=40, right=200),
                TestValueRange(column_name="bmi", left=10, right=80),
                TestValueRange(column_name="age", left=18, right=120),
                # Statistical tests
                TestMeanInNSigmas(column_name="glucose"),
                TestMeanInNSigmas(column_name="bmi"),
                TestMeanInNSigmas(column_name="age"),
                # Drift tests
                TestNumberOfDriftedColumns(stattest="psi", stattest_threshold=0.2),
            ]
        )

        data_quality_tests.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = os.path.join(
            self.monitoring_reports_dir, f"data_quality_tests_{timestamp}.html"
        )
        data_quality_tests.save_html(test_path)

        # Extract test results
        test_results = self._extract_test_results(data_quality_tests)

        logger.info(f"Data quality tests saved: {test_path}")
        return test_results

    def detect_bias_and_fairness(
        self,
        current_data: pd.DataFrame,
        predictions: np.ndarray,
        sensitive_features: List[str] = ["age"],
    ) -> Dict:
        """
        Detect bias and fairness issues in predictions.

        Args:
            current_data: Current data with demographics
            predictions: Model predictions
            sensitive_features: Features to check for bias

        Returns:
            Dict: Bias analysis results
        """
        logger.info("Analyzing bias and fairness...")

        bias_results = {}

        for feature in sensitive_features:
            if feature in current_data.columns:
                # Create age groups for bias analysis
                if feature == "age":
                    current_data["age_group"] = pd.cut(
                        current_data["age"],
                        bins=[0, 30, 50, 100],
                        labels=["young", "middle", "older"],
                    )

                    # Calculate prediction rates by age group
                    group_stats = (
                        current_data.groupby("age_group")
                        .agg({"target": ["count", "mean"]})
                        .round(3)
                    )

                    prediction_rates = (
                        pd.Series(predictions).groupby(current_data["age_group"]).mean()
                    )

                    bias_results[feature] = {
                        "group_statistics": group_stats.to_dict(),
                        "prediction_rates": prediction_rates.to_dict(),
                        "demographic_parity_difference": self._calculate_demographic_parity(
                            current_data["age_group"], predictions
                        ),
                    }

        logger.info("Bias analysis completed")
        return bias_results

    def _calculate_demographic_parity(
        self, groups: pd.Series, predictions: np.ndarray
    ) -> float:
        """Calculate demographic parity difference."""
        group_rates = pd.Series(predictions).groupby(groups).mean()
        return float(group_rates.max() - group_rates.min())

    def _extract_drift_metrics(self, report_dict: Dict) -> Dict:
        """Extract key drift metrics from Evidently report."""
        try:
            metrics = report_dict["metrics"][0]["result"]
            return {
                "dataset_drift": metrics.get("dataset_drift", False),
                "drift_share": metrics.get("drift_share", 0.0),
                "number_of_drifted_columns": metrics.get(
                    "number_of_drifted_columns", 0
                ),
                "drift_by_columns": metrics.get("drift_by_columns", {}),
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting drift metrics: {e}")
            return {"error": str(e)}

    def _extract_performance_metrics(self, report_dict: Dict) -> Dict:
        """Extract key performance metrics from Evidently report."""
        try:
            quality_metric = report_dict["metrics"][0]["result"]["current"]
            return {
                "accuracy": quality_metric.get("accuracy", 0.0),
                "precision": quality_metric.get("precision", 0.0),
                "recall": quality_metric.get("recall", 0.0),
                "f1": quality_metric.get("f1", 0.0),
                "roc_auc": quality_metric.get("roc_auc", 0.0),
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting performance metrics: {e}")
            return {"error": str(e)}

    def _extract_test_results(self, test_suite: TestSuite) -> Dict:
        """Extract test results from test suite."""
        results = test_suite.as_dict()

        test_summary = {
            "total_tests": len(results["tests"]),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
        }

        for test in results["tests"]:
            test_status = test["status"]
            test_summary["test_details"].append(
                {
                    "test_name": test["name"],
                    "status": test_status,
                    "description": test.get("description", ""),
                }
            )

            if test_status == "SUCCESS":
                test_summary["passed_tests"] += 1
            else:
                test_summary["failed_tests"] += 1

        return test_summary

    def comprehensive_monitoring_check(
        self, current_data: pd.DataFrame, predictions: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run comprehensive monitoring check including all analyses.

        Args:
            current_data: Current production data
            predictions: Model predictions (optional)

        Returns:
            Dict: Complete monitoring results
        """
        logger.info("Starting comprehensive monitoring check...")

        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": current_data.shape,
            "monitoring_status": "healthy",
        }

        # 1. Data drift analysis
        try:
            drift_results = self.generate_data_drift_report(current_data)
            monitoring_results["data_drift"] = drift_results

            # Check drift threshold
            if drift_results.get("drift_share", 0) > self.alert_threshold:
                monitoring_results["monitoring_status"] = "alert"
                monitoring_results["alerts"] = monitoring_results.get("alerts", [])
                monitoring_results["alerts"].append("Data drift detected")

        except Exception as e:
            logger.error(f"Data drift analysis failed: {e}")
            monitoring_results["data_drift"] = {"error": str(e)}

        # 2. Data quality tests
        try:
            quality_results = self.run_data_quality_tests(current_data)
            monitoring_results["data_quality"] = quality_results

            # Check quality test failures
            if quality_results.get("failed_tests", 0) > 0:
                monitoring_results["monitoring_status"] = "warning"
                monitoring_results["alerts"] = monitoring_results.get("alerts", [])
                monitoring_results["alerts"].append(
                    f"Data quality tests failed: {quality_results['failed_tests']}"
                )

        except Exception as e:
            logger.error(f"Data quality tests failed: {e}")
            monitoring_results["data_quality"] = {"error": str(e)}

        # 3. Model performance analysis (if predictions provided)
        if predictions is not None:
            try:
                performance_results = self.generate_model_performance_report(
                    current_data, predictions
                )
                monitoring_results["model_performance"] = performance_results

                # Check performance degradation
                if (
                    performance_results.get("accuracy", 1.0) < 0.75
                ):  # Singapore healthcare threshold
                    monitoring_results["monitoring_status"] = "critical"
                    monitoring_results["alerts"] = monitoring_results.get("alerts", [])
                    monitoring_results["alerts"].append(
                        "Model performance below threshold"
                    )

            except Exception as e:
                logger.error(f"Performance analysis failed: {e}")
                monitoring_results["model_performance"] = {"error": str(e)}

            # 4. Bias and fairness analysis
            try:
                bias_results = self.detect_bias_and_fairness(current_data, predictions)
                monitoring_results["bias_analysis"] = bias_results

                # Check for significant bias
                for feature, analysis in bias_results.items():
                    if analysis.get("demographic_parity_difference", 0) > 0.1:
                        monitoring_results["monitoring_status"] = "warning"
                        monitoring_results["alerts"] = monitoring_results.get(
                            "alerts", []
                        )
                        monitoring_results["alerts"].append(
                            f"Bias detected in {feature}"
                        )

            except Exception as e:
                logger.error(f"Bias analysis failed: {e}")
                monitoring_results["bias_analysis"] = {"error": str(e)}

        # 5. Save monitoring summary
        self._save_monitoring_summary(monitoring_results)

        # 6. Send alerts if necessary
        if monitoring_results["monitoring_status"] in ["alert", "critical"]:
            self._send_alerts(monitoring_results)

        logger.info(
            f"Comprehensive monitoring completed - Status: {monitoring_results['monitoring_status']}"
        )
        return monitoring_results

    def _save_monitoring_summary(self, results: Dict):
        """Save monitoring summary to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(
            self.monitoring_reports_dir, f"monitoring_summary_{timestamp}.json"
        )

        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Monitoring summary saved: {summary_path}")

    def _send_alerts(self, monitoring_results: Dict):
        """Send alert notifications."""
        if not self.alert_email:
            logger.warning("No alert email configured - skipping alert notification")
            return

        try:
            alerts = monitoring_results.get("alerts", [])
            status = monitoring_results["monitoring_status"]

            subject = f"🚨 Diabetes Model Alert - Status: {status.upper()}"
            body = f"""
Diabetes Model Monitoring Alert

Status: {status.upper()}
Timestamp: {monitoring_results['timestamp']}

Alerts:
{chr(10).join(f"- {alert}" for alert in alerts)}

Data Shape: {monitoring_results['data_shape']}

Please check the monitoring dashboard for detailed analysis.
"""

            logger.info(f"Alert would be sent to {self.alert_email}: {subject}")
            logger.debug(f"Alert body: {body}")
            # Note: Actual email sending would require SMTP configuration

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")


def load_model_and_predict(data: pd.DataFrame) -> np.ndarray:
    """Load model and make predictions for monitoring."""
    try:
        # Try to load from MLflow registry
        mlflow.set_tracking_uri("sqlite:///models/artifacts/mlflow.db")
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(
            "diabetes_prediction_champion", stages=["Production", "Staging", "None"]
        )

        if model_versions:
            latest_version = model_versions[0]
            model_uri = f"models:/{latest_version.name}/{latest_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
        else:
            # Fallback to local model
            model_path = "models/trained/diabetes_optimized_random_forest.pkl"
            model = joblib.load(model_path)

        # Load preprocessor
        from data.preprocess import DiabetesPreprocessor

        preprocessor = DiabetesPreprocessor.load(
            "models/artifacts/diabetes_preprocessor.pkl"
        )

        # Preprocess and predict
        X_processed = preprocessor.transform(
            data.drop("target", axis=1, errors="ignore")
        )
        predictions = model.predict(X_processed)

        return predictions

    except Exception as e:
        logger.error(f"Failed to load model and predict: {e}")
        return np.array([])


def main():
    """Main monitoring execution."""
    logger.info("Starting diabetes model monitoring...")

    # Initialize monitor
    monitor = DiabetesModelMonitor(
        alert_email="data-science-team@singapore-health.gov.sg", alert_threshold=0.25
    )

    # Load test data for monitoring
    test_data_path = "data/processed/diabetes_test_processed.csv"
    if not os.path.exists(test_data_path):
        logger.error("Test data not found for monitoring")
        return

    current_data = pd.read_csv(test_data_path)
    logger.info(f"Loaded current data for monitoring: {current_data.shape}")

    # Generate predictions for performance monitoring
    predictions = load_model_and_predict(current_data)

    # Run comprehensive monitoring
    if len(predictions) > 0:
        results = monitor.comprehensive_monitoring_check(current_data, predictions)
    else:
        results = monitor.comprehensive_monitoring_check(current_data)

    # Print summary
    print("\n" + "=" * 60)
    print("DIABETES MODEL MONITORING SUMMARY")
    print("=" * 60)
    print(f"Status: {results['monitoring_status'].upper()}")
    print(f"Data Shape: {results['data_shape']}")

    if "alerts" in results:
        print(f"Alerts: {len(results['alerts'])}")
        for alert in results["alerts"]:
            print(f"  ⚠️  {alert}")

    if "data_drift" in results and "drift_share" in results["data_drift"]:
        print(f"Drift Share: {results['data_drift']['drift_share']:.3f}")

    if "data_quality" in results:
        quality = results["data_quality"]
        print(
            f"Quality Tests: {quality.get('passed_tests', 0)}/{quality.get('total_tests', 0)} passed"
        )

    print(f"Reports saved in: {monitor.monitoring_reports_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
