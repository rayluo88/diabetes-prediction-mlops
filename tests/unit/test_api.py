"""
Unit Tests for Diabetes Prediction API

Test suite for FastAPI endpoints, validation, and error handling.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to Python path (must be before local imports)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.diabetes_api import app
from api.schemas import DiabetesInput, DiabetesPrediction


class TestDiabetesAPI:
    """Test diabetes prediction API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient data."""
        return {
            "patient_id": "TEST_001",
            "pregnancies": 2.0,
            "glucose": 120.0,
            "blood_pressure": 80.0,
            "skin_thickness": 25.0,
            "insulin": 100.0,
            "bmi": 28.5,
            "diabetes_pedigree_function": 0.5,
            "age": 35.0,
        }

    @pytest.fixture
    def sample_batch_data(self):
        """Create sample batch data."""
        return {
            "batch_id": "BATCH_TEST_001",
            "patients": [
                {
                    "patient_id": "TEST_001",
                    "pregnancies": 2.0,
                    "glucose": 120.0,
                    "blood_pressure": 80.0,
                    "skin_thickness": 25.0,
                    "insulin": 100.0,
                    "bmi": 28.5,
                    "diabetes_pedigree_function": 0.5,
                    "age": 35.0,
                },
                {
                    "patient_id": "TEST_002",
                    "pregnancies": 0.0,
                    "glucose": 100.0,
                    "blood_pressure": 70.0,
                    "skin_thickness": 20.0,
                    "insulin": 80.0,
                    "bmi": 22.0,
                    "diabetes_pedigree_function": 0.2,
                    "age": 25.0,
                },
            ],
        }

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    @patch("api.diabetes_api.model", new=MagicMock())
    @patch("api.diabetes_api.preprocessor", new=MagicMock())
    @patch("api.diabetes_api.model_version", new="test_v1")
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_version" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data

    @patch("api.diabetes_api.model")
    @patch("api.diabetes_api.preprocessor")
    @patch("api.diabetes_api.model_version", "test_v1")
    def test_predict_endpoint_success(
        self, mock_preprocessor, mock_model, client, sample_patient_data
    ):
        """Test successful prediction endpoint."""
        # Mock model prediction
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

        # Mock preprocessor
        mock_processed_data = MagicMock()
        mock_preprocessor.transform.return_value = mock_processed_data

        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 200

        data = response.json()
        assert "patient_id" in data
        assert "diabetes_prediction" in data
        assert "diabetes_probability" in data
        assert "confidence_score" in data
        assert "risk_category" in data
        assert "recommendation" in data
        assert "model_version" in data
        assert "prediction_timestamp" in data

        # Check data types and ranges
        assert isinstance(data["diabetes_prediction"], bool)
        assert 0 <= data["diabetes_probability"] <= 1
        assert 0 <= data["confidence_score"] <= 1
        assert data["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
        assert data["patient_id"] == sample_patient_data["patient_id"]

    def test_predict_endpoint_validation_error(self, client):
        """Test prediction endpoint with validation errors."""
        invalid_data = {
            "patient_id": "TEST_001",
            "pregnancies": -1.0,  # Invalid: negative
            "glucose": 400.0,  # Invalid: too high
            "blood_pressure": 30.0,  # Invalid: too low
            "skin_thickness": 25.0,
            "insulin": 100.0,
            "bmi": 5.0,  # Invalid: too low
            "diabetes_pedigree_function": 0.5,
            "age": 15.0,  # Invalid: too young
        }

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_data = {
            "patient_id": "TEST_001",
            "pregnancies": 2.0,
            "glucose": 120.0,
            # Missing other required fields
        }

        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    @patch("api.diabetes_api.model")
    @patch("api.diabetes_api.preprocessor")
    @patch("api.diabetes_api.model_version", "test_v1")
    def test_batch_predict_endpoint_success(
        self, mock_preprocessor, mock_model, client, sample_batch_data
    ):
        """Test successful batch prediction endpoint."""
        # Mock model predictions
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])

        # Mock preprocessor
        mock_processed_data = MagicMock()
        mock_preprocessor.transform.return_value = mock_processed_data

        response = client.post("/predict/batch", json=sample_batch_data)
        assert response.status_code == 200

        data = response.json()
        assert "batch_id" in data
        assert "predictions" in data
        assert "summary" in data
        assert "processing_timestamp" in data

        # Check batch structure
        assert data["batch_id"] == sample_batch_data["batch_id"]
        assert len(data["predictions"]) == len(sample_batch_data["patients"])

        # Check summary
        summary = data["summary"]
        assert "total_patients" in summary
        assert "predicted_diabetic" in summary
        assert "high_risk_count" in summary
        assert "average_diabetes_probability" in summary

        # Check individual predictions
        for prediction in data["predictions"]:
            assert "patient_id" in prediction
            assert "diabetes_prediction" in prediction
            assert "diabetes_probability" in prediction
            assert "risk_category" in prediction

    def test_batch_predict_endpoint_validation_error(self, client):
        """Test batch prediction with validation errors."""
        invalid_batch_data = {
            "batch_id": "BATCH_TEST_001",
            "patients": [
                {
                    "patient_id": "TEST_001",
                    "pregnancies": -1.0,  # Invalid
                    "glucose": 120.0,
                    "blood_pressure": 80.0,
                    "skin_thickness": 25.0,
                    "insulin": 100.0,
                    "bmi": 28.5,
                    "diabetes_pedigree_function": 0.5,
                    "age": 35.0,
                }
            ],
        }

        response = client.post("/predict/batch", json=invalid_batch_data)
        assert response.status_code == 422

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        with (
            patch("api.diabetes_api.model_version", "test_v1"),
            patch("api.diabetes_api.model", MagicMock()),
            patch("api.diabetes_api.preprocessor", MagicMock()),
        ):

            response = client.get("/model/info")
            assert response.status_code == 200

            data = response.json()
            assert "model_version" in data
            assert "model_type" in data
            assert "features" in data
            assert "target" in data
            assert "model_loaded" in data
            assert "preprocessor_loaded" in data

            # Check features list
            expected_features = [
                "pregnancies",
                "glucose",
                "blood_pressure",
                "skin_thickness",
                "insulin",
                "bmi",
                "diabetes_pedigree_function",
                "age",
            ]
            assert data["features"] == expected_features

    @patch("api.diabetes_api.model", None)
    def test_prediction_without_model_loaded(self, client, sample_patient_data):
        """Test prediction when model is not loaded."""
        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 503


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_diabetes_input_valid(self):
        """Test valid diabetes input schema."""
        valid_data = {
            "patient_id": "TEST_001",
            "pregnancies": 2.0,
            "glucose": 120.0,
            "blood_pressure": 80.0,
            "skin_thickness": 25.0,
            "insulin": 100.0,
            "bmi": 28.5,
            "diabetes_pedigree_function": 0.5,
            "age": 35.0,
        }

        # Should not raise any validation errors
        diabetes_input = DiabetesInput(**valid_data)
        assert diabetes_input.patient_id == "TEST_001"
        assert diabetes_input.glucose == 120.0

    def test_diabetes_input_validation_errors(self):
        """Test diabetes input validation errors."""
        # Test negative pregnancies
        with pytest.raises(ValueError):
            DiabetesInput(
                patient_id="TEST_001",
                pregnancies=-1.0,
                glucose=120.0,
                blood_pressure=80.0,
                skin_thickness=25.0,
                insulin=100.0,
                bmi=28.5,
                diabetes_pedigree_function=0.5,
                age=35.0,
            )

        # Test extreme BMI
        with pytest.raises(ValueError):
            DiabetesInput(
                patient_id="TEST_001",
                pregnancies=2.0,
                glucose=120.0,
                blood_pressure=80.0,
                skin_thickness=25.0,
                insulin=100.0,
                bmi=5.0,  # Too low
                diabetes_pedigree_function=0.5,
                age=35.0,
            )

        # Test invalid age
        with pytest.raises(ValueError):
            DiabetesInput(
                patient_id="TEST_001",
                pregnancies=2.0,
                glucose=120.0,
                blood_pressure=80.0,
                skin_thickness=25.0,
                insulin=100.0,
                bmi=28.5,
                diabetes_pedigree_function=0.5,
                age=15.0,  # Too young
            )

    def test_diabetes_prediction_schema(self):
        """Test diabetes prediction output schema."""
        from datetime import datetime

        prediction_data = {
            "patient_id": "TEST_001",
            "diabetes_prediction": False,
            "diabetes_probability": 0.25,
            "confidence_score": 0.85,
            "risk_category": "Low Risk",
            "recommendation": "Maintain healthy lifestyle.",
            "model_version": "1.0",
            "prediction_timestamp": datetime.now(),
        }

        # Should not raise any validation errors
        prediction = DiabetesPrediction(**prediction_data)
        assert prediction.patient_id == "TEST_001"
        assert prediction.diabetes_prediction is False
        assert 0 <= prediction.diabetes_probability <= 1
        assert 0 <= prediction.confidence_score <= 1


class TestAPIErrorHandling:
    """Test API error handling scenarios."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @patch("api.diabetes_api.model")
    @patch("api.diabetes_api.preprocessor")
    def test_prediction_internal_error(
        self, mock_preprocessor, mock_model, client, sample_patient_data
    ):
        """Test handling of internal server errors during prediction."""
        # Mock model to raise an exception
        mock_model.predict.side_effect = Exception("Model error")
        mock_preprocessor.transform.return_value = MagicMock()

        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 500

    @patch("api.diabetes_api.model")
    @patch("api.diabetes_api.preprocessor")
    def test_preprocessing_error(
        self, mock_preprocessor, mock_model, client, sample_patient_data
    ):
        """Test handling of preprocessing errors."""
        # Mock preprocessor to raise an exception
        mock_preprocessor.transform.side_effect = Exception("Preprocessing error")

        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 500

    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestAPIPerformance:
    """Test API performance characteristics."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @patch("api.diabetes_api.model")
    @patch("api.diabetes_api.preprocessor")
    def test_prediction_response_time(
        self, mock_preprocessor, mock_model, client, sample_patient_data
    ):
        """Test that predictions complete within reasonable time."""
        import time

        # Mock fast prediction
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_preprocessor.transform.return_value = MagicMock()

        start_time = time.time()
        response = client.post("/predict", json=sample_patient_data)
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds

    @patch("api.diabetes_api.model")
    @patch("api.diabetes_api.preprocessor")
    def test_batch_prediction_scalability(self, mock_preprocessor, mock_model, client):
        """Test batch prediction with larger datasets."""
        # Create larger batch
        large_batch = {"batch_id": "LARGE_BATCH_001", "patients": []}

        # Add 50 patients
        for i in range(50):
            patient = {
                "patient_id": f"TEST_{i:03d}",
                "pregnancies": 2.0,
                "glucose": 120.0,
                "blood_pressure": 80.0,
                "skin_thickness": 25.0,
                "insulin": 100.0,
                "bmi": 28.5,
                "diabetes_pedigree_function": 0.5,
                "age": 35.0,
            }
            large_batch["patients"].append(patient)

        # Mock predictions
        mock_model.predict.return_value = np.zeros(50)
        mock_model.predict_proba.return_value = np.random.rand(50, 2)
        mock_preprocessor.transform.return_value = MagicMock()

        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
