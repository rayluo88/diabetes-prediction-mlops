"""
Pydantic schemas for Diabetes Prediction API

Data models for request/response validation and API documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskCategory(str, Enum):
    """Risk category enumeration."""

    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


class DiabetesInput(BaseModel):
    """Input schema for diabetes prediction."""

    patient_id: str = Field(
        ..., description="Unique patient identifier", examples=["PAT_001"]
    )

    pregnancies: float = Field(
        ..., ge=0, le=20, description="Number of pregnancies", examples=[1.0]
    )

    glucose: float = Field(
        ...,
        ge=0,
        le=300,
        description="Plasma glucose concentration (mg/dL)",
        examples=[120.0],
    )

    blood_pressure: float = Field(
        ...,
        ge=0,
        le=200,
        description="Diastolic blood pressure (mm Hg)",
        examples=[80.0],
    )

    skin_thickness: float = Field(
        ...,
        ge=0,
        le=100,
        description="Triceps skin fold thickness (mm)",
        examples=[20.0],
    )

    insulin: float = Field(
        ...,
        ge=0,
        le=1000,
        description="2-Hour serum insulin (mu U/ml)",
        examples=[85.0],
    )

    bmi: float = Field(
        ..., ge=10, le=80, description="Body mass index (kg/m²)", examples=[25.5]
    )

    diabetes_pedigree_function: float = Field(
        ..., ge=0, le=5, description="Diabetes pedigree function", examples=[0.5]
    )

    age: float = Field(..., ge=18, le=120, description="Age in years", examples=[35.0])

    @field_validator("glucose")
    @classmethod
    def validate_glucose(cls, v: float) -> float:
        """Validate glucose levels."""
        if v < 50:
            raise ValueError("Glucose level too low for valid measurement")
        return v

    @field_validator("blood_pressure")
    @classmethod
    def validate_blood_pressure(cls, v: float) -> float:
        """Validate blood pressure."""
        if v < 40:
            raise ValueError("Blood pressure too low for valid measurement")
        return v

    @field_validator("bmi")
    @classmethod
    def validate_bmi(cls, v: float) -> float:
        """Validate BMI."""
        if v < 10 or v > 80:
            raise ValueError("BMI outside normal human range")
        return v

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "patient_id": "PAT_001",
                "pregnancies": 1.0,
                "glucose": 120.0,
                "blood_pressure": 80.0,
                "skin_thickness": 20.0,
                "insulin": 85.0,
                "bmi": 25.5,
                "diabetes_pedigree_function": 0.5,
                "age": 35.0,
            }
        },
    )


class DiabetesPrediction(BaseModel):
    """Output schema for diabetes prediction."""

    patient_id: str = Field(..., description="Patient identifier")

    diabetes_prediction: bool = Field(
        ..., description="Binary diabetes prediction (True = diabetic)"
    )

    diabetes_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of diabetes (0-1)"
    )

    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in prediction (0-1)"
    )

    risk_category: RiskCategory = Field(
        ..., description="Risk category based on probability"
    )

    recommendation: str = Field(
        ..., description="Clinical recommendation based on risk level"
    )

    model_version: str = Field(..., description="Version of the model used")

    prediction_timestamp: datetime = Field(
        ..., description="When the prediction was made"
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "patient_id": "PAT_001",
                "diabetes_prediction": False,
                "diabetes_probability": 0.25,
                "confidence_score": 0.85,
                "risk_category": "Low Risk",
                "recommendation": "Maintain healthy lifestyle. Regular checkups recommended.",
                "model_version": "1.0",
                "prediction_timestamp": "2024-01-15T10:30:00",
            }
        },
    )


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""

    batch_id: str = Field(
        ..., description="Unique batch identifier", examples=["BATCH_2024_001"]
    )

    patients: List[DiabetesInput] = Field(
        ...,
        description="List of patient data for prediction",
        min_length=1,
        max_length=1000,  # Limit batch size
    )

    metadata: Optional[Dict] = Field(
        default={}, description="Additional metadata for the batch"
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "batch_id": "BATCH_2024_001",
                "patients": [
                    {
                        "patient_id": "PAT_001",
                        "pregnancies": 1.0,
                        "glucose": 120.0,
                        "blood_pressure": 80.0,
                        "skin_thickness": 20.0,
                        "insulin": 85.0,
                        "bmi": 25.5,
                        "diabetes_pedigree_function": 0.5,
                        "age": 35.0,
                    }
                ],
                "metadata": {"clinic_id": "SGH_001", "date": "2024-01-15"},
            }
        },
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""

    batch_id: str = Field(..., description="Batch identifier")

    predictions: List[DiabetesPrediction] = Field(
        ..., description="List of predictions for each patient"
    )

    summary: Dict = Field(..., description="Summary statistics for the batch")

    processing_timestamp: datetime = Field(
        ..., description="When the batch was processed"
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "batch_id": "BATCH_2024_001",
                "predictions": [
                    {
                        "patient_id": "PAT_001",
                        "diabetes_prediction": False,
                        "diabetes_probability": 0.25,
                        "confidence_score": 0.85,
                        "risk_category": "Low Risk",
                        "recommendation": "Maintain healthy lifestyle.",
                        "model_version": "1.0",
                        "prediction_timestamp": "2024-01-15T10:30:00",
                    }
                ],
                "summary": {
                    "total_patients": 1,
                    "predicted_diabetic": 0,
                    "high_risk_count": 0,
                    "average_diabetes_probability": 0.25,
                },
                "processing_timestamp": "2024-01-15T10:30:00",
            }
        },
    )


class HealthCheck(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status", examples=["healthy"])

    timestamp: datetime = Field(..., description="Health check timestamp")

    model_version: Optional[str] = Field(None, description="Current model version")

    model_loaded: bool = Field(..., description="Whether model is loaded")

    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")

    load_time: Optional[datetime] = Field(None, description="When model was loaded")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "model_version": "1.0",
                "model_loaded": True,
                "preprocessor_loaded": True,
                "load_time": "2024-01-15T08:00:00",
            }
        },
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")

    error_code: Optional[str] = Field(None, description="Specific error code")

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp"
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "detail": "Validation error in input data",
                "error_code": "VALIDATION_ERROR",
                "timestamp": "2024-01-15T10:30:00",
            }
        },
    )
