"""
FastAPI Diabetes Prediction Service

Production-ready API for real-time diabetes prediction with
comprehensive logging, validation, and monitoring capabilities.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

# Add src to Python path (must be before local imports)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DiabetesInput,
    DiabetesPrediction,
    HealthCheck,
)
from data.preprocess import DiabetesPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_version = None
load_time = None


async def load_model_and_preprocessor():
    """Load model and preprocessor on startup."""
    global model, preprocessor, model_version, load_time

    try:
        logger.info("Loading diabetes prediction model...")

        # Set MLflow tracking URI
        mlflow.set_tracking_uri("sqlite:///models/artifacts/mlflow.db")

        # Try to load champion model from MLflow registry
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(
                "diabetes_prediction_champion", stages=["Production", "Staging", "None"]
            )

            if model_versions:
                latest_version = model_versions[0]
                model_uri = f"models:/{latest_version.name}/{latest_version.version}"
                model = mlflow.sklearn.load_model(model_uri)
                model_version = latest_version.version
                logger.info(f"Loaded champion model version {model_version}")
            else:
                raise Exception("No champion model found")

        except Exception as e:
            logger.warning(f"Failed to load from MLflow registry: {e}")
            logger.info("Loading fallback model...")
            model_path = "models/trained/diabetes_optimized_random_forest.pkl"
            model = joblib.load(model_path)
            model_version = "fallback_v1"

        # Load preprocessor
        preprocessor_path = "models/artifacts/diabetes_preprocessor.pkl"
        preprocessor = DiabetesPreprocessor.load(preprocessor_path)

        load_time = datetime.now()
        logger.info("Model and preprocessor loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await load_model_and_preprocessor()
    yield
    # Shutdown
    logger.info("Shutting down diabetes prediction API")


# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Production-ready API for diabetes prediction using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    """Dependency to get loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def get_preprocessor():
    """Dependency to get loaded preprocessor."""
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    return preprocessor


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        model_version=model_version,
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        load_time=load_time,
    )


@app.post("/predict", response_model=DiabetesPrediction)
async def predict_diabetes(
    patient_data: DiabetesInput,
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor),
):
    """
    Predict diabetes for a single patient.

    Returns prediction with probability scores and risk assessment.
    """
    try:
        logger.info("Processing diabetes prediction request")

        # Convert input to DataFrame
        input_df = pd.DataFrame([patient_data.dict()])

        # Validate input ranges
        validation_errors = validate_medical_inputs(patient_data)
        if validation_errors:
            raise HTTPException(status_code=422, detail=validation_errors)

        # Preprocess data
        X_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0]

        # Calculate risk assessment
        diabetes_probability = float(prediction_proba[1])
        confidence_score = float(np.max(prediction_proba))

        # Determine risk category
        if diabetes_probability < 0.3:
            risk_category = "Low Risk"
            recommendation = "Maintain healthy lifestyle. Regular checkups recommended."
        elif diabetes_probability < 0.7:
            risk_category = "Medium Risk"
            recommendation = (
                "Lifestyle modifications recommended. Consult healthcare provider."
            )
        else:
            risk_category = "High Risk"
            recommendation = "Immediate medical consultation recommended."

        result = DiabetesPrediction(
            patient_id=patient_data.patient_id,
            diabetes_prediction=bool(prediction),
            diabetes_probability=diabetes_probability,
            confidence_score=confidence_score,
            risk_category=risk_category,
            recommendation=recommendation,
            model_version=model_version,
            prediction_timestamp=datetime.now(),
        )

        logger.info(
            f"Prediction completed - Risk: {risk_category}, Probability: {diabetes_probability:.3f}"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor),
):
    """
    Batch prediction for multiple patients.

    Processes multiple patients efficiently with comprehensive results.
    """
    try:
        logger.info(f"Processing batch prediction for {len(request.patients)} patients")

        # Convert batch input to DataFrame
        input_data = [patient.dict() for patient in request.patients]
        input_df = pd.DataFrame(input_data)

        # Validate all inputs
        validation_errors = []
        for i, patient in enumerate(request.patients):
            errors = validate_medical_inputs(patient)
            if errors:
                validation_errors.append(f"Patient {i+1}: {errors}")

        if validation_errors:
            raise HTTPException(status_code=422, detail=validation_errors)

        # Preprocess data
        X_processed = preprocessor.transform(input_df)

        # Make batch predictions
        predictions = model.predict(X_processed)
        prediction_probas = model.predict_proba(X_processed)

        # Process results
        results = []
        for i, (patient, pred, proba) in enumerate(
            zip(request.patients, predictions, prediction_probas)
        ):
            diabetes_probability = float(proba[1])
            confidence_score = float(np.max(proba))

            # Risk categorization
            if diabetes_probability < 0.3:
                risk_category = "Low Risk"
                recommendation = (
                    "Maintain healthy lifestyle. Regular checkups recommended."
                )
            elif diabetes_probability < 0.7:
                risk_category = "Medium Risk"
                recommendation = (
                    "Lifestyle modifications recommended. Consult healthcare provider."
                )
            else:
                risk_category = "High Risk"
                recommendation = "Immediate medical consultation recommended."

            result = DiabetesPrediction(
                patient_id=patient.patient_id,
                diabetes_prediction=bool(pred),
                diabetes_probability=diabetes_probability,
                confidence_score=confidence_score,
                risk_category=risk_category,
                recommendation=recommendation,
                model_version=model_version,
                prediction_timestamp=datetime.now(),
            )
            results.append(result)

        # Generate batch summary
        summary = {
            "total_patients": len(results),
            "predicted_diabetic": sum(1 for r in results if r.diabetes_prediction),
            "high_risk_count": sum(
                1 for r in results if r.risk_category == "High Risk"
            ),
            "average_diabetes_probability": np.mean(
                [r.diabetes_probability for r in results]
            ),
            "processing_time_seconds": (
                datetime.now() - datetime.now()
            ).total_seconds(),
        }

        # Schedule background task for logging
        background_tasks.add_task(log_batch_prediction, request.batch_id, summary)

        response = BatchPredictionResponse(
            batch_id=request.batch_id,
            predictions=results,
            summary=summary,
            processing_timestamp=datetime.now(),
        )

        logger.info(
            f"Batch prediction completed - {summary['predicted_diabetic']}/{summary['total_patients']} predicted diabetic"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during batch prediction"
        )


def validate_medical_inputs(patient_data: DiabetesInput) -> Optional[str]:
    """Validate medical input ranges."""
    errors = []

    # Define realistic medical ranges
    ranges = {
        "pregnancies": (0, 20),
        "glucose": (50, 300),
        "blood_pressure": (40, 200),
        "skin_thickness": (0, 100),
        "insulin": (0, 1000),
        "bmi": (10, 80),
        "diabetes_pedigree_function": (0, 5),
        "age": (18, 120),
    }

    for field, (min_val, max_val) in ranges.items():
        value = getattr(patient_data, field)
        if not (min_val <= value <= max_val):
            errors.append(f"{field} must be between {min_val} and {max_val}")

    return "; ".join(errors) if errors else None


async def log_batch_prediction(batch_id: str, summary: Dict):
    """Background task to log batch prediction details."""
    logger.info(f"Batch {batch_id} summary: {summary}")


@app.get("/model/info")
async def model_info():
    """Get model information and metadata."""
    return {
        "model_version": model_version,
        "model_type": "Random Forest Classifier",
        "load_time": load_time,
        "features": [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree_function",
            "age",
        ],
        "target": "diabetes_prediction",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "diabetes_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
