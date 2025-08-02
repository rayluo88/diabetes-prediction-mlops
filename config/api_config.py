"""
API Configuration for Diabetes Prediction Service

Configuration settings for the FastAPI application including
model paths, thresholds, and deployment settings.
"""

import os
from typing import Dict, Optional

# Model configuration
MODEL_CONFIG = {
    "model_path": "models/trained/diabetes_optimized_random_forest.pkl",
    "preprocessor_path": "models/artifacts/diabetes_preprocessor.pkl",
    "model_version": "1.0.0",
    "fallback_model_path": "models/trained/diabetes_random_forest.pkl",
}

# API configuration
API_CONFIG = {
    "title": "Diabetes Prediction API",
    "description": "AI-powered diabetes prediction for Singapore healthcare system",
    "version": "1.0.0",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
}

# Healthcare thresholds (Singapore-specific)
HEALTHCARE_CONFIG = {
    "risk_thresholds": {
        "low": 0.3,
        "medium": 0.7,
        "high": 1.0,
    },
    "confidence_threshold": 0.75,
    "batch_size_limit": 1000,
    "prediction_timeout": 30,  # seconds
}

# MLflow configuration
MLFLOW_CONFIG = {
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///models/artifacts/mlflow.db"),
    "experiment_name": "diabetes_prediction",
    "model_registry_name": "diabetes_prediction_champion",
}

# Monitoring configuration
MONITORING_CONFIG = {
    "drift_threshold": 0.25,
    "alert_email": os.getenv("ALERT_EMAIL", "alerts@singapore-health.gov.sg"),
    "monitoring_interval": 3600,  # 1 hour in seconds
}


def get_config(section: str) -> Optional[Dict]:
    """Get configuration section."""
    config_map = {
        "model": MODEL_CONFIG,
        "api": API_CONFIG,
        "healthcare": HEALTHCARE_CONFIG,
        "mlflow": MLFLOW_CONFIG,
        "monitoring": MONITORING_CONFIG,
    }
    return config_map.get(section)