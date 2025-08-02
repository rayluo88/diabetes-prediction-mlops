"""
Prefect Workflow for Diabetes Batch Prediction Pipeline

Orchestrated workflow for batch diabetes predictions for clinic use
with data validation, model loading, and result storage.
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import IntervalSchedule
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocess import DiabetesPreprocessor


@task(name="load_champion_model", retries=2)
def load_champion_model_task():
    """Load the champion model from MLflow model registry."""
    logger = get_run_logger()
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("sqlite:///models/artifacts/mlflow.db")
        
        # Load latest champion model
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(
            "diabetes_prediction_champion",
            stages=["Production", "Staging", "None"]
        )
        
        if not model_versions:
            logger.warning("No champion model found, loading latest trained model")
            model_path = "models/trained/diabetes_optimized_random_forest.pkl"
            model = joblib.load(model_path)
        else:
            latest_version = model_versions[0]
            model_uri = f"models:/{latest_version.name}/{latest_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded champion model version {latest_version.version}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@task(name="load_preprocessor")
def load_preprocessor_task():
    """Load the fitted preprocessor."""
    logger = get_run_logger()
    
    try:
        preprocessor_path = "models/artifacts/diabetes_preprocessor.pkl"
        preprocessor = DiabetesPreprocessor.load(preprocessor_path)
        logger.info("Preprocessor loaded successfully")
        return preprocessor
        
    except Exception as e:
        logger.error(f"Failed to load preprocessor: {e}")
        raise


@task(name="validate_input_data")
def validate_input_data_task(data_path: str) -> pd.DataFrame:
    """Validate input data for batch predictions."""
    logger = get_run_logger()
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded input data: {df.shape}")
        
        # Expected feature columns
        expected_features = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree_function', 'age'
        ]
        
        # Check required columns
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data quality
        logger.info("Validating data quality...")
        
        # Check for negative values in medical features
        medical_features = ['glucose', 'blood_pressure', 'bmi', 'age']
        for feature in medical_features:
            if (df[feature] < 0).any():
                logger.warning(f"Negative values found in {feature}")
        
        # Check for extreme outliers
        outlier_checks = {
            'glucose': (0, 300),
            'blood_pressure': (40, 200),
            'bmi': (10, 70),
            'age': (18, 100)
        }
        
        for feature, (min_val, max_val) in outlier_checks.items():
            outliers = (df[feature] < min_val) | (df[feature] > max_val)
            if outliers.any():
                logger.warning(f"Outliers detected in {feature}: {outliers.sum()} values")
        
        logger.info("Data validation completed")
        return df
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise


@task(name="generate_predictions")
def generate_predictions_task(
    model, 
    preprocessor, 
    input_data: pd.DataFrame
) -> Dict:
    """Generate diabetes predictions with confidence scores."""
    logger = get_run_logger()
    
    try:
        # Preprocess input data
        X_processed = preprocessor.transform(input_data)
        logger.info(f"Data preprocessed: {X_processed.shape}")
        
        # Generate predictions
        predictions = model.predict(X_processed)
        prediction_probabilities = model.predict_proba(X_processed)
        
        # Calculate confidence scores
        confidence_scores = np.max(prediction_probabilities, axis=1)
        
        # Create results DataFrame
        results_df = input_data.copy()
        results_df['diabetes_prediction'] = predictions
        results_df['diabetes_probability'] = prediction_probabilities[:, 1]
        results_df['confidence_score'] = confidence_scores
        results_df['risk_category'] = pd.cut(
            prediction_probabilities[:, 1],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        results_df['prediction_timestamp'] = datetime.now().isoformat()
        
        # Generate summary statistics
        summary = {
            'total_patients': len(results_df),
            'predicted_diabetic': int(predictions.sum()),
            'predicted_non_diabetic': int(len(predictions) - predictions.sum()),
            'high_risk_patients': int((prediction_probabilities[:, 1] > 0.7).sum()),
            'average_risk_score': float(prediction_probabilities[:, 1].mean()),
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Generated predictions for {len(results_df)} patients")
        logger.info(f"Predicted diabetic: {summary['predicted_diabetic']}")
        logger.info(f"High risk patients: {summary['high_risk_patients']}")
        
        return {
            'predictions_df': results_df,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise


@task(name="save_predictions")
def save_predictions_task(prediction_results: Dict, output_path: str = None) -> str:
    """Save prediction results to CSV with timestamp."""
    logger = get_run_logger()
    
    try:
        # Create output directory
        os.makedirs("data/predictions", exist_ok=True)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/predictions/diabetes_predictions_{timestamp}.csv"
        
        # Save predictions
        predictions_df = prediction_results['predictions_df']
        predictions_df.to_csv(output_path, index=False)
        
        # Save summary report
        summary_path = output_path.replace('.csv', '_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(prediction_results['summary'], f, indent=2)
        
        logger.info(f"Predictions saved to: {output_path}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        raise


@flow(
    name="diabetes_batch_prediction",
    task_runner=SequentialTaskRunner(),
    description="Batch diabetes prediction pipeline for clinic use"
)
def diabetes_prediction_flow(input_data_path: str, output_path: str = None):
    """Main diabetes batch prediction workflow."""
    logger = get_run_logger()
    logger.info(f"🔮 Starting Diabetes Batch Prediction Pipeline")
    logger.info(f"📁 Input data: {input_data_path}")
    
    # Task 1: Load champion model
    model = load_champion_model_task()
    
    # Task 2: Load preprocessor
    preprocessor = load_preprocessor_task()
    
    # Task 3: Validate input data
    input_data = validate_input_data_task(input_data_path)
    
    # Task 4: Generate predictions
    prediction_results = generate_predictions_task(model, preprocessor, input_data)
    
    # Task 5: Save predictions
    output_file = save_predictions_task(prediction_results, output_path)
    
    logger.info("✅ Diabetes Prediction Pipeline Completed Successfully")
    
    return {
        'status': 'completed',
        'input_file': input_data_path,
        'output_file': output_file,
        'summary': prediction_results['summary']
    }


# Deployment for scheduled batch predictions
def create_prediction_deployment():
    """Create Prefect deployment for scheduled batch predictions."""
    
    deployment = Deployment.build_from_flow(
        flow=diabetes_prediction_flow,
        name="diabetes-batch-prediction",
        version="1.0",
        description="Daily batch diabetes predictions for Singapore clinics",
        tags=["diabetes", "prediction", "batch", "clinic"],
        parameters={
            "input_data_path": "data/processed/daily_clinic_data.csv",
            "output_path": None
        },
        work_queue_name="diabetes-prediction-queue",
        schedule=IntervalSchedule(
            interval=timedelta(hours=6),  # Run every 6 hours
            anchor_date=datetime(2024, 1, 1, 6, 0, 0)  # Start at 6 AM
        )
    )
    
    return deployment


if __name__ == "__main__":
    # Example usage with test data
    test_data_path = "data/processed/diabetes_test.csv"
    
    if os.path.exists(test_data_path):
        print("🏥 Running Diabetes Prediction Pipeline...")
        result = diabetes_prediction_flow(test_data_path)
        print(f"📊 Pipeline Result: {result}")
    else:
        print("❌ Test data not found. Run training pipeline first.")
    
    # Create deployment
    deployment = create_prediction_deployment()
    print(f"⏰ Prediction deployment created: {deployment.name}")