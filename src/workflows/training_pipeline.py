"""
Prefect Workflow for Diabetes Model Training Pipeline

Fully orchestrated workflow for end-to-end diabetes model training
with proper error handling, logging, and scheduling capabilities.
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.load_diabetes_data import main as load_data
from data.preprocess import preprocess_diabetes_data
from models.train_diabetes_model import DiabetesModelTrainer


@task(name="load_diabetes_dataset", retries=2, retry_delay_seconds=10)
def load_diabetes_data_task():
    """Load UCI diabetes dataset with error handling."""
    logger = get_run_logger()
    logger.info("Loading diabetes dataset...")
    
    try:
        file_paths = load_data()
        logger.info(f"Dataset loaded successfully: {file_paths}")
        return file_paths
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


@task(name="preprocess_diabetes_data", retries=1, retry_delay_seconds=5)
def preprocess_data_task(file_paths):
    """Preprocess diabetes data for training."""
    logger = get_run_logger()
    logger.info("Preprocessing diabetes data...")
    
    try:
        train_processed, test_processed, preprocessor = preprocess_diabetes_data(
            file_paths['train'], 
            file_paths['test']
        )
        
        logger.info(f"Data preprocessing completed")
        logger.info(f"Training samples: {len(train_processed)}")
        logger.info(f"Test samples: {len(test_processed)}")
        
        return {
            'train_processed': train_processed,
            'test_processed': test_processed,
            'preprocessor': preprocessor
        }
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


@task(name="train_diabetes_models", retries=1, retry_delay_seconds=30)
def train_models_task(processed_data):
    """Train diabetes prediction models with MLflow tracking."""
    logger = get_run_logger()
    logger.info("Starting model training...")
    
    try:
        trainer = DiabetesModelTrainer(experiment_name="diabetes_prediction_workflow")
        model, metrics = trainer.train_complete_pipeline()
        
        logger.info("Model training completed successfully")
        logger.info(f"Best model metrics: {metrics}")
        
        return {
            'model': model,
            'metrics': metrics,
            'training_timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


@task(name="validate_model_performance")
def validate_model_task(training_results):
    """Validate model meets minimum performance thresholds."""
    logger = get_run_logger()
    
    metrics = training_results['metrics']
    
    # Define minimum thresholds for Singapore healthcare deployment
    min_thresholds = {
        'accuracy': 0.75,
        'precision': 0.70,
        'recall': 0.65,
        'f1_score': 0.70,
        'roc_auc': 0.75
    }
    
    validation_passed = True
    for metric, threshold in min_thresholds.items():
        if metrics[metric] < threshold:
            logger.error(f"{metric} ({metrics[metric]:.3f}) below threshold ({threshold})")
            validation_passed = False
        else:
            logger.info(f"✅ {metric}: {metrics[metric]:.3f} (threshold: {threshold})")
    
    if not validation_passed:
        raise ValueError("Model performance below healthcare deployment thresholds")
    
    logger.info("Model validation passed - ready for deployment")
    return validation_passed


@task(name="generate_model_report")
def generate_report_task(training_results, file_paths):
    """Generate comprehensive model training report."""
    logger = get_run_logger()
    
    report_data = {
        'training_timestamp': training_results['training_timestamp'],
        'dataset_info': {
            'train_samples': len(pd.read_csv(file_paths['train'])),
            'test_samples': len(pd.read_csv(file_paths['test'])),
            'features': 8
        },
        'model_performance': training_results['metrics'],
        'deployment_ready': True
    }
    
    # Save report
    os.makedirs("models/reports", exist_ok=True)
    report_path = f"models/reports/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Training report saved: {report_path}")
    return report_path


@flow(
    name="diabetes_training_pipeline",
    task_runner=SequentialTaskRunner(),
    description="Complete diabetes model training pipeline with MLflow tracking"
)
def diabetes_training_flow():
    """Main diabetes training workflow."""
    logger = get_run_logger()
    logger.info("🚀 Starting Diabetes Training Pipeline")
    
    # Task 1: Load dataset
    file_paths = load_diabetes_data_task()
    
    # Task 2: Preprocess data
    processed_data = preprocess_data_task(file_paths)
    
    # Task 3: Train models
    training_results = train_models_task(processed_data)
    
    # Task 4: Validate performance
    validation_passed = validate_model_task(training_results)
    
    # Task 5: Generate report
    report_path = generate_report_task(training_results, file_paths)
    
    logger.info("✅ Diabetes Training Pipeline Completed Successfully")
    return {
        'status': 'completed',
        'model_metrics': training_results['metrics'],
        'report_path': report_path,
        'validation_passed': validation_passed
    }


# Deployment for scheduled training
def create_training_deployment():
    """Create Prefect deployment for scheduled model retraining."""
    
    deployment = Deployment.build_from_flow(
        flow=diabetes_training_flow,
        name="diabetes-model-training",
        version="1.0",
        description="Scheduled diabetes model training for Singapore healthcare",
        tags=["diabetes", "healthcare", "singapore", "mlops"],
        parameters={},
        work_queue_name="diabetes-training-queue",
        schedule=CronSchedule(
            cron="0 2 * * 1",  # Weekly retraining every Monday at 2 AM
            timezone="Asia/Singapore"
        )
    )
    
    return deployment


if __name__ == "__main__":
    # Run the workflow locally
    print("🏥 Running Diabetes Training Pipeline...")
    result = diabetes_training_flow()
    print(f"📊 Pipeline Result: {result}")
    
    # Create deployment
    deployment = create_training_deployment()
    print(f"📅 Training deployment created: {deployment.name}")