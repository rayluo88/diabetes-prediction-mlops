"""
Prefect Workflow for Diabetes Model Monitoring

Automated monitoring pipeline with conditional workflows for
retraining and alerting based on model performance and drift.
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.schedules import IntervalSchedule
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monitoring.diabetes_monitor import DiabetesModelMonitor, load_model_and_predict
from workflows.training_pipeline import diabetes_training_flow


@task(name="load_monitoring_data", retries=2)
def load_monitoring_data_task(data_path: str = "data/processed/diabetes_test_processed.csv"):
    """Load data for monitoring analysis."""
    logger = get_run_logger()
    
    if not os.path.exists(data_path):
        logger.error(f"Monitoring data not found: {data_path}")
        raise FileNotFoundError(f"Monitoring data required: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Monitoring data loaded: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load monitoring data: {e}")
        raise


@task(name="generate_model_predictions")
def generate_predictions_task(monitoring_data: pd.DataFrame):
    """Generate predictions for performance monitoring."""
    logger = get_run_logger()
    
    try:
        predictions = load_model_and_predict(monitoring_data)
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise


@task(name="run_monitoring_analysis")
def run_monitoring_analysis_task(monitoring_data: pd.DataFrame, 
                                predictions: np.ndarray) -> Dict:
    """Run comprehensive monitoring analysis."""
    logger = get_run_logger()
    
    try:
        # Initialize monitor
        monitor = DiabetesModelMonitor(
            alert_email="data-science-team@singapore-health.gov.sg",
            alert_threshold=0.25  # 25% drift threshold
        )
        
        # Run comprehensive monitoring
        if len(predictions) > 0:
            results = monitor.comprehensive_monitoring_check(monitoring_data, predictions)
        else:
            results = monitor.comprehensive_monitoring_check(monitoring_data)
        
        logger.info(f"Monitoring analysis completed - Status: {results['monitoring_status']}")
        return results
        
    except Exception as e:
        logger.error(f"Monitoring analysis failed: {e}")
        raise


@task(name="evaluate_retraining_criteria")
def evaluate_retraining_criteria_task(monitoring_results: Dict) -> Dict:
    """Evaluate if model retraining is needed."""
    logger = get_run_logger()
    
    retraining_decision = {
        'retrain_needed': False,
        'retrain_reasons': [],
        'priority': 'low'
    }
    
    # Criterion 1: Data drift above threshold
    if monitoring_results.get('data_drift', {}).get('drift_share', 0) > 0.3:
        retraining_decision['retrain_needed'] = True
        retraining_decision['retrain_reasons'].append('High data drift detected')
        retraining_decision['priority'] = 'high'
    
    # Criterion 2: Model performance degradation
    performance = monitoring_results.get('model_performance', {})
    if performance.get('accuracy', 1.0) < 0.75:  # Singapore healthcare threshold
        retraining_decision['retrain_needed'] = True
        retraining_decision['retrain_reasons'].append('Model accuracy below threshold')
        retraining_decision['priority'] = 'critical'
    
    if performance.get('f1', 1.0) < 0.70:
        retraining_decision['retrain_needed'] = True
        retraining_decision['retrain_reasons'].append('F1 score below threshold')
        retraining_decision['priority'] = 'high'
    
    # Criterion 3: Significant bias detected
    bias_analysis = monitoring_results.get('bias_analysis', {})
    for feature, analysis in bias_analysis.items():
        if analysis.get('demographic_parity_difference', 0) > 0.15:
            retraining_decision['retrain_needed'] = True
            retraining_decision['retrain_reasons'].append(f'Bias detected in {feature}')
            if retraining_decision['priority'] == 'low':
                retraining_decision['priority'] = 'medium'
    
    # Criterion 4: Data quality issues
    quality = monitoring_results.get('data_quality', {})
    if quality.get('failed_tests', 0) > 2:
        retraining_decision['retrain_needed'] = True
        retraining_decision['retrain_reasons'].append('Multiple data quality test failures')
        if retraining_decision['priority'] == 'low':
            retraining_decision['priority'] = 'medium'
    
    logger.info(f"Retraining evaluation: {retraining_decision}")
    return retraining_decision


@task(name="trigger_automated_retraining")
def trigger_retraining_task(retraining_decision: Dict):
    """Trigger automated model retraining if needed."""
    logger = get_run_logger()
    
    if not retraining_decision['retrain_needed']:
        logger.info("No retraining needed")
        return {'retraining_triggered': False}
    
    try:
        logger.info(f"Triggering automated retraining - Priority: {retraining_decision['priority']}")
        logger.info(f"Reasons: {retraining_decision['retrain_reasons']}")
        
        # Critical priority: immediate retraining
        if retraining_decision['priority'] == 'critical':
            logger.info("CRITICAL: Triggering immediate model retraining")
            # In production, this would trigger the training pipeline
            # training_result = diabetes_training_flow()
            logger.info("Immediate retraining would be triggered here")
        
        # High/Medium priority: scheduled retraining
        else:
            logger.info(f"Scheduling retraining for {retraining_decision['priority']} priority issues")
        
        return {
            'retraining_triggered': True,
            'priority': retraining_decision['priority'],
            'reasons': retraining_decision['retrain_reasons'],
            'scheduled_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise


@task(name="send_monitoring_alerts")
def send_alerts_task(monitoring_results: Dict, retraining_results: Dict):
    """Send monitoring alerts and notifications."""
    logger = get_run_logger()
    
    try:
        status = monitoring_results['monitoring_status']
        alerts = monitoring_results.get('alerts', [])
        
        # Prepare alert summary
        alert_summary = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': status,
            'alerts_count': len(alerts),
            'alerts': alerts,
            'retraining_triggered': retraining_results.get('retraining_triggered', False),
            'data_shape': monitoring_results.get('data_shape', 'unknown')
        }
        
        # Log alerts
        if status in ['alert', 'critical']:
            logger.warning(f"🚨 DIABETES MODEL ALERT - Status: {status.upper()}")
            for alert in alerts:
                logger.warning(f"  ⚠️  {alert}")
        
        if retraining_results.get('retraining_triggered'):
            logger.info(f"🔄 Model retraining triggered - Priority: {retraining_results.get('priority')}")
        
        # In production, send actual notifications
        logger.info("Alert notifications would be sent to:")
        logger.info("  - Data Science Team: data-science-team@singapore-health.gov.sg")
        logger.info("  - Healthcare Operations: ops@singapore-health.gov.sg")
        logger.info("  - Slack: #diabetes-model-alerts")
        
        return alert_summary
        
    except Exception as e:
        logger.error(f"Failed to send alerts: {e}")
        raise


@task(name="generate_monitoring_dashboard")
def generate_dashboard_task(monitoring_results: Dict):
    """Generate monitoring dashboard data."""
    logger = get_run_logger()
    
    try:
        # Create dashboard summary
        dashboard_data = {
            'last_updated': datetime.now().isoformat(),
            'overall_status': monitoring_results['monitoring_status'],
            'metrics': {
                'data_drift_share': monitoring_results.get('data_drift', {}).get('drift_share', 0),
                'data_quality_score': (
                    monitoring_results.get('data_quality', {}).get('passed_tests', 0) /
                    max(monitoring_results.get('data_quality', {}).get('total_tests', 1), 1)
                ),
                'model_accuracy': monitoring_results.get('model_performance', {}).get('accuracy', 0),
                'bias_max_difference': max([
                    analysis.get('demographic_parity_difference', 0)
                    for analysis in monitoring_results.get('bias_analysis', {}).values()
                ] or [0])
            },
            'alerts': monitoring_results.get('alerts', []),
            'recommendations': []
        }
        
        # Generate recommendations
        if dashboard_data['metrics']['data_drift_share'] > 0.2:
            dashboard_data['recommendations'].append('Monitor data sources for distribution changes')
        
        if dashboard_data['metrics']['model_accuracy'] < 0.8:
            dashboard_data['recommendations'].append('Consider model retraining or hyperparameter tuning')
        
        if dashboard_data['metrics']['bias_max_difference'] > 0.1:
            dashboard_data['recommendations'].append('Review model for fairness across demographic groups')
        
        # Save dashboard data
        os.makedirs("models/dashboard", exist_ok=True)
        dashboard_path = "models/dashboard/monitoring_dashboard.json"
        
        import json
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        logger.info(f"Dashboard data saved: {dashboard_path}")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
        raise


@flow(
    name="diabetes_monitoring_pipeline",
    task_runner=SequentialTaskRunner(),
    description="Comprehensive diabetes model monitoring with automated retraining"
)
def diabetes_monitoring_flow(data_path: str = "data/processed/diabetes_test_processed.csv"):
    """Main diabetes monitoring workflow with conditional retraining."""
    logger = get_run_logger()
    logger.info("🔍 Starting Diabetes Model Monitoring Pipeline")
    
    # Task 1: Load monitoring data
    monitoring_data = load_monitoring_data_task(data_path)
    
    # Task 2: Generate predictions
    predictions = generate_predictions_task(monitoring_data)
    
    # Task 3: Run monitoring analysis
    monitoring_results = run_monitoring_analysis_task(monitoring_data, predictions)
    
    # Task 4: Evaluate retraining criteria
    retraining_decision = evaluate_retraining_criteria_task(monitoring_results)
    
    # Task 5: Trigger retraining if needed
    retraining_results = trigger_retraining_task(retraining_decision)
    
    # Task 6: Send alerts
    alert_summary = send_alerts_task(monitoring_results, retraining_results)
    
    # Task 7: Generate dashboard
    dashboard_data = generate_dashboard_task(monitoring_results)
    
    logger.info("✅ Diabetes Monitoring Pipeline Completed")
    
    return {
        'status': 'completed',
        'monitoring_status': monitoring_results['monitoring_status'],
        'retraining_triggered': retraining_results.get('retraining_triggered', False),
        'alerts_count': len(monitoring_results.get('alerts', [])),
        'dashboard_updated': True
    }


# Deployment for scheduled monitoring
def create_monitoring_deployment():
    """Create Prefect deployment for scheduled monitoring."""
    
    deployment = Deployment.build_from_flow(
        flow=diabetes_monitoring_flow,
        name="diabetes-model-monitoring",
        version="1.0",
        description="Automated diabetes model monitoring with drift detection and retraining",
        tags=["monitoring", "diabetes", "drift-detection", "alerting"],
        parameters={
            "data_path": "data/processed/diabetes_test_processed.csv"
        },
        work_queue_name="diabetes-monitoring-queue",
        schedule=IntervalSchedule(
            interval=timedelta(hours=4),  # Monitor every 4 hours
            anchor_date=datetime(2024, 1, 1, 0, 0, 0)
        )
    )
    
    return deployment


if __name__ == "__main__":
    # Run monitoring pipeline
    print("🔍 Running Diabetes Model Monitoring Pipeline...")
    
    test_data_path = "data/processed/diabetes_test_processed.csv"
    if os.path.exists(test_data_path):
        result = diabetes_monitoring_flow(test_data_path)
        print(f"📊 Monitoring Result: {result}")
    else:
        print("❌ Test data not found. Run data pipeline first.")
    
    # Create deployment
    deployment = create_monitoring_deployment()
    print(f"📊 Monitoring deployment created: {deployment.name}")