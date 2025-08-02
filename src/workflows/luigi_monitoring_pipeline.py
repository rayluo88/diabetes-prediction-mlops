"""
Luigi Workflow for Diabetes Model Monitoring

Luigi-based monitoring pipeline with conditional workflows for
retraining and alerting based on model performance and drift.
"""

import luigi
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monitoring.diabetes_monitor import DiabetesModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadCurrentDataTask(luigi.Task):
    """Load current data for monitoring analysis."""
    
    date = luigi.DateParameter(default=datetime.now().date())
    
    def output(self):
        return luigi.LocalTarget(f'data/monitoring/current_data_{self.date}.csv')
    
    def run(self):
        logger.info(f"Loading current data for monitoring on {self.date}...")
        try:
            # For demo purposes, use test data as "current" data
            # In production, this would load real current data
            test_data_path = 'data/processed/diabetes_test_processed.csv'
            
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"Test data not found: {test_data_path}")
            
            current_data = pd.read_csv(test_data_path)
            
            # Add some synthetic drift for demonstration
            np.random.seed(42)
            drift_factor = np.random.normal(1.0, 0.1, size=len(current_data))
            current_data['glucose'] = current_data['glucose'] * drift_factor
            
            # Save current data
            os.makedirs('data/monitoring', exist_ok=True)
            current_data.to_csv(self.output().path, index=False)
            
            logger.info(f"✅ Current data loaded: {len(current_data)} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to load current data: {e}")
            raise


class RunMonitoringAnalysisTask(luigi.Task):
    """Run comprehensive monitoring analysis with Evidently."""
    
    date = luigi.DateParameter(default=datetime.now().date())
    
    def requires(self):
        return LoadCurrentDataTask(date=self.date)
    
    def output(self):
        return luigi.LocalTarget(f'models/monitoring/monitoring_results_{self.date}.json')
    
    def run(self):
        logger.info(f"Running monitoring analysis for {self.date}...")
        try:
            # Load current data
            current_data = pd.read_csv(self.input().path)
            
            # Load reference data (training data)
            reference_data = pd.read_csv('data/processed/diabetes_train_processed.csv')
            
            # Initialize monitor
            monitor = DiabetesModelMonitor()
            
            # Generate monitoring reports
            logger.info("Generating data drift report...")
            drift_report = monitor.generate_data_drift_report(current_data, reference_data)
            
            logger.info("Generating model performance report...")
            performance_report = monitor.generate_model_performance_report(current_data)
            
            logger.info("Detecting bias and fairness...")
            bias_report = monitor.detect_bias_and_fairness(current_data, 
                                                         current_data['target'] if 'target' in current_data.columns else None)
            
            # Combine results
            monitoring_results = {
                'date': str(self.date),
                'timestamp': datetime.now().isoformat(),
                'data_drift': drift_report,
                'model_performance': performance_report,
                'bias_analysis': bias_report,
                'summary': {
                    'samples_analyzed': len(current_data),
                    'drift_detected': drift_report.get('drift_share', 0) > 0.3,
                    'performance_degraded': performance_report.get('accuracy', 1.0) < 0.75,
                    'bias_detected': any(result.get('demographic_parity_difference', 0) > 0.15 
                                       for result in bias_report.values() if isinstance(result, dict))
                }
            }
            
            # Save results
            import json
            os.makedirs('models/monitoring', exist_ok=True)
            with self.output().open('w') as f:
                json.dump(monitoring_results, f, indent=2)
            
            logger.info("✅ Monitoring analysis completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to run monitoring analysis: {e}")
            raise


class EvaluateRetrainingCriteriaTask(luigi.Task):
    """Evaluate whether model retraining is needed based on monitoring results."""
    
    date = luigi.DateParameter(default=datetime.now().date())
    
    def requires(self):
        return RunMonitoringAnalysisTask(date=self.date)
    
    def output(self):
        return luigi.LocalTarget(f'models/monitoring/retraining_decision_{self.date}.json')
    
    def run(self):
        logger.info("Evaluating retraining criteria...")
        try:
            # Load monitoring results
            import json
            with self.input().open('r') as f:
                monitoring_results = json.load(f)
            
            summary = monitoring_results.get('summary', {})
            
            retraining_decision = {
                'date': str(self.date),
                'timestamp': datetime.now().isoformat(),
                'retrain_needed': False,
                'priority': 'normal',
                'reasons': [],
                'criteria_evaluation': {}
            }
            
            # Criterion 1: Data drift above threshold
            drift_detected = summary.get('drift_detected', False)
            if drift_detected:
                retraining_decision['retrain_needed'] = True
                retraining_decision['reasons'].append('Significant data drift detected')
                retraining_decision['priority'] = 'high'
            
            retraining_decision['criteria_evaluation']['data_drift'] = drift_detected
            
            # Criterion 2: Model performance degradation
            performance_degraded = summary.get('performance_degraded', False)
            if performance_degraded:
                retraining_decision['retrain_needed'] = True
                retraining_decision['reasons'].append('Model performance below healthcare threshold')
                retraining_decision['priority'] = 'critical'
            
            retraining_decision['criteria_evaluation']['performance_degraded'] = performance_degraded
            
            # Criterion 3: Bias detection
            bias_detected = summary.get('bias_detected', False)
            if bias_detected:
                retraining_decision['retrain_needed'] = True
                retraining_decision['reasons'].append('Demographic bias detected')
                if retraining_decision['priority'] != 'critical':
                    retraining_decision['priority'] = 'high'
            
            retraining_decision['criteria_evaluation']['bias_detected'] = bias_detected
            
            # Save decision
            with self.output().open('w') as f:
                json.dump(retraining_decision, f, indent=2)
            
            if retraining_decision['retrain_needed']:
                logger.warning(f"⚠️  Retraining needed! Priority: {retraining_decision['priority']}")
                logger.warning(f"Reasons: {', '.join(retraining_decision['reasons'])}")
            else:
                logger.info("✅ No retraining needed - model performance within acceptable limits")
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate retraining criteria: {e}")
            raise


class SendAlertsTask(luigi.Task):
    """Send alerts based on monitoring results."""
    
    date = luigi.DateParameter(default=datetime.now().date())
    
    def requires(self):
        return EvaluateRetrainingCriteriaTask(date=self.date)
    
    def output(self):
        return luigi.LocalTarget(f'models/monitoring/alerts_sent_{self.date}.log')
    
    def run(self):
        logger.info("Processing alerts...")
        try:
            # Load retraining decision
            import json
            with self.input().open('r') as f:
                retraining_decision = json.load(f)
            
            alert_log = []
            
            if retraining_decision['retrain_needed']:
                priority = retraining_decision['priority']
                reasons = retraining_decision['reasons']
                
                # Create alert message
                alert_message = f"""
🚨 DIABETES MODEL ALERT - Priority: {priority.upper()}

Date: {self.date}
Status: Retraining Required

Issues Detected:
{chr(10).join(f'• {reason}' for reason in reasons)}

Recommended Actions:
• Review monitoring dashboard
• Initiate model retraining pipeline
• Validate new model before deployment
• Update Singapore healthcare teams

Dashboard: http://localhost:8085
MLflow: http://localhost:5000
                """
                
                alert_log.append(f"[{datetime.now()}] ALERT SENT - Priority: {priority}")
                alert_log.append(f"Recipients: Healthcare team, Data science team")
                alert_log.append(f"Message: {alert_message}")
                
                # In production, send actual alerts (email, Slack, etc.)
                logger.warning(f"🚨 ALERT: {alert_message}")
                
            else:
                alert_log.append(f"[{datetime.now()}] No alerts needed - system healthy")
                logger.info("✅ No alerts needed - system operating normally")
            
            # Save alert log
            with self.output().open('w') as f:
                f.write('\n'.join(alert_log))
            
        except Exception as e:
            logger.error(f"❌ Failed to send alerts: {e}")
            raise


class ConditionalRetrainingTask(luigi.Task):
    """Conditionally trigger retraining if needed."""
    
    date = luigi.DateParameter(default=datetime.now().date())
    
    def requires(self):
        return EvaluateRetrainingCriteriaTask(date=self.date)
    
    def output(self):
        return luigi.LocalTarget(f'models/monitoring/retraining_triggered_{self.date}.flag')
    
    def run(self):
        logger.info("Checking if retraining should be triggered...")
        try:
            # Load retraining decision
            import json
            with self.input().open('r') as f:
                retraining_decision = json.load(f)
            
            if retraining_decision['retrain_needed']:
                logger.info("🔄 Triggering model retraining...")
                
                # In production, this would trigger the training pipeline
                # For now, just log the action
                action_log = f"""
Retraining triggered at {datetime.now()}
Priority: {retraining_decision['priority']}
Reasons: {', '.join(retraining_decision['reasons'])}

Next steps:
1. Run training pipeline
2. Validate new model
3. Deploy if validation passes
4. Update monitoring baseline
                """
                
                with self.output().open('w') as f:
                    f.write(action_log)
                
                logger.info("✅ Retraining pipeline triggered successfully")
                
            else:
                logger.info("No retraining needed")
                
                with self.output().open('w') as f:
                    f.write(f"No retraining needed on {self.date} - model performance acceptable\n")
            
        except Exception as e:
            logger.error(f"❌ Failed to process retraining trigger: {e}")
            raise


class DiabetesMonitoringPipeline(luigi.Task):
    """Main monitoring pipeline task."""
    
    date = luigi.DateParameter(default=datetime.now().date())
    
    def requires(self):
        return [
            SendAlertsTask(date=self.date),
            ConditionalRetrainingTask(date=self.date)
        ]
    
    def output(self):
        return luigi.LocalTarget(f'models/monitoring/monitoring_pipeline_complete_{self.date}.flag')
    
    def run(self):
        logger.info("🎉 Diabetes monitoring pipeline completed successfully!")
        
        with self.output().open('w') as f:
            f.write(f"Diabetes Monitoring Pipeline completed at {datetime.now()}\n")
            f.write(f"Date: {self.date}\n")
            f.write("All monitoring tasks executed successfully:\n")
            f.write("1. ✅ Data Loading\n")
            f.write("2. ✅ Monitoring Analysis\n")
            f.write("3. ✅ Retraining Evaluation\n")
            f.write("4. ✅ Alert Processing\n")
            f.write("5. ✅ Conditional Retraining\n")
            f.write("\n🏥 Singapore healthcare monitoring active!\n")


if __name__ == '__main__':
    # Run the monitoring pipeline
    luigi.run()