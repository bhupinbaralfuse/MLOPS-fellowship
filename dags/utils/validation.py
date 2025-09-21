import json
import random
import mlflow
from scipy.stats import ks_2samp

def validate_and_report(predictions_file, sample_size, mlflow_tracking_uri):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    # Sample 10% of predictions
    sample = random.sample(predictions, max(1, int(len(predictions) * sample_size)))
    
    # Calculate IoU distribution for drift detection
    ious = [pred['score'] for pred in sample]
    
    with mlflow.start_run():
        mlflow.log_param('sample_size', sample_size)
        mlflow.log_metric('num_predictions', len(predictions))
        mlflow.log_artifact(predictions_file)
        
        # Compare with previous run
        client = mlflow.tracking.MlflowClient()
        previous_run = client.search_runs()[0] if client.search_runs() else None
        if previous_run:
            try:
                previous_ious = [0.9] * len(ious)  # Placeholder for previous IoUs
                stat, p_value = ks_2samp(ious, previous_ious)
                mlflow.log_metric('drift_p_value', p_value)
                
                if p_value < 0.05:
                    mlflow.log_param('rollback_triggered', True)
                    print("Drift detected, consider reverting to previous model.")
                else:
                    mlflow.log_param('rollback_triggered', False)
            except:
                mlflow.log_param('comparison_failed', True)
    
    print(f"Validation completed: {len(sample)} samples analyzed")