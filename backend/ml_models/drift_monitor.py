# backend/ml_models/drift_monitor.py
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import joblib
import json
import os
from datetime import datetime
import mlflow
from backend.ml_models.config import CONFIG

class DataDriftMonitor:
    def __init__(self, reference_data_path=None):
        """Initialize the drift monitor with reference data or load it"""
        self.reference_stats = None
        self.reference_data_path = reference_data_path or "backend/data/reference_stats.json"
        self.drift_threshold = 0.05  # KS test p-value threshold
        self.load_reference_stats()
    
    def load_reference_stats(self):
        """Load reference statistics if they exist"""
        if os.path.exists(self.reference_data_path):
            with open(self.reference_data_path, 'r') as f:
                self.reference_stats = json.load(f)
            print(f"Loaded reference statistics from {self.reference_data_path}")
        else:
            print("No reference statistics found. Generate them with generate_reference_stats()")
    
    def generate_reference_stats(self, features_dir=CONFIG["training_data_path"]):
        """Generate reference statistics from training data"""
        all_data = []
        # Load a sample of training data files
        files = os.listdir(features_dir)
        csv_files = [f for f in files if f.endswith('.csv')][:30]  # Limit to 30 files for efficiency
        
        for file in csv_files:
            try:
                df = pd.read_csv(f"{features_dir}{file}", header=None)
                flattened = pd.concat([df.iloc[:, 0], df.iloc[:, 1]], axis=0, ignore_index=True)
                all_data.append(flattened)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, axis=1).T
            
            # Calculate statistics
            self.reference_stats = {
                "mean": combined_data.mean().tolist(),
                "std": combined_data.std().tolist(),
                "min": combined_data.min().tolist(),
                "max": combined_data.max().tolist(),
                "q25": combined_data.quantile(0.25).tolist(),
                "q50": combined_data.quantile(0.50).tolist(),
                "q75": combined_data.quantile(0.75).tolist(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save the reference statistics
            with open(self.reference_data_path, 'w') as f:
                json.dump(self.reference_stats, f)
            
            print(f"Generated reference statistics and saved to {self.reference_data_path}")
            
            # Log reference stats in MLflow
            mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
            with mlflow.start_run(run_name="reference_stats_generation"):
                mlflow.log_artifact(self.reference_data_path)
                mlflow.log_param("num_files_processed", len(csv_files))
                mlflow.log_param("num_features", len(self.reference_stats["mean"]))
        else:
            print("No data files were processed successfully.")
    
    def check_drift(self, new_data):
        """
        Check for data drift between reference data and new data
        Returns dict with drift metrics and boolean indicating if drift detected
        """
        if self.reference_stats is None:
            raise ValueError("Reference statistics not loaded. Run generate_reference_stats() first.")
        
        # Convert the new data to a pandas DataFrame if it's not already
        if isinstance(new_data, np.ndarray):
            new_data = pd.DataFrame(new_data)
        
        # If new_data is a single row/sample, we need to compare its values to the reference distributions
        drift_metrics = {}
        drift_detected = False
        
        # For a single sample, we'll check if values fall within expected ranges
        if len(new_data) == 1:
            # Flatten the data if necessary
            if new_data.shape[1] < len(self.reference_stats["mean"]):
                # Assume we need to flatten it like in the prediction code
                features = np.concatenate([new_data.iloc[:, 0], new_data.iloc[:, 1]], axis=0).reshape(1, -1)
                new_data = pd.DataFrame(features)
            
            # Check how many features are outside 3 standard deviations
            mean = np.array(self.reference_stats["mean"])
            std = np.array(self.reference_stats["std"])
            
            values = new_data.values.flatten()
            outliers = np.abs(values - mean) > 3 * std
            
            drift_metrics["percent_outliers"] = float(np.mean(outliers) * 100)
            drift_metrics["num_outliers"] = int(np.sum(outliers))
            drift_metrics["total_features"] = len(outliers)
            
            # Drift detected if more than 5% of features are outliers
            drift_detected = drift_metrics["percent_outliers"] > 5
        else:
            # For multiple samples, use KS test to compare distributions
            reference_values = np.array(self.reference_stats["mean"])
            new_values = new_data.mean().values
            
            # Perform KS test
            ks_statistic, p_value = ks_2samp(reference_values, new_values)
            
            drift_metrics["ks_statistic"] = float(ks_statistic)
            drift_metrics["p_value"] = float(p_value)
            
            # Drift detected if p-value is below threshold
            drift_detected = p_value < self.drift_threshold
        
        drift_metrics["drift_detected"] = drift_detected
        return drift_metrics

# Example usage
if __name__ == "__main__":
    monitor = DataDriftMonitor()
    monitor.generate_reference_stats()