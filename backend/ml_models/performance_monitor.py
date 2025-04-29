# backend/ml_models/performance_monitor.py
import pandas as pd
import numpy as np
import json
import os
import mlflow
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from backend.ml_models.config import CONFIG

class PerformanceMonitor:
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics_log_path = "backend/data/performance_metrics.json"
        self.metrics_history = self._load_metrics_history()
        self.current_metrics = {}
        
    def _load_metrics_history(self):
        """Load metrics history if exists"""
        if os.path.exists(self.metrics_log_path):
            with open(self.metrics_log_path, 'r') as f:
                return json.load(f)
        return {"timestamps": [], "metrics": []}
    
    def _save_metrics_history(self):
        """Save metrics history to file"""
        with open(self.metrics_log_path, 'w') as f:
            json.dump(self.metrics_history, f)
    
    def log_prediction(self, side_effect_id, true_label=None, predicted_prob=None):
        """Log a prediction for performance tracking"""
        timestamp = datetime.now().isoformat()
        
        # Create entry for this side effect if it doesn't exist
        if side_effect_id not in self.current_metrics:
            self.current_metrics[side_effect_id] = {
                "predictions": [],
                "true_labels": [],
                "timestamps": []
            }
        
        # Store prediction details
        if predicted_prob is not None:
            self.current_metrics[side_effect_id]["predictions"].append(float(predicted_prob))
            self.current_metrics[side_effect_id]["timestamps"].append(timestamp)
            
            # If we have ground truth, store it too
            if true_label is not None:
                self.current_metrics[side_effect_id]["true_labels"].append(int(true_label))
    
    def calculate_metrics(self, side_effect_id=None):
        """Calculate performance metrics for logged predictions"""
        metrics = {}
        
        # If side_effect_id specified, calculate metrics only for that model
        models_to_check = [side_effect_id] if side_effect_id else self.current_metrics.keys()
        
        for model_id in models_to_check:
            if model_id not in self.current_metrics:
                continue
                
            model_data = self.current_metrics[model_id]
            
            # Only calculate metrics if we have true labels
            if len(model_data["true_labels"]) > 0:
                y_true = np.array(model_data["true_labels"])
                y_pred = np.array(model_data["predictions"])
                
                # Calculate ROC AUC if we have both positive and negative samples
                if len(np.unique(y_true)) > 1:
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    roc_auc = auc(fpr, tpr)
                    
                    # Calculate precision-recall metrics
                    precision, recall, _ = precision_recall_curve(y_true, y_pred)
                    pr_auc = average_precision_score(y_true, y_pred)
                    
                    metrics[model_id] = {
                        "roc_auc": float(roc_auc),
                        "pr_auc": float(pr_auc),
                        "sample_count": len(y_true),
                        "positive_count": int(np.sum(y_true)),
                        "timestamp": datetime.now().isoformat()
                    }
        
        return metrics
    
    def update_metrics_history(self):
        """Update metrics history with current metrics"""
        metrics = self.calculate_metrics()
        if metrics:
            self.metrics_history["timestamps"].append(datetime.now().isoformat())
            self.metrics_history["metrics"].append(metrics)
            self._save_metrics_history()
            
            # Log to MLflow
            mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
            with mlflow.start_run(run_name="performance_metrics_update"):
                for model_id, model_metrics in metrics.items():
                    for metric_name, metric_value in model_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(f"{model_id}_{metric_name}", metric_value)
                
                # Log the metrics file as an artifact
                mlflow.log_artifact(self.metrics_log_path)
    
    def check_performance_degradation(self, side_effect_id=None, threshold=0.05):
        """
        Check if model performance has degraded
        Returns dict with degradation metrics and boolean indicating if degradation detected
        """
        if len(self.metrics_history["metrics"]) < 2:
            return {"degradation_detected": False, "message": "Insufficient history"}
        
        # Get the most recent and previous metrics
        current = self.metrics_history["metrics"][-1]
        previous = self.metrics_history["metrics"][-2]
        
        degradation_results = {}
        
        # If side_effect_id specified, check only that model
        models_to_check = [side_effect_id] if side_effect_id else set(current.keys()).intersection(previous.keys())
        
        for model_id in models_to_check:
            if model_id in current and model_id in previous:
                current_auc = current[model_id].get("roc_auc")
                previous_auc = previous[model_id].get("roc_auc")
                
                if current_auc is not None and previous_auc is not None:
                    degradation = previous_auc - current_auc
                    degradation_percent = (degradation / previous_auc) * 100 if previous_auc > 0 else 0
                    
                    degradation_results[model_id] = {
                        "degradation": float(degradation),
                        "degradation_percent": float(degradation_percent),
                        "degradation_detected": degradation > threshold
                    }
        
        # Overall degradation detected if any model has degraded
        overall_degradation = any(result["degradation_detected"] for result in degradation_results.values())
        
        return {
            "degradation_detected": overall_degradation,
            "model_details": degradation_results
        }