# backend/ml_models/model_maintainer.py
import os
import time
import pickle
import shutil
import logging
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.ml_models.config import CONFIG
from backend.ml_models.drift_monitor import DataDriftMonitor
from backend.ml_models.performance_monitor import PerformanceMonitor
from backend.ml_models.train_AMPP import train_AMPP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend/logs/model_maintenance.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("model_maintenance")

class ModelMaintainer:
    def __init__(self):
        """Initialize model maintainer with monitors"""
        self.drift_monitor = DataDriftMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.retraining_threshold = 0.1  # Performance degradation threshold for retraining
        self.drift_threshold = 0.1  # Data drift threshold for retraining
        self.retraining_cooldown = timedelta(days=7)  # Minimum time between retraining
        self.last_retraining = None
        self.retraining_log_path = "backend/data/retraining_log.json"
        
        # Check if we need to initialize reference stats
        if not os.path.exists(self.drift_monitor.reference_data_path):
            logger.info("Initializing reference statistics...")
            self.drift_monitor.generate_reference_stats()
    
    def _load_retraining_log(self):
        """Load retraining log if exists"""
        if os.path.exists(self.retraining_log_path):
            import json
            with open(self.retraining_log_path, 'r') as f:
                log = json.load(f)
                if "last_retraining" in log:
                    self.last_retraining = datetime.fromisoformat(log["last_retraining"])
                return log
        return {"retraining_history": []}
    
    def _save_retraining_log(self, reason, metrics=None):
        """Save retraining log"""
        log = self._load_retraining_log()
        
        # Update last retraining timestamp
        self.last_retraining = datetime.now()
        log["last_retraining"] = self.last_retraining.isoformat()
        
        # Add entry to history
        entry = {
            "timestamp": self.last_retraining.isoformat(),
            "reason": reason
        }
        if metrics:
            entry["metrics"] = metrics
            
        log["retraining_history"].append(entry)
        
        import json
        with open(self.retraining_log_path, 'w') as f:
            json.dump(log, f)
    
    def should_retrain(self):
        """Check if model retraining is needed"""
        # Check retraining cooldown
        if self.last_retraining and (datetime.now() - self.last_retraining) < self.retraining_cooldown:
            logger.info(f"Skipping retraining check - within cooldown period ({self.retraining_cooldown})")
            return False, "Within cooldown period"
        
        # Check for performance degradation
        degradation_results = self.performance_monitor.check_performance_degradation(threshold=self.retraining_threshold)
        
        if degradation_results["degradation_detected"]:
            logger.warning(f"Performance degradation detected: {degradation_results}")
            return True, "Performance degradation detected"
        
        # We could also check for data drift here if we collect enough production data
        # This would require implementing drift detection across multiple samples
        
        # For now, return False - no retraining needed
        return False, "No issues detected"
    
    def check_and_retrain(self):
        """Check if retraining is needed and retrain if necessary"""
        logger.info("Checking if model retraining is needed...")
        
        # Load the retraining log
        log = self._load_retraining_log()
        if "last_retraining" in log:
            self.last_retraining = datetime.fromisoformat(log["last_retraining"])
        
        # Check if we should retrain
        should_retrain, reason = self.should_retrain()
        
        if should_retrain:
            logger.info(f"Retraining models... Reason: {reason}")
            
            # Backup current model
            if os.path.exists(CONFIG["model_path"]):
                backup_path = f"{CONFIG['model_path']}.backup.{int(time.time())}"
                shutil.copy2(CONFIG["model_path"], backup_path)
                logger.info(f"Backed up current model to {backup_path}")
            
            # Retrain models
            try:
                train_AMPP()
                
                # If successful, move temp model to production
                if os.path.exists(CONFIG["model_path_temp"]):
                    shutil.move(CONFIG["model_path_temp"], CONFIG["model_path"])
                    logger.info("Successfully retrained and deployed new model")
                    
                    # Log the retraining
                    self._save_retraining_log(reason)
                    
                    return True, "Retraining successful"
                else:
                    logger.error("Retraining failed - no temporary model file found")
                    return False, "Retraining failed - no model file generated"
            except Exception as e:
                logger.exception(f"Error during retraining: {e}")
                return False, f"Retraining error: {str(e)}"
        else:
            logger.info(f"No retraining needed. Reason: {reason}")
            return False, reason

# Function to run as a background task
def maintenance_job():
    """Run model maintenance checks and retraining"""
    maintainer = ModelMaintainer()
    
    while True:
        try:
            # Update performance metrics
            maintainer.performance_monitor.update_metrics_history()
            
            # Check and retrain if needed
            retrained, message = maintainer.check_and_retrain()
            if retrained:
                logger.info(f"Models retrained: {message}")
            else:
                logger.info(f"No retraining performed: {message}")
                
        except Exception as e:
            logger.exception(f"Error in maintenance job: {e}")
        
        # Sleep for 24 hours
        time.sleep(24 * 60 * 60)

# For testing
if __name__ == "__main__":
    maintainer = ModelMaintainer()
    maintainer.check_and_retrain()