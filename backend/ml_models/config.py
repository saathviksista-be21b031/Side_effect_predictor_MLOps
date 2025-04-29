# Update config.py with new settings
CONFIG = {
    "model_path": "backend/model/AMPP.pkl",
    "model_path_temp": "backend/model/AMPP_temp.pkl",
    "training_data_path": "backend/data/FVA_Bounds/",
    "training_ref_file_1": "backend/data/side_effects_vs_causative_drugs_meta.csv",
    "training_ref_file_2": "backend/data/drug_gene_targets_meta_1.csv",
    "random_seed": 42,
    "target_column": "label",  # change to your actual target column name
    
    # Monitoring settings
    "reference_data_path": "backend/data/reference_stats.json",
    "performance_metrics_path": "backend/data/performance_metrics.json",
    "retraining_log_path": "backend/data/retraining_log.json",
    "drift_threshold": 0.05,
    "performance_degradation_threshold": 0.1,
    "retraining_cooldown_days": 7,
    
    # MLflow settings
    "mlflow_tracking_uri": "http://127.0.0.1:5000",
    "mlflow_experiment_name": "Side_Effect_Predictor",
}