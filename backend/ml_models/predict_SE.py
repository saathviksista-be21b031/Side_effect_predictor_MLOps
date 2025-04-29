import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import yaml
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from backend.ml_models.config import CONFIG

def load_prepared_data(SE, data_dir='backend/target/temp_targets/'):
    
    with open(os.path.join(data_dir, f'X_train_master.pkl'), 'rb') as f:
        X_train_master = pickle.load(f)
    
    with open(os.path.join(data_dir, f'Y_train_master.pkl'), 'rb') as f:
        Y_train_master = pickle.load(f)
    
    with open(os.path.join(data_dir, f'auc_all_runs.pkl'), 'rb') as f:
        auc_all_runs = pickle.load(f)
    
    with open(os.path.join(data_dir, f'fpr_list.pkl'), 'rb') as f:
        fpr_list = pickle.load(f)
    
    with open(os.path.join(data_dir, f'tpr_list.pkl'), 'rb') as f:
        tpr_list = pickle.load(f)
    
    return X_train_master, Y_train_master, auc_all_runs, fpr_list, tpr_list

def train_model(SE, data_dir='backend/target/temp_targets/', random_state=42, mlflow_tracking_uri="http://127.0.0.1:5000"):
    
    
    # Set up MLflow tracking
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    mlflow.set_experiment("Side_Effect_Predictor")
    
    # Load prepared data
    X_train_master, Y_train_master, auc_all_runs, fpr_list, tpr_list = load_prepared_data(SE, data_dir)
    
    with mlflow.start_run(run_name=f'training MPP for {SE}'):
        # Log parameters
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("kernel", "linear")
        mlflow.log_param("C", 0.1)
        
        # Train the model on the full dataset
        clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)
        clf.fit(X_train_master, np.ravel(Y_train_master))
        
        print(f'Trained MPP for {SE}')
        
        # Log metrics
        mlflow.log_metric("mean_auc", max(auc_all_runs))
        mlflow.set_tag("Training Call", f"MPP for {SE}")
        
        # Log model
        signature = infer_signature(X_train_master, clf.predict_proba(X_train_master))
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X_train_master,
            registered_model_name=f"MPP: {SE}",
        )
        
        # Create and log ROC curve plot
        plt.figure()
        best_idx = auc_all_runs.index(max(auc_all_runs))
        plt.plot(fpr_list[best_idx], tpr_list[best_idx], 
                 label=f'ROC curve (area = {max(auc_all_runs):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        plot_path = f"roc_curve_{SE}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        # Clean up the plot file
        os.remove(plot_path)
        
        
    
    return clf, max(auc_all_runs)

def predict_function(SE, features, model_dir='backend/target/temp_targets/'):
    model,max_auc=train_model(SE)
    proba = model.predict_proba(features)
    return proba


if __name__ == "__main__":
    
    # Extract parameters with defaults
    random_state = 42
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    #get SE from params.yaml
    import yaml

    with open("backend/target/params.yaml") as f:
        params = yaml.safe_load(f)
    with open('backend/target/inputted_features.pkl','rb') as f:
        features=pickle.load(f)
    
    SE=params["SE"]
    proba=predict_function(SE,features)[:,1]
   

    with open('backend/target/output.pkl','wb') as f:
        pickle.dump(f"Probability of side-effect '{SE}': {proba.item():.4f}", f)
    print('done')

