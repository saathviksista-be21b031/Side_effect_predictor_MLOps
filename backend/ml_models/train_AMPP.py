import numpy as np
import pandas as pd
from sklearn import svm
import random
import yaml
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from backend.ml_models.config import CONFIG
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from backend.ml_models.prepare import prepare_data


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
    
    with mlflow.start_run(run_name=f'retraining AMPP'):
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



if __name__ == "__main__":
    MPP_dict={}
    df_all_drugs=pd.read_csv(CONFIG["training_ref_file_2"])
    all_drugs=df_all_drugs['dg_id']
    df_SE=pd.read_csv(CONFIG["training_ref_file_1"])
    
    for se in range(0,len(df_SE['concept_id'])):
        #if se>2:
          #  continue
        if (df_SE['concept_id'][se]!='C0000731'):
           continue
        SE_drugs=df_SE['dg_id'][se].split(',')
        for i in range(0,len(SE_drugs)):
            SE_drugs[i]=SE_drugs[i].lstrip()
        
        SE=df_SE['concept_id'][se]
        prepare_data(SE,SE_drugs,all_drugs,CONFIG["training_data_path"])
        model,auc=train_model(SE)
        MPP_dict[se]=model
    with open(CONFIG["model_path_temp"], 'wb') as file:
        pickle.dump(MPP_dict, file)

