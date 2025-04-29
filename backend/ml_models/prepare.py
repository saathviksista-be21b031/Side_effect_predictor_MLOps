import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pickle
import os
from backend.ml_models.config import CONFIG
import yaml

def prepare_data(SE, SE_drugs, all_drugs, features_dir, random_state=42, n_outer=1):
    """
    Prepare training and test data for side effect prediction model.
    Saves the prepared data and evaluation metrics to pickle files.
    
    Args:
        SE (str): Side effect name
        SE_drugs (list): List of drugs known to cause the side effect
        all_drugs (list): List of all drugs
        features_dir (str): Directory containing feature files
        random_state (int): Random seed for reproducibility
        n_outer (int): Number of outer cross-validation runs
    
    Returns:
        tuple: Paths to saved data files
    """
    # Create model directory if it doesn't exist
    model_dir = 'backend/target/temp_targets/'
    #os.makedirs(model_dir, exist_ok=True)
    
    np.random.seed(random_state)
    auc_all_runs = []
    fpr_list = []
    tpr_list = []
    
    # Master training data variables to be populated
    X_train_master = None
    Y_train_master = None
    
    for i in range(n_outer):
        print(f"Outer loop iteration: {i}")
        n_pos = len(SE_drugs)
        n_neg = n_pos
        pos_SE_drugs = set(SE_drugs)
        eligible = [item for item in all_drugs if item not in pos_SE_drugs]
        pos_samples = SE_drugs
        scores_for_SE_score = []
        scores_for_SE_true = []
        
        try:
            neg_samples = random.sample(eligible, n_neg)  # sampling negative samples to train on
        except:
            pos_samples = random.sample(SE_drugs, len(eligible))
            neg_samples = random.sample(eligible, len(eligible))
            n_pos = len(pos_samples)
            n_neg = len(neg_samples)
        
        for j in range(0, len(pos_samples)):
            # j'th drug will be left out
            # constructing test/leave out sets
            drug_pos_test = pos_samples[j]
            drug_neg_test = neg_samples[j]  # randomly selecting leave out for negative
            
            drug_pos_test_df = pd.read_csv(f'{features_dir}fva_bounds_{drug_pos_test}.csv', header=None)
            drug_pos_test_pd = pd.concat([drug_pos_test_df[0], drug_pos_test_df[1]], axis=0, ignore_index=True)
            drug_neg_test_df = pd.read_csv(f'{features_dir}fva_bounds_{drug_neg_test}.csv', header=None)
            drug_neg_test_pd = pd.concat([drug_neg_test_df[0], drug_neg_test_df[1]], axis=0, ignore_index=True)
            X_test = np.array([drug_pos_test_pd, drug_neg_test_pd])
            Y_test = np.array([1, 0])
            
            # constructing the training set
            X_train = np.zeros((2 * (n_pos), 4732))
            Y_train = np.zeros((2 * (n_pos), 1))
            
            for k_pos in range(0, n_pos):
                drug_pos_train_df = pd.read_csv(f'{features_dir}fva_bounds_{pos_samples[k_pos]}.csv', header=None)
                drug_pos_train_pd = pd.concat([drug_pos_train_df[0], drug_pos_train_df[1]], axis=0, ignore_index=True)
                X_train[k_pos] = drug_pos_train_pd
                Y_train[k_pos] = 1

            for k_neg in range(0, n_pos):
                drug_neg_train_df = pd.read_csv(f'{features_dir}fva_bounds_{neg_samples[k_neg]}.csv', header=None)
                drug_neg_train_pd = pd.concat([drug_neg_train_df[0], drug_neg_train_df[1]], axis=0, ignore_index=True)
                X_train[n_pos + k_neg] = drug_neg_train_pd
                Y_train[n_pos + k_neg] = 0
            
            # training set has been constructed
            # Store master data for later use in full model training
            if X_train_master is None:
                X_train_master = X_train.copy()
                Y_train_master = Y_train.copy()
            
            # Remove leave-out samples
            X_train_cv = np.delete(X_train, n_pos + j, 0)
            Y_train_cv = np.delete(Y_train, n_pos + j, 0)
            X_train_cv = np.delete(X_train_cv, j, 0)
            Y_train_cv = np.delete(Y_train_cv, j, 0)
            
            Y_train_cv = list(Y_train_cv)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_cv)
            X_test_scaled = scaler.transform(X_test)
            
            # This is where we would train the model, but this part will be moved to predict_SE.py
            # For validation purposes, we're using a placeholder SVC model to get probabilities
            from sklearn import svm
            temp_clf = svm.SVC(kernel='linear', probability=True, random_state=random_state, C=0.1)
            temp_clf.fit(X_train_scaled, np.ravel(Y_train_cv))
            
            probas = temp_clf.predict_proba(X_test_scaled)[:, 1]
            
            for u in range(0, len(Y_test)):
                scores_for_SE_score.append(probas[u])
                scores_for_SE_true.append(Y_test[u])

        fpr, tpr, thresholds = roc_curve(scores_for_SE_true, scores_for_SE_score)
        roc_auc = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_all_runs.append(roc_auc)
    
    # Save all required data to pickle files
    data_paths = {}
    
    data_paths['X_train_master'] = os.path.join(model_dir, f'X_train_master.pkl')
    with open(data_paths['X_train_master'], 'wb') as f:
        pickle.dump(X_train_master, f)
    
    data_paths['Y_train_master'] = os.path.join(model_dir, f'Y_train_master.pkl')
    with open(data_paths['Y_train_master'], 'wb') as f:
        pickle.dump(Y_train_master, f)
    
    data_paths['auc_all_runs'] = os.path.join(model_dir, f'auc_all_runs.pkl')
    with open(data_paths['auc_all_runs'], 'wb') as f:
        pickle.dump(auc_all_runs, f)
    
    data_paths['fpr_list'] = os.path.join(model_dir, f'fpr_list.pkl')
    with open(data_paths['fpr_list'], 'wb') as f:
        pickle.dump(fpr_list, f)
    
    data_paths['tpr_list'] = os.path.join(model_dir, f'tpr_list.pkl')
    with open(data_paths['tpr_list'], 'wb') as f:
        pickle.dump(tpr_list, f)
    
    print(f'Data preparation completed for {SE}')
    '''print(f'Best AUC: {max(auc_all_runs)}')'''
    
    return data_paths

if __name__ == "__main__":
    # Example of how to use the function
    df_all_drugs = pd.read_csv(CONFIG["training_ref_file_2"])
    all_drugs = df_all_drugs['dg_id']
    df_SE = pd.read_csv(CONFIG["training_ref_file_1"])
    with open("backend/target/params.yaml") as f:
        params = yaml.safe_load(f)
    
    # Example for a specific side effect
    SE = params["SE"]  # Replace with actual side effect name
    row_idx = df_SE.index[df_SE[df_SE.columns[0]] == SE][0]
    SE_drugs = df_SE['dg_id'][row_idx].split(',')
    for i in range(0, len(SE_drugs)):
        SE_drugs[i] = SE_drugs[i].lstrip()
    
    prepare_data(SE, SE_drugs, all_drugs, CONFIG["training_data_path"])