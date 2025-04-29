import numpy as np
import pandas as pd
from sklearn import svm
import random
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



def training_function(SE,SE_drugs,all_drugs,features_dir,random_state=42,n_outer=10):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("Side_Effect_Predictor")
    with mlflow.start_run(run_name='retraining AMPP'):
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_outer", n_outer)
        mlflow.log_param("kernel", "linear")
        mlflow.log_param("C", 0.1)
        np.random.seed(random_state)
        #auc_SE=np.zeros((len(SE_drugs),n_outer))
        drugs_SE=None
        auc_drug_dict_mean={}
        auc_drug_dict_std={}
        auc_all_runs=[]
        fpr_list=[]
        tpr_list=[]
        for i in range(n_outer):
            print(i)
            n_pos=len(SE_drugs)
            n_neg=n_pos
            pos_SE_drugs = set(SE_drugs)
            eligible = [item for item in all_drugs if item not in pos_SE_drugs]
            pos_samples=SE_drugs
            scores_for_SE_score=[]
            scores_for_SE_true=[]
            #print(eligible)
            #print(n_neg)
            try:
                
                neg_samples=random.sample(eligible,n_neg)#sampling negative samples to train on
            except:
                
                pos_samples=random.sample(SE_drugs,len(eligible))
                neg_samples=random.sample(eligible,len(eligible))
                n_pos=len(pos_samples)
                n_neg=len(neg_samples)
            
            for j in range(0,len(pos_samples)):
                #j'th drug will be left out
                #constructing test/leave out sets
                drug_pos_test=pos_samples[j]
                drug_neg_test=neg_samples[j]#randomly selecting leave out for negative
                
                drug_pos_test_df=pd.read_csv(f'{features_dir}fva_bounds_{drug_pos_test}.csv',header=None)
                drug_pos_test_pd=pd.concat([drug_pos_test_df[0],drug_pos_test_df[1]],axis=0,ignore_index=True)
                drug_neg_test_df=pd.read_csv(f'{features_dir}fva_bounds_{drug_neg_test}.csv',header=None)
                drug_neg_test_pd=pd.concat([drug_neg_test_df[0],drug_neg_test_df[1]],axis=0,ignore_index=True)
                X_test=np.array([drug_pos_test_pd,drug_neg_test_pd])
                Y_test=np.array([1,0])
                #constructing the training set
                X_train=np.zeros((2*(n_pos),4732))
                Y_train=np.zeros((2*(n_pos),1))
                for k_pos in range(0,n_pos):
                    
                    drug_pos_train_df=pd.read_csv(f'{features_dir}fva_bounds_{pos_samples[k_pos]}.csv',header=None)
                    drug_pos_train_pd=pd.concat([drug_pos_train_df[0],drug_pos_train_df[1]],axis=0,ignore_index=True)
                    #print(f'changing {k_pos} elements')
                    X_train[k_pos]=drug_pos_train_pd
                    Y_train[k_pos]=1

                for k_neg in range(0,n_pos):
                    
                    drug_neg_train_df=pd.read_csv(f'{features_dir}fva_bounds_{neg_samples[k_neg]}.csv',header=None)
                    drug_neg_train_pd=pd.concat([drug_neg_train_df[0],drug_neg_train_df[1]],axis=0,ignore_index=True)
                    X_train[n_pos + k_neg]=drug_neg_train_pd
                    #print(f'changed {(n_pos-1) + k_neg} element')
                    Y_train[n_pos + k_neg]=0
                #training set has been constructed
                X_train_master=X_train
                Y_train_master=Y_train
                X_train=np.delete(X_train,n_pos+j,0)
                Y_train=np.delete(Y_train,n_pos+j,0)
                X_train=np.delete(X_train,j,0)
                Y_train=np.delete(Y_train,j,0)
                #print(f'leaving out {SE_drugs[j]}')
                Y_train=list(Y_train)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                
                clf = svm.SVC(kernel='linear',probability=True,random_state=random_state,C=10)
                #clf = RandomForestClassifier(n_estimators=1000, random_state=42)
                clf.fit(X_train_scaled,np.ravel(Y_train))
                
                probas = clf.predict_proba(X_test_scaled)[:,1]
                probabs=clf.predict_proba(X_test_scaled)
                
                for u in range(0,len(Y_test)):
                    scores_for_SE_score.append(probas[u])
                    scores_for_SE_true.append(Y_test[u])

            fpr,tpr,thresholds=roc_curve(scores_for_SE_true,scores_for_SE_score)
            #print(fpr,tpr)
            roc_auc=auc(fpr,tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_all_runs.append(roc_auc)
        #training and saving MPP to a dict file
        clf = svm.SVC(kernel='linear',probability=True,random_state=random_state)
        clf.fit(X_train_master,np.ravel(Y_train_master))
        
        print(f'Trained MPP for {SE}')
        mlflow.log_metric("mean_auc", max(auc_all_runs))
        mlflow.set_tag("Training Call", "Retraining AMPP")
        signature = infer_signature(X_train_master, clf.predict_proba(X_train_master))
        
        model_info = mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        signature=signature,
        input_example=X_train_master,
        registered_model_name="tracking-retraining AMPP",
    )
        plt.figure()
        plt.plot(fpr_list[auc_all_runs.index(max(auc_all_runs))],tpr_list[auc_all_runs.index(max(auc_all_runs))], label=f'ROC curve (area = {max(auc_all_runs):.2f})')
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
    
    
    #return fpr_list[auc_all_runs.index(max(auc_all_runs))],tpr_list[auc_all_runs.index(max(auc_all_runs))],max(auc_all_runs),np.mean(auc_all_runs),np.std(auc_all_runs),fpr,tpr
    return clf,max(auc_all_runs)

def train_AMPP():
    MPP_dict={}
    df_all_drugs=pd.read_csv(CONFIG["training_ref_file_2"])
    all_drugs=df_all_drugs['dg_id']
    df_SE=pd.read_csv(CONFIG["training_ref_file_1"])
    
    for se in range(0,len(df_SE['concept_id'])):
        '''if se>10:
            continue'''
        if (df_SE['concept_id'][se]!='C0000731'):
            continue
        SE_drugs=df_SE['dg_id'][se].split(',')
        for i in range(0,len(SE_drugs)):
            SE_drugs[i]=SE_drugs[i].lstrip()
        model,auc=training_function(df_SE['concept_id'][se],SE_drugs,all_drugs,CONFIG["training_data_path"])
        MPP_dict[se]=model
    with open(CONFIG["model_path_temp"], 'wb') as file:
        pickle.dump(MPP_dict, file)

