import joblib
import pandas as pd
from backend.ml_models.config import CONFIG
from sklearn import svm

def load_model():
    return joblib.load(CONFIG["model_path"])

def predict_from_df(df):# still have to figure out how to pass df data to the predict function call
    MPP_dict=load_model()
    pred_probs={}
    for i in MPP_dict.keys():
        model=MPP_dict[i]
        pred_probs[i]=model.predict_proba(df)[:,1][0]
    
    return pred_probs
