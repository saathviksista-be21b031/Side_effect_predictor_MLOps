# backend/ml_models/predict_AMPP.py

import joblib
import pandas as pd
import pickle
import numpy as np
from backend.ml_models.config import CONFIG
from sklearn import svm
from backend.ml_models.drift_monitor import monitor_and_react

def load_model():
    return joblib.load(CONFIG["model_path"])

def predict_from_df(df):
    monitor_and_react(df, load_baseline_for_monitoring())
    MPP_dict = load_model()
    #print(MPP_dict)
    pred_probs = {}
    for cid, model in MPP_dict.items():
        #print(cid)
        #print(model)

        pred_probs[cid] = model.predict_proba(df)[:, 1][0]

    #print(pred_probs)
    return pred_probs

def load_baseline_for_monitoring():
    with open("backend/target/temp_targets/X_train_master.pkl", "rb") as f:
        data = pickle.load(f)
    if hasattr(data, 'values'):
        data = data.values
    return data


if __name__ == "__main__":
    INPUT_PATH = "backend/target/inputted_features.pkl"
    OUTPUT_PATH = "backend/target/ampp_predictions.pkl"

    with open(INPUT_PATH, "rb") as f:
        df = pickle.load(f)
    if hasattr(df, 'values'):
        df = df.values

    preds = predict_from_df(df)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(preds, f)


