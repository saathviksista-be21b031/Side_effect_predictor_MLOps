import io
import traceback
import yaml
import pickle
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from backend.ml_models import predict_AMPP, train_AMPP, predict_SE
from backend.ml_models.config import CONFIG


# Adjust this path to your project root
PROJECT_ROOT = Path("/home/saathvik/project_playing_around")

app = FastAPI()

# Enable CORS so React (localhost:3000) can talk to FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Mount the static directory at /pipeline
app.mount("/pipeline", StaticFiles(directory=PROJECT_ROOT / "backend" / "static"), name="pipeline")

# Instrument all endpoints with Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/predict_AMPP")
async def predict(file: UploadFile = File(...)):
    """
    Upload a CSV of FVA bounds (no header). Runs drift monitoring and predicts top-5 side-effects.
    """
    try:
        # 1. Read uploaded CSV and reshape into prediction input format
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None)
        pred_df = pd.concat([df.iloc[:, 0], df.iloc[:, 1]], axis=0, ignore_index=True).to_frame().T

        # 2. Save as inputted_features.pkl
        feats_path = PROJECT_ROOT / "backend" / "target" / "inputted_features_2.pkl"
        with open(feats_path, "wb") as f:
            pickle.dump(pred_df.values, f)

        # 3. Run DVC stage: predict_AMPP (includes drift check + predict)
        result = subprocess.run(
            ["dvc", "repro", "predict_AMPP"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            return {"error": result.stderr}

        # 4. Load predictions
        out_path = PROJECT_ROOT / "backend" / "target" / "ampp_predictions.pkl"
        with open(out_path, "rb") as f:
            pred_probs = pickle.load(f)

        # 5. Return top-5 predictions
        top_5 = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        response = "The 5 most probable side effects are:\n"
        response += "\n".join(se for se, _ in top_5)
        response += "\n(you may wish to translate CIDs to descriptions)"
        return PlainTextResponse(response)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {"error": str(e), "traceback": tb}



@app.post("/train_AMPP")
async def train():
    """
    Re-run the DVC train_AMPP pipeline stage to retrain the full AMPP model.
    """
    try:
        result = subprocess.run(
            ["dvc", "repro", "train_AMPP"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            return {"error": result.stderr}
        return {"message": "Training complete"}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/predict_se")
async def predict_se(
    side_effect: str = Form(...),
    file: UploadFile = File(...)
):
    """
    For the specified side-effect, train (or re-train) its binary classifier via DVC,
    then return the probability that the uploaded drug causes it.
    """
    try:
        # 1. Write the requested SE into params.yaml
        params = {"SE": side_effect}
        params_path = PROJECT_ROOT / "backend" / "target" / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f)

        # 2. Read & pickle the uploaded features
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None)
        features = np.concatenate([df.iloc[:, 0], df.iloc[:, 1]], axis=0).reshape(1, -1)
        feats_path = PROJECT_ROOT / "backend" / "target" / "inputted_features.pkl"
        with open(feats_path, "wb") as f:
            pickle.dump(features, f)

        # 3. Reproduce the DVC predict_SE stage
        result = subprocess.run(
            ["dvc", "repro", "predict_SE"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            return {"error": result.stderr}

        # 4. Load and return the prediction result
        out_path = PROJECT_ROOT / "backend" / "target" / "output.pkl"
        with open(out_path, "rb") as f:
            prediction = pickle.load(f)

        return PlainTextResponse(str(prediction))

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {"error": str(e), "traceback": tb}



