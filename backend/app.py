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

# Serve any files under backend/target via /pipeline/static/*
app.mount(
    "/pipeline/static",
    StaticFiles(directory=PROJECT_ROOT / "backend" / "target"),
    name="pipeline_static",
)

# Instrument all endpoints with Prometheus metrics
Instrumentator().instrument(app).expose(app)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Given a CSV of FVA bounds (no header), return the top-5 most probable side-effects.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None)
        pred_df = pd.concat([df.iloc[:, 0], df.iloc[:, 1]], axis=0, ignore_index=True).to_frame().T

        pred_probs = predict_AMPP.predict_from_df(pred_df)
        top_5 = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)[:5]

        response = "The 5 most probable side effects are:\n"
        response += "\n".join(f"{se}: {score:.4f}" for se, score in top_5)
        response += "\n(you may wish to translate CIDs to descriptions)"
        return PlainTextResponse(response)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {"error": str(e), "traceback": tb}


@app.post("/train")
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



