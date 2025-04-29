import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from backend.ml_models import predict_AMPP, train_AMPP, predict_SE
from backend.ml_models.config import CONFIG
import io
import traceback
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the CSV file content (with no header)
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None)  # Specify header=None to avoid treating first row as column names
        # Concatenate the first two columns (if needed, based on your logic)
        pred_df = pd.concat([df.iloc[:, 0], df.iloc[:, 1]], axis=0, ignore_index=True).to_frame()
        # Predict the probabilities using the model
        pred_probs = predict_AMPP.predict_from_df(pred_df.T)
        top_5 = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        answer='The 5 most probable side effects are:\n'
        # Print the top 5 side effects with their scores
        for side_effect, score in top_5:
            answer=answer + f"{side_effect}\n"

        # Return the dictionary itself
        
        return PlainTextResponse(answer+'need to translate CIDs to desc')

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # Printing the full traceback in the server log for easier debugging
        return {"error": str(e), "traceback": tb}



@app.post("/train")
async def train():
    """
    Endpoint to trigger training directly from the function.
    """
    try:
        # Call the training function directly from train_AMPP.py
        train_AMPP.train_AMPP()
        return {"message": "Training complete"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_se")
async def predict_se(
    side_effect: str = Form(...),
    file: UploadFile = File(...)
):
    """
    For the specified side-effect, train (or re-train) its binary classifier
    and return the probability that the uploaded drug causes it.
    """
    try:
        # 1. Read & flatten the uploaded CSV into a 1×N feature array
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None)
        features = np.concatenate([df.iloc[:, 0], df.iloc[:, 1]], axis=0).reshape(1, -1)

        # 2. Call your predict_SE.predict_function
        proba = predict_SE.predict_function(side_effect, features)

        # proba is an array of shape (1, 2): [:,1] is P(class=1)
        prob_side_effect = proba[:, 1][0]

        return PlainTextResponse(
            f"Probability of side-effect '{side_effect}': {prob_side_effect:.4f}"
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # Printing the full traceback in the server log for easier debugging
        return {"error": str(e), "traceback": tb}
# Instrumentator setup
Instrumentator().instrument(app).expose(app)