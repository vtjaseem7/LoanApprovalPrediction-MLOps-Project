import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException

from api.schemas import (
    LoanRequest,
    PredictionResponse
)
from datetime import datetime

from src.predict import make_predict

from src.preprocessing import (
    preprocess_input
)
from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    COLUMNS_PATH
)

app = FastAPI(
    title="Loan Approval Prediction API",
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

training_columns = joblib.load(
    COLUMNS_PATH
)


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse
)
def predict(data: LoanRequest):

    try:

        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        input_df = preprocess_input(
            input_df,
            scaler,
            training_columns
        )

        prediction = make_predict(
            model,
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        log_file = "logs/predictions.csv"

        os.makedirs("logs", exist_ok=True)

        log_data = {
            "timestamp": datetime.now(),
            "prediction": int(prediction),
            "probability": float(probability),
            "result": (
                "Approved"
                if prediction == 1
                else "Rejected"
            )
        }

        if os.path.exists(log_file):

            pd.DataFrame([log_data]).to_csv(
                log_file,
                mode="a",
                header=False,
                index=False
            )

        else:

            pd.DataFrame([log_data]).to_csv(
                log_file,
                index=False
            )


        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            result=(
                "Approved"
                if prediction == 1
                else "Rejected"
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )