from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel

app = FastAPI(title="Diabetes Risk Predictor API")


mlflow.set_tracking_uri("http://mlflow:5000")


try:
    model = mlflow.sklearn.load_model(
        "models:/Diabetes_Risk_Classifier/Production"
    )
    print("Model loaded successfully!")

except Exception as e:
    model = None
    print(f"Error loading model: {type(e).__name__}: {e}")


try:
    imputer = joblib.load("models/imputer.joblib")
    input_columns = joblib.load("models/input_columns.joblib")
    outlier_bounds = joblib.load("models/outlier_bounds.joblib")
    scaler = joblib.load("models/scaler.joblib")

    print("Preprocessing objects loaded successfully!")

except Exception as e:
    imputer = None
    input_columns = None
    outlier_bounds = None
    scaler = None
    print(f"Error loading preprocessing objects: {type(e).__name__}: {e}")


class InputData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Risk Predictor API"}


@app.post("/predict")
def predict_diabetes_risk(input_data: InputData):
    try:
        input_df = pd.DataFrame([input_data.model_dump()])

        input_df = input_df[input_columns]

        cols_with_zeros = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI"
        ]

        input_df[cols_with_zeros] = input_df[cols_with_zeros].replace(
            0, np.nan
        )

        input_df = pd.DataFrame(
            imputer.transform(input_df),
            columns=input_columns
        )

        input_df = input_df.clip(
            lower=outlier_bounds["lower"],
            upper=outlier_bounds["upper"],
            axis=1
        )

        input_df["Insulin_Glucose_Ratio"] = (
            input_df["Insulin"] / input_df["Glucose"].replace(0, 1)
        )

        input_df["BMI-Category"] = input_df["BMI"].apply(
            lambda x: 1 if 18.5 <= x < 25
            else (2 if 25 <= x < 30 else 3)
        )

        input_df = input_df.reindex(
            columns=scaler.feature_names_in_
        )

        scaled_data = scaler.transform(input_df)

        scaled_df = pd.DataFrame(
            scaled_data,
            columns=scaler.feature_names_in_
        )

        prediction = model.predict(scaled_df)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {e}"
        )
