import os
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np


#Mlflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:7001")

MODEL_URI = "models:/IrisRandomForestModel@champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# MODEL_PATH = Path("models/model.pkl")
# model = joblib.load(MODEL_PATH)
try:
    #Load the champion model at the application startup
    model = mlflow.sklearn.load_model(MODEL_URI)
    client = mlflow.tracking.MlflowClient()
    champion_version = client.get_model_version_by_alias(name="IrisRandomForestModel", alias="champion")

    model_version = champion_version.version
    print(f"Loaded champion model version: {model_version} from MLflow Model Registry")
except Exception as e:
    print(f"Error loading champion model: {e}")

app = FastAPI(title="Iris Classifier API", description="API for predicting iris species using a trained Random Forest model.")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)
@app.get("/")
def health_check():
    return {"status": "Ready to serve"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = int(model.predict(features)[0])
    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }