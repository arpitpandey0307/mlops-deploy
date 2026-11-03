import os
from pathlib import Path
import joblib
from fastapi import FastAPI,HTTPException, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np
from contextlib import asynccontextmanager
from typing import Callable


#Mlflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:7001")

MODEL_URI = "models:/IrisRandomForestModel@champion"

def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    return model

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

def create_app(
        model_loader: Callable=load_mlflow_model,
        version_loader: Callable = None):

    if version_loader is None:
        def version_loader():
            client = mlflow.tracking.MlflowClient(
                tracking_uri=MLFLOW_TRACKING_URI
            )
            champion_version = client.get_model_version_by_alias(name="IrisRandomForestModel", alias="champion")
            return champion_version.version
        
    @asynccontextmanager
    def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            print(f"Error loading model: {ex}")
            raise RuntimeError("Unable to load the champion model")
        
        # client = mlflow.tracking.MlflowClient(
        #     tracking_uri=MLFLOW_TRACKING_URI
        #     )
        # champion_version = client.get_model_version_by_alias(name="IrisRandomForestModel", alias="champion")
        
        app.state.model_version = version_loader()
        yield

        app = FastAPI(
        title="Iris Classifiaction API",
        description="Iris prediction API powered by MLFlow",
        version="3.0.0",
        lifespan=lifespan
    )

        @app.get("/")
        def health_check():
            return {"status": "Ready to serve"}

        @app.get("/health")
        def health_check():
            return {"status": "healthy"}

        @app.post("/predict")
        def predict(data: IrisInput, request: Request):
            model = request.app.state.model
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

app = create_app()

    #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# MODEL_PATH = Path("models/model.pkl")
# model = joblib.load(MODEL_PATH)
# try:
#     #Load the champion model at the application startup
#     model = mlflow.sklearn.load_model(MODEL_URI)
#     client = mlflow.tracking.MlflowClient()
#     champion_version = client.get_model_version_by_alias(name="IrisRandomForestModel", alias="champion")

#     model_version = champion_version.version
#     print(f"Loaded champion model version: {model_version} from MLflow Model Registry")
# except Exception as e:
#     print(f"Error loading champion model: {e}")
#     try:
#         model = joblib.load(Path("models/model.pkl"))
#         print("Loaded local model from models/model.pkl")
#     except Exception as fallback_error:
#         raise RuntimeError("Unable to load the champion or local model") from fallback_error

# app = FastAPI(title="Iris Classifier API", description="API for predicting iris species using a trained Random Forest model.")
